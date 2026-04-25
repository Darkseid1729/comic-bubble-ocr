"""
detection.py - Shape Analysis & Bubble Type Detection Module
==============================================================
Implements advanced bubble detection:

    - Shape feature extraction (area, perimeter, circularity,
      aspect ratio, solidity, ellipse fitting)
    - Contour approximation (Douglas-Peucker algorithm)
    - Bubble type classification based on geometric features
    - Multi-criteria filtering logic
    - ROI extraction and reading order sorting

Bubble Types Detected:
    1. Elliptical speech bubbles   (smooth, high circularity)
    2. Spiky/shouting bubbles      (jagged edges, many vertices)
    3. Cloud-like thought bubbles  (irregular but rounded)
    4. Rectangular narration boxes (high solidity, ~90° corners)
"""

import cv2
import numpy as np
import math
from utils import logger, BUBBLE_COLORS, BUBBLE_TYPE_LABELS


# ═══════════════════════════════════════════════════════════════
# A. SHAPE FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
def compute_shape_features(contour: np.ndarray, gray_image: np.ndarray = None) -> dict:
    """
    Compute geometric features for a single contour.

    Features:
        - area: Contour area in pixels.
        - perimeter: Arc length of the contour.
        - circularity: C = 4πA / P² (1.0 = perfect circle).
        - aspect_ratio: Width / Height of bounding rectangle.
        - solidity: Area / ConvexHullArea (1.0 = fully convex).
        - extent: Area / BoundingRectArea (how much of bbox is filled).
        - num_vertices: Number of vertices after polygon approximation.
        - bbox: (x, y, w, h) bounding rectangle.
        - centroid: (cx, cy) center of mass.
        - ellipse: Fitted ellipse parameters (if >= 5 points).

    Parameters
    ----------
    contour : np.ndarray
        Single contour (array of 2D points).

    Returns
    -------
    dict
        Feature dictionary.
    """
    features = {}

    # ── Area ──
    area = cv2.contourArea(contour)
    features["area"] = area

    # ── Perimeter ──
    perimeter = cv2.arcLength(contour, closed=True)
    features["perimeter"] = perimeter

    # ── Circularity: C = 4πA / P² ──
    # A perfect circle has circularity = 1.0
    # Lower values indicate more irregular/elongated shapes
    if perimeter > 0:
        circularity = (4 * math.pi * area) / (perimeter ** 2)
    else:
        circularity = 0.0
    features["circularity"] = circularity

    # ── Bounding Rectangle ──
    x, y, w, h = cv2.boundingRect(contour)
    features["bbox"] = (x, y, w, h)

    # ── Aspect Ratio ──
    aspect_ratio = float(w) / h if h > 0 else 0.0
    features["aspect_ratio"] = aspect_ratio

    # ── Solidity = Area / ConvexHullArea ──
    # Measures how "convex" the shape is. Speech bubbles tend to
    # have high solidity (> 0.7), while spiky bubbles are lower.
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    features["solidity"] = solidity

    # ── Extent = Area / BoundingRectArea ──
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0.0
    features["extent"] = extent

    # ── Equivalent Diameter ──
    eq_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0
    features["equivalent_diameter"] = eq_diameter

    # ── Centroid (Moments) ──
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    features["centroid"] = (cx, cy)

    # ── Contour Approximation (Douglas-Peucker) ──
    # epsilon controls approximation accuracy:
    #   smaller epsilon = more vertices retained = closer to original
    #   larger epsilon = fewer vertices = more simplified
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features["num_vertices"] = len(approx)
    features["approx_contour"] = approx

    # ── Ellipse Fitting ──
    # Requires at least 5 points to fit an ellipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        features["ellipse"] = ellipse  # ((cx,cy), (MA,ma), angle)
        # Compute how well the contour matches an ellipse
        # by comparing contour area to ellipse area
        (_, (MA, ma), _) = ellipse
        ellipse_area = math.pi * (MA / 2) * (ma / 2)
        features["ellipse_fit_ratio"] = area / ellipse_area if ellipse_area > 0 else 0.0
    else:
        features["ellipse"] = None
        features["ellipse_fit_ratio"] = 0.0

    # ── Min Area Rotated Rectangle ──
    if len(contour) >= 5:
        min_rect = cv2.minAreaRect(contour)
        features["min_rect"] = min_rect
        rect_area = min_rect[1][0] * min_rect[1][1]
        features["rectangularity"] = area / rect_area if rect_area > 0 else 0.0
    else:
        features["min_rect"] = None
        features["rectangularity"] = 0.0

    # ── Mean Intensity ──
    # Speech bubbles are typically white inside with black text.
    # Therefore, the mean intensity should be very high (> 210).
    if gray_image is not None:
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(gray_image, mask=mask)[0]
        features["mean_intensity"] = mean_val
    else:
        features["mean_intensity"] = 255.0

    return features


def compute_all_features(contours: list, gray_image: np.ndarray = None) -> list:
    """
    Compute shape features for all contours.

    Parameters
    ----------
    contours : list
        List of contours.

    Returns
    -------
    list[dict]
        List of feature dictionaries, one per contour.
    """
    all_features = []
    for i, cnt in enumerate(contours):
        feat = compute_shape_features(cnt, gray_image)
        feat["contour_index"] = i
        feat["contour"] = cnt
        all_features.append(feat)
    logger.info(f"Computed shape features for {len(all_features)} contours")
    return all_features


# ═══════════════════════════════════════════════════════════════
# B. BUBBLE TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
def classify_bubble_type(features: dict) -> str:
    """
    Classify a contour into a bubble type based on geometric features.

    Classification Rules:
    ─────────────────────
    1. RECTANGULAR NARRATION BOX:
       - High rectangularity (> 0.85)
       - Few vertices after approximation (4-6)
       - High solidity (> 0.85)

    2. ELLIPTICAL SPEECH BUBBLE:
       - High circularity (> 0.5)
       - High ellipse fit ratio (> 0.7)
       - High solidity (> 0.75)

    3. SPIKY / SHOUTING BUBBLE:
       - Many vertices (> 12) after approximation
       - Lower solidity (< 0.75) due to jagged edges
       - Lower circularity but not too low

    4. CLOUD-LIKE THOUGHT BUBBLE:
       - Moderate circularity (0.3-0.6)
       - Many vertices (> 8) but rounded overall
       - Moderate solidity

    Parameters
    ----------
    features : dict
        Shape features from compute_shape_features.

    Returns
    -------
    str
        Bubble type: 'speech', 'shout', 'thought', 'narration', 'unknown'.
    """
    circ = features["circularity"]
    solidity = features["solidity"]
    aspect = features["aspect_ratio"]
    n_verts = features["num_vertices"]
    rect = features["rectangularity"]
    ellipse_fit = features["ellipse_fit_ratio"]

    # ── Rule 1: Rectangular narration box ──
    if rect > 0.85 and n_verts <= 6 and solidity > 0.85:
        return "narration"

    # ── Rule 2: Elliptical speech bubble ──
    if circ > 0.5 and ellipse_fit > 0.7 and solidity > 0.75:
        return "speech"

    # ── Rule 3: Spiky / shouting bubble ──
    if n_verts > 12 and solidity < 0.75 and circ < 0.5:
        return "shout"

    # ── Rule 4: Cloud-like thought bubble ──
    if 0.2 < circ < 0.6 and n_verts > 8 and 0.5 < solidity < 0.85:
        return "thought"

    # ── Fallback: still likely a bubble if basic shape criteria met ──
    if solidity > 0.65 and circ > 0.25 and 0.2 < aspect < 5.0:
        return "speech"  # Default to speech for reasonable shapes

    return "unknown"


# ═══════════════════════════════════════════════════════════════
# C. MULTI-CRITERIA FILTERING
# ═══════════════════════════════════════════════════════════════
def filter_bubble_candidates(features_list: list,
                             min_area: int = 500,
                             max_area: int = None,
                             min_circularity: float = 0.1,
                             max_circularity: float = 1.5,
                             min_solidity: float = 0.3,
                             min_aspect: float = 0.15,
                             max_aspect: float = 6.0,
                             max_width_ratio: float = 0.6,
                             max_height_ratio: float = 0.85,
                             image_shape: tuple = None) -> list:
    """
    Apply multi-criteria filtering to identify speech bubble candidates.

    This is the core filtering logic that distinguishes actual speech
    bubbles from noise, characters, panels, and other shapes.

    Parameters
    ----------
    features_list : list[dict]
        Shape features for all contours.
    min_area : int
        Minimum contour area (removes tiny noise).
    max_area : int
        Maximum contour area (removes full-page or panel contours).
    min_circularity : float
        Minimum circularity threshold.
    max_circularity : float
        Maximum circularity (values > 1 can occur due to discretization).
    min_solidity : float
        Minimum solidity (very jagged shapes are likely not bubbles).
    min_aspect : float
        Minimum aspect ratio (very thin shapes are not bubbles).
    max_aspect : float
        Maximum aspect ratio.
    image_shape : tuple
        (H, W) of the image for computing max_area if not given.

    Returns
    -------
    list[dict]
        Filtered features list with added 'bubble_type' key.
    """
    if max_area is None and image_shape is not None:
        max_area = image_shape[0] * image_shape[1] * 0.20 # 20% of image max
    elif max_area is None:
        max_area = float("inf")

    candidates = []
    rejected_reasons = {"area": 0, "circularity": 0, "solidity": 0,
                        "aspect_ratio": 0, "dimensions": 0}

    for feat in features_list:
        area = feat["area"]
        circ = feat["circularity"]
        sol = feat["solidity"]
        aspect = feat["aspect_ratio"]
        x, y, w, h = feat["bbox"]

        # ── Filter checks ──
        if image_shape is not None:
            max_w = image_shape[1] * max_width_ratio
            max_h = image_shape[0] * max_height_ratio
            if w > max_w or h > max_h:
                rejected_reasons["dimensions"] += 1
                continue
        if not (min_area <= area <= max_area):
            rejected_reasons["area"] += 1
            continue

        if not (min_circularity <= circ <= max_circularity):
            rejected_reasons["circularity"] += 1
            continue

        if sol < min_solidity:
            rejected_reasons["solidity"] += 1
            continue

        if not (min_aspect <= aspect <= max_aspect):
            rejected_reasons["aspect_ratio"] += 1
            continue

        # Speech bubbles should be primarily uniform background (white or black).
        # Heavy text can drop a white bubble's mean intensity to ~140.
        mean_int = feat.get("mean_intensity", 255)
        if 50 < mean_int < 130:
            rejected_reasons["mean_intensity"] = rejected_reasons.get("mean_intensity", 0) + 1
            continue

        # ── Classify bubble type ──
        btype = classify_bubble_type(feat)
        if btype == "unknown":
            rejected_reasons["unknown_type"] = rejected_reasons.get("unknown_type", 0) + 1
            continue
            
        feat["bubble_type"] = btype
        candidates.append(feat)

    logger.info(f"Bubble filtering: {len(features_list)} → {len(candidates)} candidates")
    logger.info(f"  Rejected by area: {rejected_reasons['area']}")
    logger.info(f"  Rejected by circularity: {rejected_reasons['circularity']}")
    logger.info(f"  Rejected by solidity: {rejected_reasons['solidity']}")
    logger.info(f"  Rejected by aspect ratio: {rejected_reasons['aspect_ratio']}")
    logger.info(f"  Rejected by dimensions: {rejected_reasons.get('dimensions', 0)}")
    logger.info(f"  Rejected by intensity: {rejected_reasons.get('mean_intensity', 0)}")
    logger.info(f"  Rejected by type unknown: {rejected_reasons.get('unknown_type', 0)}")

    return candidates


# ═══════════════════════════════════════════════════════════════
# C2. MERGE OVERLAPPING CANDIDATES
# ═══════════════════════════════════════════════════════════════
def merge_overlapping_candidates(candidates: list,
                                  iou_threshold: float = 0.15,
                                  distance_ratio: float = 0.4) -> list:
    """
    Merge candidates whose bounding boxes overlap significantly or
    whose centers are very close.

    This fixes over-segmentation where bubble tails, connectors, or
    fragments of the same bubble are detected as separate regions.

    Merge strategy:
        - Compute IoU (Intersection over Union) for each pair.
        - If IoU > iou_threshold, merge into the larger candidate.
        - Also merge if center distance < distance_ratio * avg_size.

    Parameters
    ----------
    candidates : list[dict]
        Filtered bubble candidates with 'bbox' and 'area' keys.
    iou_threshold : float
        IoU above which two candidates are merged.
    distance_ratio : float
        If center distance < this * average(w,h), merge.

    Returns
    -------
    list[dict]
        Merged candidates list.
    """
    if len(candidates) <= 1:
        return candidates

    def bbox_iou(b1, b2):
        """Compute IoU between two bboxes (x,y,w,h)."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        # Intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def bbox_contains(outer, inner):
        """Check if outer bbox fully contains inner bbox."""
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (ix >= ox and iy >= oy and
                ix + iw <= ox + ow and iy + ih <= oy + oh)

    def center_distance(b1, b2):
        """Distance between bbox centers."""
        cx1 = b1[0] + b1[2] / 2
        cy1 = b1[1] + b1[3] / 2
        cx2 = b2[0] + b2[2] / 2
        cy2 = b2[1] + b2[3] / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    # Build merge groups using union-find
    n = len(candidates)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the one with larger area as root
            if candidates[ra]["area"] >= candidates[rb]["area"]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            b1 = candidates[i]["bbox"]
            b2 = candidates[j]["bbox"]

            # Check IoU overlap
            iou = bbox_iou(b1, b2)
            if iou > iou_threshold:
                union(i, j)
                continue

            # Check containment
            if bbox_contains(b1, b2) or bbox_contains(b2, b1):
                union(i, j)
                continue

            # Check center distance relative to avg size
            avg_size = (b1[2] + b1[3] + b2[2] + b2[3]) / 4
            dist = center_distance(b1, b2)
            if dist < distance_ratio * avg_size:
                union(i, j)

    # Group candidates by their root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge each group: take the candidate with the largest area,
    # expand its bbox to enclose all members
    merged = []
    for root, indices in groups.items():
        if len(indices) == 1:
            merged.append(candidates[indices[0]])
        else:
            # Compute enclosing bbox
            bboxes = [candidates[i]["bbox"] for i in indices]
            min_x = min(b[0] for b in bboxes)
            min_y = min(b[1] for b in bboxes)
            max_x = max(b[0] + b[2] for b in bboxes)
            max_y = max(b[1] + b[3] for b in bboxes)

            # Use the largest candidate as the base
            best_idx = max(indices, key=lambda i: candidates[i]["area"])
            base = candidates[best_idx].copy()
            base["bbox"] = (min_x, min_y, max_x - min_x, max_y - min_y)
            merged.append(base)

    logger.info(f"Merge overlapping: {len(candidates)} -> {len(merged)} candidates")
    return merged


# ═══════════════════════════════════════════════════════════════
# D. ROI EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_rois(image: np.ndarray, candidates: list,
                 padding: int = 5) -> list:
    """
    Extract Region of Interest (ROI) images for each bubble candidate.

    Each ROI is cropped from the original image using the bounding box,
    with optional padding to include surrounding context. A contour mask
    is applied to zero-out pixels outside the bubble boundary, which
    prevents OCR from reading outline artifacts and background noise.

    Parameters
    ----------
    image : np.ndarray
        Original BGR or grayscale image.
    candidates : list[dict]
        Filtered bubble candidates with 'bbox' key.
    padding : int
        Pixels of padding around each bounding box.

    Returns
    -------
    list[dict]
        Updated candidates with added 'roi' key containing the cropped image.
    """
    h, w = image.shape[:2]

    for cand in candidates:
        x, y, bw, bh = cand["bbox"]

        # Apply padding, clamping to image boundaries
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        roi = image[y1:y2, x1:x2].copy()

        # Apply contour mask if available — zero out pixels outside
        # the bubble contour to prevent OCR from reading outlines
        if "contour" in cand and cand["contour"] is not None:
            # Create mask for just this ROI region
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            # Shift contour coordinates relative to ROI origin
            shifted = cand["contour"] - np.array([x1, y1])
            cv2.drawContours(mask, [shifted], -1, 255, -1)

            # Erode mask slightly to cut away boundary pixels
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, erode_k, iterations=1)

            # Apply mask: set outside-bubble pixels to white (background)
            if len(roi.shape) == 3:
                roi[mask == 0] = [255, 255, 255]
            else:
                roi[mask == 0] = 255

        cand["roi"] = roi
        cand["roi_coords"] = (x1, y1, x2, y2)

    logger.info(f"Extracted {len(candidates)} ROIs with padding={padding}px + contour masking")
    return candidates


# ═══════════════════════════════════════════════════════════════
# E. READING ORDER SORTING
# ═══════════════════════════════════════════════════════════════
def sort_reading_order(candidates: list,
                       row_threshold: float = 0.3) -> list:
    """
    Sort bubble candidates in natural reading order.

    Reading order for comics/manga:
        - Group bubbles into rows based on vertical position.
        - Within each row, sort left-to-right (or right-to-left for manga).
        - Rows are sorted top-to-bottom.

    The row_threshold parameter determines when two bubbles are
    considered to be on the same row: if their vertical centers
    are within (row_threshold * avg_height), they're on the same row.

    Parameters
    ----------
    candidates : list[dict]
        Bubble candidates with 'bbox' and 'centroid' keys.
    row_threshold : float
        Fraction of average bubble height used to group rows.

    Returns
    -------
    list[dict]
        Candidates sorted in reading order.
    """
    if not candidates:
        return []

    # Compute average height for row grouping
    avg_height = np.mean([c["bbox"][3] for c in candidates])
    threshold = avg_height * row_threshold

    # Sort by y-coordinate first
    sorted_by_y = sorted(candidates, key=lambda c: c["centroid"][1])

    # Group into rows
    rows = []
    current_row = [sorted_by_y[0]]

    for i in range(1, len(sorted_by_y)):
        current_y = sorted_by_y[i]["centroid"][1]
        prev_y = current_row[-1]["centroid"][1]

        if abs(current_y - prev_y) < threshold:
            # Same row
            current_row.append(sorted_by_y[i])
        else:
            # New row
            rows.append(current_row)
            current_row = [sorted_by_y[i]]

    rows.append(current_row)  # Don't forget last row

    # Sort each row by x-coordinate (left-to-right)
    ordered = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c["centroid"][0])
        ordered.extend(row_sorted)

    # Assign bubble IDs
    for i, cand in enumerate(ordered):
        cand["bubble_id"] = i + 1

    logger.info(f"Sorted {len(ordered)} bubbles into {len(rows)} rows (reading order)")
    return ordered


# ═══════════════════════════════════════════════════════════════
# F. DETECTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def draw_detected_bubbles(image: np.ndarray,
                          candidates: list) -> np.ndarray:
    """
    Draw detected bubbles with color-coded bounding boxes and labels.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    candidates : list[dict]
        Detected bubble candidates with type and ID.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    canvas = image.copy()

    for cand in candidates:
        bid = cand.get("bubble_id", 0)
        btype = cand.get("bubble_type", "unknown")
        x, y, w, h = cand["bbox"]

        color = BUBBLE_COLORS.get(btype, BUBBLE_COLORS["unknown"])
        label = BUBBLE_TYPE_LABELS.get(btype, "?")

        # Draw bounding box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # Draw contour
        if "contour" in cand:
            cv2.drawContours(canvas, [cand["contour"]], -1, color, 1)

        # Label
        text = f"#{bid} {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(canvas, text, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return canvas


# ═══════════════════════════════════════════════════════════════
# G. FULL DETECTION PIPELINE
# ═══════════════════════════════════════════════════════════════
def detection_pipeline(original: np.ndarray,
                       contours: list,
                       min_area: int = 500,
                       debug: bool = False,
                       save_dir: str = None) -> dict:
    """
    Run the complete detection pipeline.

    Steps:
        1. Compute shape features for all contours.
        2. Filter candidates using multi-criteria rules.
        3. Classify bubble types.
        4. Extract ROIs from original image.
        5. Sort in reading order.

    Parameters
    ----------
    original : np.ndarray
        Original BGR image.
    contours : list
        Filtered contours from segmentation stage.
    min_area : int
        Minimum area threshold for bubble candidates.
    debug : bool
        Display debug visualizations.
    save_dir : str, optional
        Directory to save debug images.

    Returns
    -------
    dict
        Dictionary with keys:
            'features': all shape features,
            'candidates': filtered bubble candidates,
            'detection_image': annotated visualization
    """
    logger.info("=" * 60)
    logger.info("DETECTION PIPELINE START")
    logger.info("=" * 60)

    results = {}

    # Step 1: Compute shape features
    gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    features = compute_all_features(contours, gray_image)
    results["features"] = features

    # Step 2 & 3: Filter and classify
    candidates = filter_bubble_candidates(
        features,
        min_area=min_area,
        image_shape=original.shape[:2]
    )

    # Step 3.5: Merge overlapping/nearby candidates
    # This merges bubble tails, connectors, and fragments
    candidates = merge_overlapping_candidates(
        candidates, iou_threshold=0.15, distance_ratio=0.4
    )
    results["candidates"] = candidates

    # Step 4: Extract ROIs (with contour masking)
    candidates = extract_rois(original, candidates, padding=8)

    # Step 5: Sort in reading order
    candidates = sort_reading_order(candidates)
    results["candidates"] = candidates

    # Step 6: Visualization
    detection_image = draw_detected_bubbles(original, candidates)
    results["detection_image"] = detection_image

    # Log summary
    type_counts = {}
    for c in candidates:
        bt = c.get("bubble_type", "unknown")
        type_counts[bt] = type_counts.get(bt, 0) + 1
    logger.info(f"Detection summary: {len(candidates)} bubbles found")
    for bt, count in type_counts.items():
        logger.info(f"  {BUBBLE_TYPE_LABELS.get(bt, bt)}: {count}")

    if debug:
        from utils import show_images, save_image
        if save_dir:
            save_image(detection_image, f"{save_dir}/03_detection.png")

            # Save individual ROIs
            roi_dir = f"{save_dir}/rois"
            import os
            os.makedirs(roi_dir, exist_ok=True)
            for cand in candidates:
                bid = cand["bubble_id"]
                roi = cand["roi"]
                save_image(roi, f"{roi_dir}/bubble_{bid:02d}.png")

            show_images(
                [detection_image],
                [f"Detected Bubbles ({len(candidates)})"],
                save_path=f"{save_dir}/03_detection_overview.png"
            )
        else:
            show_images(
                [detection_image],
                [f"Detected Bubbles ({len(candidates)})"],
            )

    logger.info("DETECTION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results
