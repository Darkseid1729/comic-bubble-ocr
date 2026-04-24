"""
segmentation.py - Contour Detection & Segmentation Module
==========================================================
Implements the segmentation stage of the pipeline:

    - Contour Detection (cv2.findContours)
    - Connected Component Analysis
    - Watershed Algorithm (for touching/overlapping bubbles)
    - Contour filtering by geometric properties

All functions operate on binary/morphed images from preprocessing.
"""

import cv2
import numpy as np
from utils import logger, show_images, draw_contours_debug


# ═══════════════════════════════════════════════════════════════
# A. CONTOUR DETECTION
# ═══════════════════════════════════════════════════════════════
def find_contours(binary_image: np.ndarray,
                  mode: int = cv2.RETR_EXTERNAL,
                  method: int = cv2.CHAIN_APPROX_SIMPLE) -> list:
    """
    Find contours in a binary image.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image (white = foreground).
    mode : int
        Contour retrieval mode:
        - cv2.RETR_EXTERNAL: Only outermost contours (best for bubbles).
        - cv2.RETR_TREE: Full hierarchy (useful for nested analysis).
        - cv2.RETR_LIST: All contours, no hierarchy.
    method : int
        Contour approximation method:
        - cv2.CHAIN_APPROX_SIMPLE: Compresses segments.
        - cv2.CHAIN_APPROX_NONE: Stores all boundary points.

    Returns
    -------
    list
        List of contours (each is an np.ndarray of points).
    """
    contours, hierarchy = cv2.findContours(
        binary_image, mode, method
    )
    logger.info(f"Found {len(contours)} raw contours (mode={mode})")
    return contours


# ═══════════════════════════════════════════════════════════════
# B. CONNECTED COMPONENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
def connected_components(binary_image: np.ndarray,
                         connectivity: int = 8) -> tuple:
    """
    Perform connected component analysis.

    This labels each connected white region with a unique integer.
    Useful as an alternative to contour detection, especially
    when you need pixel-level region membership.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image (white = foreground).
    connectivity : int
        4-connectivity or 8-connectivity.

    Returns
    -------
    tuple
        (num_labels, labels, stats, centroids)
        - num_labels: Total number of components (including background=0).
        - labels: Label image (H x W), each pixel has its component ID.
        - stats: Stats array [x, y, w, h, area] for each component.
        - centroids: Centroid (cx, cy) for each component.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )
    # Label 0 is always the background
    logger.info(f"Connected Components: {num_labels - 1} foreground regions")
    return num_labels, labels, stats, centroids


def filter_components_by_area(stats: np.ndarray, centroids: np.ndarray,
                              min_area: int = 500,
                              max_area: int = None,
                              image_area: int = None) -> list:
    """
    Filter connected components by area.

    Parameters
    ----------
    stats : np.ndarray
        Stats from connectedComponentsWithStats.
    centroids : np.ndarray
        Centroids from connectedComponentsWithStats.
    min_area : int
        Minimum area threshold.
    max_area : int
        Maximum area threshold. If None, uses 50% of image area.
    image_area : int
        Total image area (H * W) for computing max threshold.

    Returns
    -------
    list[dict]
        List of dicts with keys: label, x, y, w, h, area, centroid.
    """
    if max_area is None and image_area is not None:
        max_area = image_area * 0.5
    elif max_area is None:
        max_area = float("inf")

    results = []
    for i in range(1, len(stats)):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            results.append({
                "label": i,
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "area": int(area),
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            })

    logger.info(f"Filtered components: {len(results)} passed area filter "
                f"(min={min_area}, max={max_area:.0f})")
    return results


# ═══════════════════════════════════════════════════════════════
# C. WATERSHED ALGORITHM (for touching/overlapping bubbles)
# ═══════════════════════════════════════════════════════════════
def watershed_segmentation(original_bgr: np.ndarray,
                           binary_image: np.ndarray) -> tuple:
    """
    Apply the Watershed algorithm to separate touching bubbles.

    The Watershed algorithm treats the image as a topographic surface
    and finds the "watershed lines" that separate different basins.
    This is critical for comics where speech bubbles overlap or touch.

    Pipeline:
        1. Distance Transform on binary image.
        2. Threshold distance map to find sure foreground (seed markers).
        3. Dilate binary to find sure background.
        4. Unknown region = sure_bg - sure_fg.
        5. Label markers and apply watershed.

    Parameters
    ----------
    original_bgr : np.ndarray
        Original color image (needed by cv2.watershed).
    binary_image : np.ndarray
        Binary image with white foreground regions.

    Returns
    -------
    tuple
        (markers, segmented_image)
        - markers: Label image where each region has a unique ID.
                   Boundary pixels have value -1.
        - segmented_image: Original image with watershed boundaries drawn.
    """
    # Step 1: Distance Transform
    # Computes distance of each foreground pixel to nearest background pixel
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)

    # Step 2: Sure foreground (high confidence regions)
    # Pixels far from any edge are definitely part of a bubble
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    # Step 3: Sure background (dilated binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(binary_image, kernel, iterations=3)

    # Step 4: Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 5: Label markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background becomes 1 instead of 0
    markers[unknown == 255] = 0  # Unknown region marked as 0

    # Step 6: Apply Watershed
    markers = cv2.watershed(original_bgr, markers)

    # Draw boundaries on the image
    segmented = original_bgr.copy()
    segmented[markers == -1] = [0, 0, 255]  # Red boundaries

    logger.info(f"Watershed segmentation: {num_labels} regions found")
    return markers, segmented


# ═══════════════════════════════════════════════════════════════
# D. FULL SEGMENTATION PIPELINE
# ═══════════════════════════════════════════════════════════════
def segmentation_pipeline(original: np.ndarray,
                          morphed: np.ndarray,
                          min_area: int = 500,
                          max_area_ratio: float = 0.5,
                          use_watershed: bool = False,
                          debug: bool = False,
                          save_dir: str = None) -> dict:
    """
    Run the complete segmentation pipeline.

    Steps:
        1. Find contours on morphed binary image.
        2. Optionally apply Watershed for touching bubbles.
        3. Run Connected Component Analysis.
        4. Filter components by area.

    Parameters
    ----------
    original : np.ndarray
        Original BGR image (for watershed and visualization).
    morphed : np.ndarray
        Morphologically processed binary image from preprocessing.
    min_area : int
        Minimum contour area to consider as a potential bubble.
    max_area_ratio : float
        Maximum area as fraction of total image area.
    use_watershed : bool
        Whether to apply watershed for touching bubble separation.
    debug : bool
        If True, display intermediate results.
    save_dir : str, optional
        Directory to save debug images.

    Returns
    -------
    dict
        Dictionary with keys:
            'contours': raw contours,
            'filtered_contours': area-filtered contours,
            'components': connected component analysis results,
            'markers': watershed markers (if used),
            'contour_image': debug visualization
    """
    logger.info("=" * 60)
    logger.info("SEGMENTATION PIPELINE START")
    logger.info("=" * 60)

    image_area = morphed.shape[0] * morphed.shape[1]
    max_area = int(image_area * max_area_ratio)

    results = {}

    # Step 1: Find all contours
    contours = find_contours(morphed, mode=cv2.RETR_EXTERNAL)
    results["contours"] = contours

    # Step 2: Filter contours by area
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered.append(cnt)
    results["filtered_contours"] = filtered
    logger.info(f"Area filtering: {len(contours)} → {len(filtered)} contours "
                f"(min={min_area}, max={max_area})")

    # Step 3: Connected Component Analysis
    num_labels, labels, stats, centroids = connected_components(morphed)
    components = filter_components_by_area(
        stats, centroids, min_area=min_area,
        max_area=max_area, image_area=image_area
    )
    results["components"] = components
    results["labels"] = labels

    # Step 4: Watershed (optional, for touching bubbles)
    if use_watershed and len(filtered) > 0:
        markers, segmented = watershed_segmentation(original, morphed)
        results["markers"] = markers
        results["watershed_image"] = segmented
        logger.info("Watershed segmentation applied for touching bubbles")
    else:
        results["markers"] = None

    # Debug visualization
    contour_image = draw_contours_debug(original, contours, "All Contours")
    filtered_image = draw_contours_debug(original, filtered, "Filtered Contours")
    results["contour_image_all"] = contour_image
    results["contour_image_filtered"] = filtered_image

    if debug:
        debug_imgs = [contour_image, filtered_image]
        debug_titles = [
            f"All Contours ({len(contours)})",
            f"Filtered Contours ({len(filtered)})"
        ]
        if "watershed_image" in results:
            debug_imgs.append(results["watershed_image"])
            debug_titles.append("Watershed Boundaries")

        if save_dir:
            show_images(debug_imgs, debug_titles,
                        save_path=f"{save_dir}/02_segmentation.png")
        else:
            show_images(debug_imgs, debug_titles)

    logger.info("SEGMENTATION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results
