"""
ocr.py - OCR Integration & Post-Processing Module
====================================================
Handles text extraction from detected speech bubbles using Tesseract OCR.

Implements:
    - ROI-specific preprocessing (resize, threshold, invert, deskew)
    - Tesseract OCR with tuned configurations (psm modes)
    - Text post-processing (noise removal, spacing fixes)
    - Full-image OCR baseline for comparison
    - Evaluation metrics (character accuracy, word accuracy)
"""

import cv2
import numpy as np
import pytesseract
import re
import math
import os
from utils import logger


# ─── Auto-detect Tesseract path on Windows ───
# If tesseract is not on PATH, try common install locations
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]
for _path in _TESSERACT_PATHS:
    if os.path.isfile(_path):
        pytesseract.pytesseract.tesseract_cmd = _path
        logger.info(f"Tesseract found at: {_path}")
        break
else:
    # Check if it's on PATH already
    import shutil
    if shutil.which("tesseract"):
        logger.info("Tesseract found on system PATH")
    else:
        logger.warning(
            "Tesseract OCR not found! Install from: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        )


# ═══════════════════════════════════════════════════════════════
# A. ROI PREPROCESSING FOR OCR
# ═══════════════════════════════════════════════════════════════
def resize_for_ocr(roi: np.ndarray, target_dpi: int = 300,
                   min_height: int = 50) -> np.ndarray:
    """
    Resize an ROI to simulate a target DPI for better OCR accuracy.

    Tesseract works best at ~300 DPI. If the ROI is small, upscaling
    dramatically improves character recognition.

    Parameters
    ----------
    roi : np.ndarray
        Cropped bubble image.
    target_dpi : int
        Target DPI equivalent. Higher = larger image.
    min_height : int
        Minimum height in pixels after resize.

    Returns
    -------
    np.ndarray
        Resized ROI.
    """
    h, w = roi.shape[:2]

    # Scale factor: aim for at least min_height pixels tall
    if h < min_height:
        scale = max(2.0, min_height / h)
    else:
        scale = max(1.0, target_dpi / 150.0)  # Assume ~150 DPI input

    # Cap scale to avoid excessively large images
    scale = min(scale, 4.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.debug(f"Resized ROI: {w}x{h} → {new_w}x{new_h} (scale={scale:.1f}x)")
    return resized


def prepare_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Apply OCR-specific preprocessing to a single ROI.

    Pipeline:
        1. Convert to grayscale (if color).
        2. Resize for ~300 DPI equivalent.
        3. Apply adaptive thresholding for clean binary text.
        4. Invert if background is dark (white text on dark bg).
        5. Optional deskewing.

    Parameters
    ----------
    roi : np.ndarray
        Cropped bubble image (BGR or grayscale).

    Returns
    -------
    np.ndarray
        Preprocessed binary image ready for Tesseract.
    """
    # Step 1: Grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Step 2: Resize for better OCR
    resized = resize_for_ocr(gray)

    # Step 3: Denoise
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)

    # Step 4: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )

    # Step 5: Check if we need to invert
    # If the background is dark (mean < 128), invert so text is dark on white
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)
        logger.debug("Inverted ROI (dark background detected)")

    # Step 6: Deskew if needed
    binary = deskew_image(binary)

    return binary


def deskew_image(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Deskew a binary image using the Hough Line Transform.

    If text lines are slightly rotated, this corrects the angle
    to improve OCR accuracy.

    Parameters
    ----------
    image : np.ndarray
        Binary image.
    max_angle : float
        Maximum skew angle to correct (degrees). Larger skews
        are likely not text skew and are ignored.

    Returns
    -------
    np.ndarray
        Deskewed image.
    """
    # Detect lines using Hough Transform
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=50, minLineLength=30, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return image

    # Compute the median angle of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)

    # Only correct if skew is significant (> 0.5 degrees)
    if abs(median_angle) < 0.5:
        return image

    # Rotate to correct skew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    logger.debug(f"Deskewed by {median_angle:.1f}°")
    return rotated


# ═══════════════════════════════════════════════════════════════
# B. TESSERACT OCR
# ═══════════════════════════════════════════════════════════════
def run_tesseract(image: np.ndarray, psm: int = 6,
                  lang: str = "eng",
                  whitelist: str = None) -> str:
    """
    Run Tesseract OCR on a preprocessed image.

    PSM (Page Segmentation Mode) guide:
        - psm 3: Fully automatic page segmentation (default).
        - psm 6: Assume a single uniform block of text.
                  **Best for speech bubbles with multiple lines.**
        - psm 7: Treat the image as a single text line.
                  Good for narrow bubbles with one line.
        - psm 11: Sparse text. No particular order.
                   Good for scattered text in complex layouts.
        - psm 13: Raw line. Treat as a single line without OSD.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed binary image.
    psm : int
        Page segmentation mode.
    lang : str
        Language for OCR (e.g., 'eng', 'jpn' for manga).
    whitelist : str, optional
        Characters to restrict OCR to (e.g., only alphanumeric).

    Returns
    -------
    str
        Extracted text.
    """
    config = f"--psm {psm}"

    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    # Add OEM (OCR Engine Mode) - use LSTM engine
    config += " --oem 3"

    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        text = ""

    return text


def ocr_with_fallback(image: np.ndarray, lang: str = "eng") -> str:
    """
    Try multiple PSM modes and return the best result.

    Strategy:
        1. Try psm 6 (block of text) - most common for bubbles.
        2. Try psm 7 (single line) - for narrow/small bubbles.
        3. Try psm 11 (sparse text) - for unusual layouts.

    The "best" result is the one with the most non-whitespace characters,
    as empty results indicate the PSM mode didn't match the layout.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed binary image.
    lang : str
        OCR language.

    Returns
    -------
    str
        Best OCR result.
    """
    results = {}
    for psm in [6, 5, 7, 11]:
        text = run_tesseract(image, psm=psm, lang=lang)
        clean = text.strip()
        results[psm] = clean

    # Pick the result with the most meaningful content
    best_psm = max(results, key=lambda k: len(results[k].replace(" ", "")))
    best_text = results[best_psm]

    if best_text:
        logger.debug(f"Best OCR result from psm={best_psm}: '{best_text[:50]}...'")
    else:
        logger.debug("OCR returned empty result for all PSM modes")

    return best_text


def estimate_confidence(image: np.ndarray, lang: str = "eng", psm: int = 6) -> float:
    """
    Estimate OCR confidence using Tesseract word-level confidences.

    Returns a value between 0.0 and 1.0. If no confidence values are
    available, returns 0.0.
    """
    config = f"--psm {psm} --oem 3"
    try:
        data = pytesseract.image_to_data(
            image, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        logger.debug(f"Confidence estimation failed: {e}")
        return 0.0

    confs = []
    for value in data.get("conf", []):
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num >= 0:
            confs.append(num)

    if not confs:
        return 0.0

    return sum(confs) / len(confs) / 100.0


# ═══════════════════════════════════════════════════════════════
# C. TEXT POST-PROCESSING
# ═══════════════════════════════════════════════════════════════
def clean_ocr_text(text: str) -> str:
    """
    Clean and normalize OCR output text.

    Operations:
        1. Remove common OCR noise characters (|, \\, ~, etc.)
        2. Fix excessive whitespace.
        3. Remove leading/trailing whitespace per line.
        4. Merge hyphenated line breaks.
        5. Remove empty lines.

    Parameters
    ----------
    text : str
        Raw OCR output.

    Returns
    -------
    str
        Cleaned text.
    """
    if not text:
        return ""

    # Remove common OCR noise characters
    # These are characters that Tesseract often misreads from
    # bubble outlines, background textures, or scan artifacts
    noise_chars = r'[|\\~`{}<>^°©®™•]'
    text = re.sub(noise_chars, '', text)

    # Fix multiple spaces → single space
    text = re.sub(r' {2,}', ' ', text)

    # Fix multiple newlines → single newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace per line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]

    # Merge hyphenated words across lines
    # e.g., "some-\nword" → "someword"
    merged = []
    for i, line in enumerate(lines):
        if line.endswith('-') and i + 1 < len(lines):
            merged.append(line[:-1])  # Remove hyphen, don't add newline
        else:
            merged.append(line)

    # Remove empty lines
    merged = [line for line in merged if line.strip()]

    result = '\n'.join(merged).strip()
    return result


def merge_bubble_text(text: str) -> str:
    """
    Merge multi-line text within a single bubble into a coherent block.

    Speech bubbles often contain text broken across multiple lines
    due to the bubble's shape. This merges those lines while
    preserving intentional paragraph breaks.

    Parameters
    ----------
    text : str
        Cleaned OCR text from a single bubble.

    Returns
    -------
    str
        Merged text as a single paragraph or preserved paragraphs.
    """
    if not text:
        return ""

    lines = text.split('\n')

    # If text has very few lines, just join with spaces
    if len(lines) <= 3:
        return ' '.join(lines)

    # For longer text, preserve paragraph breaks (double newlines)
    # but merge single-newline breaks
    paragraphs = []
    current = []
    for line in lines:
        if line.strip() == '':
            if current:
                paragraphs.append(' '.join(current))
                current = []
        else:
            current.append(line.strip())

    if current:
        paragraphs.append(' '.join(current))

    return '\n'.join(paragraphs)


# ═══════════════════════════════════════════════════════════════
# D. FULL-IMAGE OCR (BASELINE)
# ═══════════════════════════════════════════════════════════════
def ocr_full_image(image: np.ndarray, lang: str = "eng") -> str:
    """
    Run OCR on the entire image without bubble segmentation.

    This serves as a BASELINE to compare against the bubble-aware
    approach. It demonstrates how text from different speakers gets
    mixed together when no segmentation is applied.

    Parameters
    ----------
    image : np.ndarray
        Full comic page image (BGR).
    lang : str
        OCR language.

    Returns
    -------
    str
        Raw OCR text from the full image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    text = run_tesseract(binary, psm=3, lang=lang)
    clean = clean_ocr_text(text)
    logger.info(f"Full-image OCR baseline: {len(clean)} characters extracted")
    return clean


# ═══════════════════════════════════════════════════════════════
# E. EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════
def character_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Compute character-level accuracy.

    Uses Levenshtein-like comparison: counts matching characters
    at corresponding positions.

    Parameters
    ----------
    predicted : str
        OCR output text.
    ground_truth : str
        Expected correct text.

    Returns
    -------
    float
        Accuracy between 0.0 and 1.0.
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    # Simple character-level comparison
    matches = 0
    max_len = max(len(predicted), len(ground_truth))

    for i in range(min(len(predicted), len(ground_truth))):
        if predicted[i] == ground_truth[i]:
            matches += 1

    return matches / max_len if max_len > 0 else 1.0


def word_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Compute word-level accuracy.

    Compares sets of words and computes the fraction of
    ground truth words that appear in the prediction.

    Parameters
    ----------
    predicted : str
        OCR output text.
    ground_truth : str
        Expected correct text.

    Returns
    -------
    float
        Accuracy between 0.0 and 1.0.
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())

    if not gt_words:
        return 1.0

    matches = len(pred_words & gt_words)
    return matches / len(gt_words)


# ═══════════════════════════════════════════════════════════════
# F. OCR PIPELINE
# ═══════════════════════════════════════════════════════════════
def ocr_pipeline(candidates: list, original: np.ndarray,
                 lang: str = "eng",
                 debug: bool = False,
                 save_dir: str = None) -> list:
    """
    Run OCR on all detected bubble candidates.

    For each candidate:
        1. Take the ROI image.
        2. Apply OCR-specific preprocessing.
        3. Run Tesseract with fallback PSM modes.
        4. Clean and merge the extracted text.

    Also runs full-image OCR baseline for comparison.

    Parameters
    ----------
    candidates : list[dict]
        Detected bubble candidates with 'roi' key.
    original : np.ndarray
        Original full image (for baseline comparison).
    lang : str
        OCR language code.
    debug : bool
        Show debug visualizations.
    save_dir : str, optional
        Directory to save debug images.

    Returns
    -------
    list[dict]
        Results list with keys: bubble_id, bubble_type, text, bbox, confidence.
    """
    logger.info("=" * 60)
    logger.info("OCR PIPELINE START")
    logger.info("=" * 60)

    results = []

    # ── Run OCR on each bubble ──
    for cand in candidates:
        bid = cand.get("bubble_id", 0)
        btype = cand.get("bubble_type", "unknown")
        roi = cand.get("roi")

        if roi is None or roi.size == 0:
            logger.warning(f"Bubble #{bid}: Empty ROI, skipping")
            continue

        # Preprocess ROI for OCR
        prepared = prepare_roi_for_ocr(roi)

        # Run OCR with fallback
        raw_text = ocr_with_fallback(prepared, lang=lang)

        # Clean and merge text
        cleaned = clean_ocr_text(raw_text)
        merged = merge_bubble_text(cleaned)

        # Estimate OCR confidence (0.0 - 1.0)
        confidence = estimate_confidence(prepared, lang=lang)

        # Skip bubbles with no meaningful text
        clean_merged = merged.strip()
        if len(clean_merged) < 2 and clean_merged not in ["I", "a", "O", "?", "!"]:
            logger.info(f"Bubble #{bid} ({btype}) rejected due to empty/garbage text.")
            continue

        # Reject long gibberish (e.g. Tesseract reading line art)
        # Check if at least 50% of the characters (excluding spaces) are alphanumeric
        no_spaces = clean_merged.replace(" ", "").replace("\n", "")
        if len(no_spaces) > 0:
            alnum_count = sum(1 for c in no_spaces if c.isalnum())
            if alnum_count / len(no_spaces) < 0.5:
                logger.info(f"Bubble #{bid} ({btype}) rejected due to low alphanumeric ratio (gibberish).")
                continue

        result = {
            "bubble_id": bid,
            "bubble_type": btype,
            "text": merged,
            "bbox": list(cand["bbox"]),
            "confidence": confidence,
        }
        results.append(result)

        logger.info(f"Bubble #{bid} ({btype}): '{merged[:60]}{'...' if len(merged) > 60 else ''}'")

        # Save debug images for OCR preprocessing
        if debug and save_dir:
            import os
            ocr_debug_dir = os.path.join(save_dir, "ocr_debug")
            os.makedirs(ocr_debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(ocr_debug_dir, f"bubble_{bid:02d}_prepared.png"),
                prepared
            )

    # ── Baseline: Full-image OCR ──
    baseline_text = ocr_full_image(original, lang=lang)
    logger.info(f"\n{'='*40}")
    logger.info(f"BASELINE (full-image OCR): {len(baseline_text)} chars")
    logger.info(f"BUBBLE-AWARE OCR: {sum(len(r['text']) for r in results)} chars "
                f"across {len(results)} bubbles")
    logger.info(f"{'='*40}")

    if debug and save_dir:
        # Save baseline comparison
        with open(os.path.join(save_dir, "baseline_ocr.txt"), "w", encoding="utf-8") as f:
            f.write("=== FULL IMAGE OCR (BASELINE) ===\n\n")
            f.write(baseline_text)
            f.write("\n\n=== BUBBLE-AWARE OCR ===\n\n")
            for r in results:
                f.write(f"--- Bubble #{r['bubble_id']} ({r['bubble_type']}) ---\n")
                f.write(r['text'])
                f.write("\n\n")

    logger.info("OCR PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results
