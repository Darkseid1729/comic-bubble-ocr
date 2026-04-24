"""
preprocessing.py - Image Preprocessing Module
===============================================
Implements the first stage of the Speech Bubble-Aware OCR pipeline.

Techniques implemented:
    A. Grayscale Conversion  (Gray = 0.299R + 0.587G + 0.114B)
    B. Noise Reduction       (Gaussian, Median, Bilateral)
    C. Contrast Enhancement  (Histogram Equalization, CLAHE)
    D. Binarization          (Global, Adaptive Mean/Gaussian, Otsu)
    E. Edge Detection        (Canny, Sobel)
    F. Morphological Ops     (Erosion, Dilation, Opening, Closing)

All functions accept and return numpy arrays. Debug visualization
can be enabled by passing `debug=True` to the pipeline wrapper.
"""

import cv2
import numpy as np
from utils import logger, show_images


# ═══════════════════════════════════════════════════════════════
# A. GRAYSCALE CONVERSION
# ═══════════════════════════════════════════════════════════════
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Uses the luminance formula:
        Gray = 0.299 * R + 0.587 * G + 0.114 * B

    OpenCV's cvtColor uses the same ITU-R BT.601 weighting internally.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (H x W x 3).

    Returns
    -------
    np.ndarray
        Grayscale image (H x W).
    """
    if len(image.shape) == 2:
        logger.info("Image is already grayscale, skipping conversion.")
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.info(f"Converted to grayscale: {gray.shape}")
    return gray


# ═══════════════════════════════════════════════════════════════
# B. NOISE REDUCTION
# ═══════════════════════════════════════════════════════════════
def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur for general noise reduction.

    WHEN TO USE:
        - General-purpose smoothing
        - When edge preservation is NOT critical
        - Fast and effective for mild noise

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    ksize : int
        Kernel size (must be odd). Larger = more blur.
        Recommended: 3 for mild noise, 5 for moderate, 7 for heavy.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    result = cv2.GaussianBlur(image, (ksize, ksize), 0)
    logger.info(f"Applied Gaussian Blur (kernel={ksize})")
    return result


def median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply median filter for salt-and-pepper noise removal.

    WHEN TO USE:
        - Scanned comic pages with speckle/salt-pepper noise
        - When you want to preserve edges better than Gaussian
        - Excellent for removing isolated bright/dark pixels

    Parameters
    ----------
    image : np.ndarray
        Input image.
    ksize : int
        Kernel size (must be odd). Typically 3 or 5.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    result = cv2.medianBlur(image, ksize)
    logger.info(f"Applied Median Filter (kernel={ksize})")
    return result


def bilateral_filter(image: np.ndarray, d: int = 9,
                     sigma_color: float = 75,
                     sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving noise reduction.

    WHEN TO USE:
        - **Best choice for comics** because it smooths texture/noise
          while preserving the sharp edges of speech bubble outlines.
        - When bubble boundaries must remain crisp after denoising.
        - Slower than Gaussian/Median but produces superior results for
          subsequent contour detection.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    d : int
        Diameter of each pixel neighborhood. Use -1 for auto.
    sigma_color : float
        Filter sigma in the color space. Larger = farther colors mixed.
    sigma_space : float
        Filter sigma in the coordinate space. Larger = farther pixels influence.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    logger.info(f"Applied Bilateral Filter (d={d}, σ_color={sigma_color}, σ_space={sigma_space})")
    return result


def reduce_noise(image: np.ndarray, method: str = "bilateral",
                 ksize: int = 5) -> np.ndarray:
    """
    Unified noise reduction interface.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    method : str
        One of 'gaussian', 'median', 'bilateral'.
    ksize : int
        Kernel size for gaussian/median filters.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    if method == "gaussian":
        return gaussian_blur(image, ksize)
    elif method == "median":
        return median_blur(image, ksize)
    elif method == "bilateral":
        return bilateral_filter(image)
    else:
        logger.warning(f"Unknown noise reduction method '{method}', returning original.")
        return image


# ═══════════════════════════════════════════════════════════════
# C. CONTRAST ENHANCEMENT
# ═══════════════════════════════════════════════════════════════
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization.

    WHEN TO USE:
        - Images with uniform lighting but low overall contrast.
        - Simple and fast, but can over-amplify noise in regions
          that are already well-contrasted.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Equalized image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(image)
    logger.info("Applied global Histogram Equalization")
    return result


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    WHEN TO USE:
        - **Preferred for comics** because comic pages often have
          uneven illumination (scan artifacts, shadows near spine).
        - CLAHE operates on local tiles, so it enhances contrast
          adaptively without over-amplifying already-bright regions.
        - clip_limit controls noise amplification (higher = more contrast
          but also more noise).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    clip_limit : float
        Contrast limit for histogram clipping. Default 2.0.
    tile_grid_size : tuple
        Size of the grid for local equalization (rows, cols).

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(image)
    logger.info(f"Applied CLAHE (clip={clip_limit}, grid={tile_grid_size})")
    return result


def enhance_contrast(image: np.ndarray, method: str = "clahe",
                     **kwargs) -> np.ndarray:
    """
    Unified contrast enhancement interface.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    method : str
        'he' for Histogram Equalization, 'clahe' for CLAHE.

    Returns
    -------
    np.ndarray
        Enhanced image.
    """
    if method == "he":
        return histogram_equalization(image)
    elif method == "clahe":
        return clahe_enhancement(image, **kwargs)
    else:
        logger.warning(f"Unknown contrast method '{method}', returning original.")
        return image


# ═══════════════════════════════════════════════════════════════
# D. BINARIZATION / THRESHOLDING
# ═══════════════════════════════════════════════════════════════
def global_threshold(image: np.ndarray, thresh: int = 127) -> np.ndarray:
    """
    Apply global (fixed) thresholding.

    WHEN TO USE:
        - Uniformly lit images where a single threshold separates
          foreground from background.
        - Fast but fails on images with uneven lighting.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    thresh : int
        Threshold value (0-255).

    Returns
    -------
    np.ndarray
        Binary image.
    """
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    logger.info(f"Applied Global Threshold (T={thresh})")
    return binary


def adaptive_threshold(image: np.ndarray, method: str = "gaussian",
                       block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding.

    WHEN TO USE:
        - **Best for comics with uneven lighting** (scan artifacts,
          shadows, gradients across the page).
        - 'gaussian' weighted is generally smoother than 'mean'.
        - block_size controls the neighborhood size (larger = more global).
        - C is subtracted from the computed threshold (tuning parameter).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    method : str
        'mean' or 'gaussian'.
    block_size : int
        Size of the neighborhood area (must be odd, >= 3).
    C : int
        Constant subtracted from the mean/weighted mean.

    Returns
    -------
    np.ndarray
        Binary image.
    """
    adapt_method = (cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean"
                    else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    binary = cv2.adaptiveThreshold(
        image, 255, adapt_method, cv2.THRESH_BINARY, block_size, C
    )
    logger.info(f"Applied Adaptive Threshold (method={method}, block={block_size}, C={C})")
    return binary


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding.

    WHEN TO USE:
        - When you don't know the optimal threshold and the image
          has a bimodal histogram (clear foreground/background separation).
        - Good default for well-scanned pages.
        - Can be combined with Gaussian blur for better results.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Binary image.
    """
    # Apply Gaussian blur before Otsu for better results
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_val, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    logger.info(f"Applied Otsu's Threshold (auto T={thresh_val:.0f})")
    return binary


def binarize(image: np.ndarray, method: str = "adaptive_gaussian",
             **kwargs) -> np.ndarray:
    """
    Unified binarization interface.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    method : str
        One of: 'global', 'adaptive_mean', 'adaptive_gaussian', 'otsu'.

    Returns
    -------
    np.ndarray
        Binary image.
    """
    if method == "global":
        return global_threshold(image, **kwargs)
    elif method == "adaptive_mean":
        return adaptive_threshold(image, method="mean", **kwargs)
    elif method == "adaptive_gaussian":
        return adaptive_threshold(image, method="gaussian", **kwargs)
    elif method == "otsu":
        return otsu_threshold(image)
    else:
        logger.warning(f"Unknown binarization method '{method}', using Otsu.")
        return otsu_threshold(image)


# ═══════════════════════════════════════════════════════════════
# E. EDGE DETECTION
# ═══════════════════════════════════════════════════════════════
def canny_edge(image: np.ndarray, low: int = 50, high: int = 150,
               aperture: int = 3) -> np.ndarray:
    """
    Apply Canny edge detection.

    The Canny detector uses two thresholds:
        - Edges with gradient > high are STRONG edges (kept).
        - Edges between low and high are WEAK edges (kept only if
          connected to a strong edge via hysteresis).
        - Edges below low are discarded.

    TUNING GUIDE:
        - For clean comics: low=50, high=150 works well.
        - For noisy scans: increase both thresholds (80, 200).
        - Ratio of 1:2 or 1:3 between low:high is a good starting point.
        - aperture_size affects the Sobel kernel used internally.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (should be denoised first).
    low : int
        Lower hysteresis threshold.
    high : int
        Upper hysteresis threshold.
    aperture : int
        Aperture size for the Sobel operator (3, 5, or 7).

    Returns
    -------
    np.ndarray
        Binary edge map.
    """
    edges = cv2.Canny(image, low, high, apertureSize=aperture)
    logger.info(f"Applied Canny Edge Detection (low={low}, high={high}, aperture={aperture})")
    return edges


def sobel_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Sobel edge detection (magnitude of X and Y gradients).

    Useful for comparison with Canny. Sobel provides gradient magnitude
    but does NOT perform non-maximum suppression or hysteresis, so
    edges are thicker and noisier.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    ksize : int
        Kernel size for the Sobel operator.

    Returns
    -------
    np.ndarray
        Edge magnitude image (normalized to 0-255).
    """
    # Compute gradients in X and Y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize to 0-255 range
    magnitude = np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))
    logger.info(f"Applied Sobel Edge Detection (kernel={ksize})")
    return magnitude


def detect_edges(image: np.ndarray, method: str = "canny",
                 **kwargs) -> np.ndarray:
    """
    Unified edge detection interface.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    method : str
        'canny' or 'sobel'.

    Returns
    -------
    np.ndarray
        Edge map.
    """
    if method == "canny":
        return canny_edge(image, **kwargs)
    elif method == "sobel":
        return sobel_edge(image, **kwargs)
    else:
        logger.warning(f"Unknown edge method '{method}', using Canny.")
        return canny_edge(image, **kwargs)


# ═══════════════════════════════════════════════════════════════
# F. MORPHOLOGICAL OPERATIONS
# ═══════════════════════════════════════════════════════════════
def get_kernel(shape: str = "rect", size: tuple = (5, 5)) -> np.ndarray:
    """
    Create a structuring element (kernel) for morphological operations.

    Types:
        - 'rect': Rectangular kernel. Good for general-purpose ops.
        - 'ellipse': Elliptical kernel. Better for rounded bubble shapes.
        - 'cross': Cross-shaped kernel. Good for fine-grained control.

    Parameters
    ----------
    shape : str
        Kernel shape: 'rect', 'ellipse', or 'cross'.
    size : tuple
        Kernel dimensions (width, height).

    Returns
    -------
    np.ndarray
        Structuring element.
    """
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    morph_shape = shapes.get(shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, size)


def morphological_erode(image: np.ndarray, kernel: np.ndarray = None,
                        iterations: int = 1) -> np.ndarray:
    """
    Erosion: Shrinks white regions / expands black regions.

    USE CASE: Separate touching objects, remove small bright noise.

    Parameters
    ----------
    image : np.ndarray
        Binary or grayscale image.
    kernel : np.ndarray
        Structuring element. If None, uses 5x5 rect.
    iterations : int
        Number of erosion passes.

    Returns
    -------
    np.ndarray
        Eroded image.
    """
    if kernel is None:
        kernel = get_kernel("rect", (5, 5))
    result = cv2.erode(image, kernel, iterations=iterations)
    logger.info(f"Applied Erosion (iterations={iterations})")
    return result


def morphological_dilate(image: np.ndarray, kernel: np.ndarray = None,
                         iterations: int = 1) -> np.ndarray:
    """
    Dilation: Expands white regions / shrinks black regions.

    USE CASE: Fill small gaps in bubble outlines, strengthen boundaries.

    Parameters
    ----------
    image : np.ndarray
        Binary or grayscale image.
    kernel : np.ndarray
        Structuring element. If None, uses 5x5 rect.
    iterations : int
        Number of dilation passes.

    Returns
    -------
    np.ndarray
        Dilated image.
    """
    if kernel is None:
        kernel = get_kernel("rect", (5, 5))
    result = cv2.dilate(image, kernel, iterations=iterations)
    logger.info(f"Applied Dilation (iterations={iterations})")
    return result


def morphological_open(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Opening = Erosion → Dilation.

    USE CASE: Remove small noise while preserving larger shapes.
    Effectively removes small bright spots (noise) from the image
    without significantly affecting the size of larger objects.

    Parameters
    ----------
    image : np.ndarray
        Binary or grayscale image.
    kernel : np.ndarray
        Structuring element. If None, uses 5x5 rect.

    Returns
    -------
    np.ndarray
        Opened image.
    """
    if kernel is None:
        kernel = get_kernel("rect", (5, 5))
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    logger.info("Applied Morphological Opening (erosion → dilation)")
    return result


def morphological_close(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Closing = Dilation → Erosion.

    USE CASE: **Critical for speech bubbles** — closes small gaps and
    breaks in bubble outlines caused by stylized drawing or scan artifacts.
    This ensures that contour detection can find complete, closed contours.

    Parameters
    ----------
    image : np.ndarray
        Binary or grayscale image.
    kernel : np.ndarray
        Structuring element. If None, uses 7x7 ellipse.

    Returns
    -------
    np.ndarray
        Closed image.
    """
    if kernel is None:
        # Elliptical kernel works well for rounded bubble shapes
        kernel = get_kernel("ellipse", (7, 7))
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    logger.info("Applied Morphological Closing (dilation → erosion)")
    return result


def apply_morphology(image: np.ndarray, operations: list = None) -> np.ndarray:
    """
    Apply a sequence of morphological operations.

    This is the recommended way to prepare a binary image for contour
    detection on comic pages:
        1. Close gaps in bubble outlines (closing with elliptical kernel)
        2. Remove small noise particles (opening with small kernel)

    Parameters
    ----------
    image : np.ndarray
        Binary image.
    operations : list[tuple], optional
        List of (operation, kernel_shape, kernel_size, iterations).
        Default: [("close", "ellipse", (7,7), 2), ("open", "rect", (3,3), 1)]

    Returns
    -------
    np.ndarray
        Morphologically processed image.
    """
    if operations is None:
        operations = [
            ("close", "ellipse", (7, 7), 2),   # Close gaps in bubble outlines
            ("open", "rect", (3, 3), 1),        # Remove small noise
        ]

    result = image.copy()
    for op, shape, size, iters in operations:
        kernel = get_kernel(shape, size)
        if op == "erode":
            result = cv2.erode(result, kernel, iterations=iters)
        elif op == "dilate":
            result = cv2.dilate(result, kernel, iterations=iters)
        elif op == "open":
            for _ in range(iters):
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            for _ in range(iters):
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        logger.info(f"  Morphology step: {op} (kernel={shape} {size}, iters={iters})")

    return result


# ═══════════════════════════════════════════════════════════════
# G. FULL PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════
def preprocess_pipeline(image: np.ndarray, debug: bool = False,
                        save_dir: str = None) -> dict:
    """
    Run the complete preprocessing pipeline on a comic image.

    Strategy: Use an EDGE-BASED approach to detect bubble outlines.

    Speech bubbles are typically white/light regions enclosed by dark
    outlines on a colored/textured background. The pipeline:
        1. Convert to grayscale.
        2. Denoise with bilateral filter (preserves edges).
        3. Enhance contrast with CLAHE.
        4. Detect edges with Canny to find bubble outlines.
        5. Apply heavy morphological closing on edges to connect
           broken outline segments into closed contours.
        6. Flood-fill from the edges inward to create solid bubble masks.
        7. Also produce a threshold-based binary for comparison/fallback.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    debug : bool
        If True, display intermediate results.
    save_dir : str, optional
        Directory to save intermediate images.

    Returns
    -------
    dict
        Dictionary with keys:
            'original', 'gray', 'denoised', 'enhanced',
            'binary', 'edges', 'morphed'
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info("=" * 60)

    results = {"original": image.copy()}

    # Step 1: Grayscale
    gray = to_grayscale(image)
    results["gray"] = gray

    # Step 2: Noise Reduction (bilateral preserves bubble edges)
    denoised = bilateral_filter(gray)
    results["denoised"] = denoised

    # Step 3: Contrast Enhancement (CLAHE for uneven lighting)
    enhanced = clahe_enhancement(denoised, clip_limit=2.0)
    results["enhanced"] = enhanced

    # Step 4: Binarization — use Otsu for a clean global split
    # Speech bubbles are bright white; background is beige/colored
    # Otsu finds the optimal split between these two intensity modes
    blurred_for_thresh = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary_otsu = cv2.threshold(
        blurred_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    results["binary"] = binary_otsu
    logger.info("Applied Otsu binarization for bubble/background separation")

    # Step 5: Edge Detection (Canny on denoised image)
    edges = canny_edge(denoised, low=30, high=100)
    results["edges"] = edges

    # Step 6: Edge-based bubble mask creation
    # ────────────────────────────────────────────────────────────
    # Problem: Text inside bubbles creates edge artifacts that split
    # bubble regions into fragments. Solution: suppress text edges
    # BEFORE forming the bubble boundary barrier.
    h, w = edges.shape[:2]

    # 6a: Suppress internal text edges
    # Text edges are thin and short compared to bubble outlines.
    # Dilate edges first, then find contours and keep only large ones
    # (bubble outlines) while removing small ones (text edges).
    temp_dilate = cv2.dilate(edges, get_kernel("rect", (3, 3)), iterations=1)
    edge_contours, _ = cv2.findContours(
        temp_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Keep only edges with sufficient arc length (bubble outlines are long)
    # Text edges are short fragments; bubble outlines span the full shape
    min_edge_length = min(h, w) * 0.15  # At least 15% of image dimension
    clean_edges = np.zeros_like(edges)
    kept = 0
    for cnt in edge_contours:
        arc = cv2.arcLength(cnt, closed=False)
        if arc > min_edge_length:
            cv2.drawContours(clean_edges, [cnt], -1, 255, 2)
            kept += 1
    logger.info(f"Edge filtering: {len(edge_contours)} -> {kept} edges "
                f"(min_length={min_edge_length:.0f}px)")

    # 6b: Heavy morphological closing to connect edge segments
    edge_kernel = get_kernel("ellipse", (15, 15))
    closed_edges = cv2.morphologyEx(clean_edges, cv2.MORPH_CLOSE,
                                     edge_kernel, iterations=4)

    # 6c: Dilate to form thick continuous barriers
    dilate_kernel = get_kernel("ellipse", (9, 9))
    thick_edges = cv2.dilate(closed_edges, dilate_kernel, iterations=2)

    # 6d: Flood-fill from ALL border pixels to mark background
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = thick_edges.copy()

    for x in range(w):
        if filled[0, x] == 0:
            cv2.floodFill(filled, flood_mask, (x, 0), 255)
        if filled[h - 1, x] == 0:
            cv2.floodFill(filled, flood_mask, (x, h - 1), 255)
    for y in range(h):
        if filled[y, 0] == 0:
            cv2.floodFill(filled, flood_mask, (0, y), 255)
        if filled[y, w - 1] == 0:
            cv2.floodFill(filled, flood_mask, (w - 1, y), 255)

    # Bubbles = regions NOT reached by flood fill
    bubble_mask_from_edges = cv2.bitwise_not(filled)

    # 6e: Fill internal holes using contour filling
    contours_fill, _ = cv2.findContours(
        bubble_mask_from_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled_mask = np.zeros_like(bubble_mask_from_edges)
    cv2.drawContours(filled_mask, contours_fill, -1, 255, -1)
    logger.info(f"Created {len(contours_fill)} filled bubble regions")

    # Also create bubble mask from thresholding for reference/debug
    bubble_mask_from_thresh = binary_otsu.copy()

    # Use the EDGE-BASED filled mask as primary
    primary_mask = filled_mask

    # 6f: Final cleanup — heavy closing to merge remaining fragments
    morphed = apply_morphology(primary_mask, operations=[
        ("close", "ellipse", (15, 15), 3),   # Merge nearby fragments
        ("open", "rect", (7, 7), 1),          # Remove small noise blobs
    ])

    results["morphed"] = morphed

    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)

    # Debug visualization
    if debug:
        debug_images = [
            results["original"], results["gray"], results["denoised"],
            results["enhanced"], results["binary"], results["edges"],
            bubble_mask_from_edges, bubble_mask_from_thresh, results["morphed"]
        ]
        debug_titles = [
            "Original", "Grayscale", "Denoised (Bilateral)",
            "CLAHE Enhanced", "Otsu Binary", "Canny Edges",
            "Edge-Based Mask", "Threshold Mask", "Final Morphed"
        ]
        if save_dir:
            show_images(debug_images, debug_titles, cols=3,
                        save_path=f"{save_dir}/01_preprocessing.png")
            # Also save individual images
            for name, img in results.items():
                if name != "original":
                    from utils import save_image
                    save_image(img, f"{save_dir}/preprocess_{name}.png")
            from utils import save_image
            save_image(bubble_mask_from_edges, f"{save_dir}/preprocess_edge_mask.png")
            save_image(bubble_mask_from_thresh, f"{save_dir}/preprocess_thresh_mask.png")
        else:
            show_images(debug_images, debug_titles, cols=3)

    return results
