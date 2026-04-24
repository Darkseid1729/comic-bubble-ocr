"""
utils.py - Utility Functions for Speech Bubble-Aware OCR System
================================================================
Provides helper functions for:
    - Image loading and saving
    - Debug visualization (matplotlib-based)
    - JSON output generation
    - Directory management
    - Color palette for annotations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BubbleOCR")


# ──────────────────────────────────────────────────────────────
# Color Palette for Bubble Annotations
# ──────────────────────────────────────────────────────────────
# BGR format for OpenCV drawing
BUBBLE_COLORS = {
    "speech":    (0, 255, 0),      # Green
    "shout":     (0, 0, 255),      # Red
    "thought":   (255, 165, 0),    # Orange
    "narration": (255, 0, 255),    # Magenta
    "unknown":   (255, 255, 0),    # Cyan
}

BUBBLE_TYPE_LABELS = {
    "speech":    "Speech",
    "shout":     "Shout!",
    "thought":   "Thought",
    "narration": "Narration",
    "unknown":   "Bubble",
}


# ──────────────────────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        BGR image as loaded by OpenCV.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    logger.info(f"Loaded image: {path} | Shape: {img.shape}")
    return img


def save_image(img: np.ndarray, path: str) -> str:
    """
    Save an image to disk, creating parent directories if needed.

    Parameters
    ----------
    img : np.ndarray
        Image array (BGR or grayscale).
    path : str
        Destination file path.

    Returns
    -------
    str
        The path the image was saved to.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)
    logger.info(f"Saved image: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# Visualization / Debug Helpers
# ──────────────────────────────────────────────────────────────
def show_images(images: list, titles: list, cols: int = 3,
                figsize: tuple = (18, 8), save_path: str = None):
    """
    Display a grid of images using matplotlib.

    Parameters
    ----------
    images : list[np.ndarray]
        List of images to display.
    titles : list[str]
        Corresponding titles for each image.
    cols : int
        Number of columns in the grid.
    figsize : tuple
        Figure size (width, height) in inches.
    save_path : str, optional
        If provided, save the figure to this path instead of showing.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes for easy indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            img = images[i]
            # Convert BGR to RGB for matplotlib if color image
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(titles[i], fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved debug visualization: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def show_single(img: np.ndarray, title: str = "Image",
                cmap: str = "gray", figsize: tuple = (10, 8)):
    """Display a single image with matplotlib."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 3 and img.shape[2] == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_contours_debug(image: np.ndarray, contours: list,
                        title: str = "Contours") -> np.ndarray:
    """
    Draw all contours on a copy of the image for debugging.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    contours : list
        List of contours from cv2.findContours.
    title : str
        Title string for logging.

    Returns
    -------
    np.ndarray
        Image with contours drawn on it.
    """
    canvas = image.copy()
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    logger.info(f"[{title}] Drew {len(contours)} contours")
    return canvas


# ──────────────────────────────────────────────────────────────
# JSON Output
# ──────────────────────────────────────────────────────────────
def save_json(data: list, path: str) -> str:
    """
    Save structured OCR results to a JSON file.

    Expected format per entry:
    {
        "bubble_id": int,
        "bubble_type": str,
        "text": str,
        "bbox": [x, y, w, h]
    }

    Parameters
    ----------
    data : list[dict]
        List of bubble result dictionaries.
    path : str
        Destination JSON file path.

    Returns
    -------
    str
        The path the JSON was saved to.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved JSON output: {path} ({len(data)} bubbles)")
    return path


def load_json(path: str) -> list:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# Annotation Drawing
# ──────────────────────────────────────────────────────────────
def draw_annotated_output(image: np.ndarray, results: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image for final output.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    results : list[dict]
        List of dicts with keys: bubble_id, bubble_type, text, bbox.

    Returns
    -------
    np.ndarray
        Annotated image copy.
    """
    annotated = image.copy()

    for res in results:
        bid = res["bubble_id"]
        btype = res.get("bubble_type", "unknown")
        text = res.get("text", "")
        x, y, w, h = res["bbox"]

        # Pick color based on bubble type
        color = BUBBLE_COLORS.get(btype, BUBBLE_COLORS["unknown"])
        label = BUBBLE_TYPE_LABELS.get(btype, "Bubble")

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        label_text = f"#{bid} {label}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(
            annotated, label_text, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw extracted text preview (first 40 chars)
        preview = text[:40].replace("\n", " ")
        if preview:
            cv2.putText(
                annotated, preview, (x + 2, y + h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )

    return annotated


# ──────────────────────────────────────────────────────────────
# Directory Helpers
# ──────────────────────────────────────────────────────────────
def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_output_dir(base: str = "outputs") -> str:
    """
    Create a timestamped output directory inside `base/`.

    Returns
    -------
    str
        Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ──────────────────────────────────────────────────────────────
# Image Conversion Helpers
# ──────────────────────────────────────────────────────────────
def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale to BGR if needed."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_for_display(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Resize an image so its largest dimension is at most max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
