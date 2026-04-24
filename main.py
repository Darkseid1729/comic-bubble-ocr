"""
main.py - Speech Bubble-Aware OCR System (Main Pipeline)
==========================================================
Orchestrates the complete DIP-based pipeline:

    Input Image
    → Preprocessing (grayscale, denoise, enhance, binarize, edge, morph)
    → Segmentation (contours, connected components, watershed)
    → Detection (shape analysis, filtering, classification)
    → ROI Extraction & Reading Order
    → OCR (per-bubble Tesseract with fallback)
    → Post-processing (text cleanup, merging)
    → Output (Annotated Image + JSON)

Usage:
    python main.py                                  # Process sample images
    python main.py --input path/to/image.png        # Single image
    python main.py --input_dir path/to/folder/      # Batch processing
    python main.py --input image.png --debug        # With debug output
    python main.py --input image.png --lang jpn     # Japanese manga

Author: Speech Bubble-Aware OCR System (DIP-Based)
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

# ── Local modules ──
from utils import (
    logger, load_image, save_image, save_json,
    draw_annotated_output, show_images, ensure_dirs, get_output_dir
)
from preprocessing import preprocess_pipeline
from segmentation import segmentation_pipeline
from detection import detection_pipeline
from ocr import ocr_pipeline


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def process_image(image_path: str,
                  output_dir: str = "outputs",
                  debug: bool = False,
                  lang: str = "eng",
                  min_bubble_area: int = 500,
                  use_watershed: bool = False) -> dict:
    """
    Process a single comic image through the full pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input comic/manga image.
    output_dir : str
        Base directory for saving outputs.
    debug : bool
        If True, save intermediate debug visualizations.
    lang : str
        OCR language code ('eng', 'jpn', etc.).
    min_bubble_area : int
        Minimum contour area to consider as a bubble.
        Tune this based on image resolution:
            - Low-res (< 800px): 200-500
            - Medium (800-2000px): 500-2000
            - High-res (> 2000px): 2000-5000
    use_watershed : bool
        Apply watershed for touching/overlapping bubbles.

    Returns
    -------
    dict
        Complete pipeline results including:
            'image_path', 'results', 'annotated_image',
            'output_dir', 'processing_time'
    """
    start_time = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  SPEECH BUBBLE-AWARE OCR SYSTEM                        ║")
    logger.info("║  DIP-Based Pipeline                                    ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"Input: {image_path}")

    # ── Create output directory ──
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_dir, basename)
    ensure_dirs(save_dir)

    # ── Load image ──
    original = load_image(image_path)

    # ── Auto-tune min_area based on image resolution ──
    h, w = original.shape[:2]
    image_area = h * w
    if min_bubble_area == 500:  # Default, auto-tune
        # Bubbles typically occupy 0.5% to 15% of image area
        min_bubble_area = max(200, int(image_area * 0.002))
        logger.info(f"Auto-tuned min_bubble_area = {min_bubble_area} "
                    f"(image: {w}x{h}, area: {image_area})")

    # ══════════════════════════════════════════════════════════
    # STAGE 1: PREPROCESSING
    # ══════════════════════════════════════════════════════════
    logger.info("\n▶ STAGE 1: PREPROCESSING")
    preprocess_results = preprocess_pipeline(
        original, debug=debug, save_dir=save_dir if debug else None
    )

    # ══════════════════════════════════════════════════════════
    # STAGE 2: SEGMENTATION
    # ══════════════════════════════════════════════════════════
    logger.info("\n▶ STAGE 2: SEGMENTATION")
    seg_results = segmentation_pipeline(
        original=original,
        morphed=preprocess_results["morphed"],
        min_area=min_bubble_area,
        use_watershed=use_watershed,
        debug=debug,
        save_dir=save_dir if debug else None
    )

    # ══════════════════════════════════════════════════════════
    # STAGE 3: DETECTION (Shape Analysis + Classification)
    # ══════════════════════════════════════════════════════════
    logger.info("\n▶ STAGE 3: DETECTION")
    det_results = detection_pipeline(
        original=original,
        contours=seg_results["filtered_contours"],
        min_area=min_bubble_area,
        debug=debug,
        save_dir=save_dir if debug else None
    )

    candidates = det_results["candidates"]

    if not candidates:
        logger.warning("No speech bubbles detected! Consider adjusting parameters:")
        logger.warning("  - Lower min_bubble_area")
        logger.warning("  - Adjust morphological kernel sizes")
        logger.warning("  - Check if image needs inversion")

        # Try with inverted image as fallback
        logger.info("Attempting with inverted binary image...")
        morphed_inv = cv2.bitwise_not(preprocess_results["morphed"])
        seg_results_inv = segmentation_pipeline(
            original=original,
            morphed=morphed_inv,
            min_area=min_bubble_area,
            debug=False
        )
        det_results = detection_pipeline(
            original=original,
            contours=seg_results_inv["filtered_contours"],
            min_area=min_bubble_area,
            debug=debug,
            save_dir=save_dir if debug else None
        )
        candidates = det_results["candidates"]
        if candidates:
            logger.info(f"Inversion recovered {len(candidates)} bubbles!")

    # ══════════════════════════════════════════════════════════
    # STAGE 4: OCR
    # ══════════════════════════════════════════════════════════
    logger.info("\n▶ STAGE 4: OCR")
    ocr_results = ocr_pipeline(
        candidates=candidates,
        original=original,
        lang=lang,
        debug=debug,
        save_dir=save_dir if debug else None
    )

    # Filter out empty/trivial-text results (likely false positive detections)
    # Single characters are typically noise from bubble tails or artifacts
    ocr_results = [r for r in ocr_results if len(r["text"].strip()) >= 2]
    # Re-number bubble IDs after filtering
    for i, r in enumerate(ocr_results):
        r["bubble_id"] = i + 1
    logger.info(f"After text filtering: {len(ocr_results)} bubbles with meaningful text")

    # ══════════════════════════════════════════════════════════
    # STAGE 5: OUTPUT GENERATION
    # ══════════════════════════════════════════════════════════
    logger.info("\n▶ STAGE 5: OUTPUT GENERATION")

    # Draw final annotated image
    annotated = draw_annotated_output(original, ocr_results)

    # Save outputs
    save_image(annotated, os.path.join(save_dir, "annotated_output.png"))
    save_json(ocr_results, os.path.join(save_dir, "ocr_results.json"))

    # Also save a clean text file with all extracted text
    text_output_path = os.path.join(save_dir, "extracted_text.txt")
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(f"Speech Bubble-Aware OCR Results\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"{'=' * 50}\n\n")
        for r in ocr_results:
            f.write(f"── Bubble #{r['bubble_id']} [{r['bubble_type']}] ──\n")
            f.write(f"{r['text']}\n\n")
    logger.info(f"Saved text output: {text_output_path}")

    elapsed = time.time() - start_time

    # ── Summary ──
    logger.info("\n" + "═" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Image: {image_path}")
    logger.info(f"  Bubbles detected: {len(ocr_results)}")
    logger.info(f"  Total text extracted: {sum(len(r['text']) for r in ocr_results)} chars")
    logger.info(f"  Processing time: {elapsed:.2f}s")
    logger.info(f"  Output directory: {save_dir}")
    logger.info("═" * 60)

    return {
        "image_path": image_path,
        "results": ocr_results,
        "annotated_image": annotated,
        "output_dir": save_dir,
        "processing_time": elapsed,
        "num_bubbles": len(ocr_results),
    }


# ═══════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════
def batch_process(input_dir: str, output_dir: str = "outputs",
                  debug: bool = False, lang: str = "eng") -> list:
    """
    Process all images in a directory.

    Supports: .png, .jpg, .jpeg, .bmp, .tiff, .webp

    Parameters
    ----------
    input_dir : str
        Directory containing comic images.
    output_dir : str
        Base output directory.
    debug : bool
        Enable debug visualizations.
    lang : str
        OCR language.

    Returns
    -------
    list[dict]
        Results for each processed image.
    """
    supported = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not image_files:
        logger.error(f"No supported images found in {input_dir}")
        return []

    logger.info(f"Batch processing: {len(image_files)} images from {input_dir}")

    all_results = []
    for i, img_path in enumerate(image_files):
        logger.info(f"\n{'━' * 60}")
        logger.info(f"Processing [{i+1}/{len(image_files)}]: {os.path.basename(img_path)}")
        logger.info(f"{'━' * 60}")

        try:
            result = process_image(
                img_path, output_dir=output_dir,
                debug=debug, lang=lang
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            import traceback
            traceback.print_exc()

    # ── Batch Summary ──
    total_bubbles = sum(r["num_bubbles"] for r in all_results)
    total_time = sum(r["processing_time"] for r in all_results)

    logger.info(f"\n{'═' * 60}")
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"  Images processed: {len(all_results)}/{len(image_files)}")
    logger.info(f"  Total bubbles detected: {total_bubbles}")
    logger.info(f"  Total processing time: {total_time:.2f}s")
    logger.info(f"  Average per image: {total_time/len(all_results):.2f}s")
    logger.info(f"{'═' * 60}")

    return all_results


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    """
    Command-line interface for the Speech Bubble-Aware OCR System.
    """
    parser = argparse.ArgumentParser(
        description="Speech Bubble-Aware OCR System (DIP-Based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input comic_page.png
  python main.py --input comic_page.png --debug
  python main.py --input_dir ./samples/ --output_dir ./results/
  python main.py --input manga.jpg --lang jpn
  python main.py --input page.png --min_area 1000

Parameter Tuning:
  --min_area    Minimum bubble area (pixels). Lower = more detections.
                Default auto-tunes based on image resolution.
  --debug       Saves intermediate images for every processing stage.
  --watershed   Enable watershed for overlapping/touching bubbles.
        """
    )

    parser.add_argument("--input", "-i", type=str,
                        help="Path to a single input image.")
    parser.add_argument("--input_dir", "-d", type=str,
                        help="Directory of images for batch processing.")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs",
                        help="Output directory (default: outputs/).")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug visualizations.")
    parser.add_argument("--lang", type=str, default="eng",
                        help="OCR language (default: eng). Use 'jpn' for manga.")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Minimum bubble area in pixels.")
    parser.add_argument("--watershed", action="store_true",
                        help="Enable watershed for touching bubbles.")

    args = parser.parse_args()

    # ── Validate inputs ──
    if not args.input and not args.input_dir:
        # Default: look for samples directory
        if os.path.isdir("samples"):
            logger.info("No input specified. Processing ./samples/ directory...")
            args.input_dir = "samples"
        else:
            parser.print_help()
            print("\n[ERROR] Please provide --input or --input_dir")
            print("   Or create a 'samples/' directory with comic images.")
            sys.exit(1)

    # ── Process ──
    if args.input:
        if not os.path.isfile(args.input):
            print(f"[ERROR] File not found: {args.input}")
            sys.exit(1)

        result = process_image(
            args.input,
            output_dir=args.output_dir,
            debug=args.debug,
            lang=args.lang,
            min_bubble_area=args.min_area,
            use_watershed=args.watershed,
        )

        print(f"\n[DONE] {result['num_bubbles']} bubbles detected.")
        print(f"[OUTPUT] Saved to: {result['output_dir']}")

    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[ERROR] Directory not found: {args.input_dir}")
            sys.exit(1)

        results = batch_process(
            args.input_dir,
            output_dir=args.output_dir,
            debug=args.debug,
            lang=args.lang,
        )

        total = sum(r["num_bubbles"] for r in results)
        print(f"\n[DONE] Batch complete! {total} bubbles across {len(results)} images.")
        print(f"[OUTPUT] Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
