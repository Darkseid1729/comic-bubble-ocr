"""
api.py - FastAPI backend for Bubble OCR Studio

Provides a simple HTTP API for uploading images and receiving
bubble-aware OCR results with annotated image artifacts.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from main import process_image


APP_TITLE = "Bubble OCR Studio API"
OUTPUT_BASE = Path("outputs")
UPLOAD_DIR = OUTPUT_BASE / "uploads"
API_OUTPUT_DIR = OUTPUT_BASE / "api"


def _safe_name(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return base.strip("_") or "upload"


def _build_asset_url(path: Path) -> str:
    rel = path.relative_to(OUTPUT_BASE).as_posix()
    return f"/assets/outputs/{rel}"


def _collect_pipeline_images(output_dir: Path) -> list:
    stages = [
        ("Grayscale", "preprocess_gray.png"),
        ("Binarized", "preprocess_binary.png"),
        ("Edges", "preprocess_edges.png"),
        ("Segmentation", "02_segmentation.png"),
        ("Detections", "03_detection_overview.png")
    ]

    images = []
    for title, filename in stages:
        path = output_dir / filename
        if path.exists():
            images.append({"title": title, "image": _build_asset_url(path)})
    return images


app = FastAPI(title=APP_TITLE, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/assets/outputs", StaticFiles(directory=str(OUTPUT_BASE)), name="outputs")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    lang: str = Form("eng"),
    min_area: int = Form(500),
    watershed: bool = Form(False),
    debug: bool = Form(False)
):
    """
    Upload an image and run bubble-aware OCR.

    Returns:
        - bubbles: list of OCR results
        - metrics: summary stats
        - artifacts: URLs for annotated image, input image, JSON
    """
    OUTPUT_BASE.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "upload.png")
    safe_stem = _safe_name(original_name.stem)
    suffix = original_name.suffix if original_name.suffix else ".png"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{safe_stem}_{run_id}"

    upload_path = UPLOAD_DIR / f"{base_name}{suffix}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = process_image(
            image_path=str(upload_path),
            output_dir=str(API_OUTPUT_DIR),
            debug=debug,
            lang=lang,
            min_bubble_area=min_area,
            use_watershed=watershed
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_dir = Path(result["output_dir"])
    input_copy_path = output_dir / "input.png"
    shutil.copyfile(upload_path, input_copy_path)

    annotated_path = output_dir / "annotated_output.png"
    json_path = output_dir / "ocr_results.json"
    text_path = output_dir / "extracted_text.txt"

    bubbles = []
    confidences = []
    type_counts = {}

    for item in result["results"]:
        confidence = item.get("confidence")
        if confidence is None:
            confidence = 0.0
        confidences.append(confidence)

        bubble_type = item.get("bubble_type", "unknown")
        type_counts[bubble_type] = type_counts.get(bubble_type, 0) + 1

        bubbles.append({
            "id": item.get("bubble_id"),
            "type": bubble_type,
            "text": item.get("text", ""),
            "bbox": item.get("bbox", []),
            "confidence": confidence
        })

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    char_count = sum(len(b["text"]) for b in bubbles)

    response = {
        "run_id": base_name,
        "image_name": original_name.name,
        "processing_time": result.get("processing_time", 0.0),
        "metrics": {
            "bubbles": len(bubbles),
            "chars": char_count,
            "avgConfidence": avg_conf,
            "typeCounts": type_counts
        },
        "bubbles": bubbles,
        "artifacts": {
            "annotated_image": _build_asset_url(annotated_path),
            "input_image": _build_asset_url(input_copy_path),
            "ocr_json": _build_asset_url(json_path),
            "text_output": _build_asset_url(text_path) if text_path.exists() else "",
            "pipeline": _collect_pipeline_images(output_dir)
        }
    }

    return response
