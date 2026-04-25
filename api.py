"""
api.py - FastAPI backend for Bubble OCR Studio

Provides a simple HTTP API for uploading images and receiving
bubble-aware OCR results with annotated image artifacts.
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

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


def _get_deepl_endpoint(api_key: str) -> str:
    if api_key.endswith(":fx"):
        return "https://api-free.deepl.com/v2/translate"
    return "https://api.deepl.com/v2/translate"


def _translate_text_deepl(text: str, target_lang: str, source_lang: str | None = None) -> str:
    try:
        import requests
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Translation requires the 'requests' package"
        ) from exc

    api_key = os.environ.get("DEEPL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="DEEPL_API_KEY is not set")

    endpoint = os.environ.get("DEEPL_API_URL") or _get_deepl_endpoint(api_key)
    payload = {
        "auth_key": api_key,
        "text": text,
        "target_lang": target_lang
    }
    if source_lang:
        payload["source_lang"] = source_lang

    try:
        response = requests.post(endpoint, data=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        translations = data.get("translations", [])
        if not translations:
            return ""
        return translations[0].get("text", "")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Translation failed: {exc}") from exc


def _prune_items(base_dir: Path, keep: int = 3) -> None:
    if not base_dir.exists():
        return

    items = [item for item in base_dir.iterdir() if item.is_dir() or item.is_file()]
    if len(items) <= keep:
        return

    items.sort(key=lambda item: item.stat().st_mtime)
    for item in items[:-keep]:
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except Exception:
            pass


app = FastAPI(title=APP_TITLE, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
app.mount("/assets/outputs", StaticFiles(directory=str(OUTPUT_BASE)), name="outputs")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/shutdown")
def shutdown():
    """
    Terminates the server process.
    """
    import os
    # Terminate process immediately
    os._exit(0)


@app.post("/api/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    lang: str = Form("eng"),
    min_area: int = Form(500),
    watershed: bool = Form(False),
    debug: bool = Form(False),
    translate: bool = Form(False),
    target_lang: str = Form("EN")
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
            confidence = item.get("confidence", 0.0)
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

        if translate:
            source_lang = "JA" if lang.lower().startswith("jp") or lang.lower().startswith("ja") else None
            for bubble in bubbles:
                text = bubble.get("text", "").strip()
                if not text:
                    bubble["translated_text"] = ""
                    continue
                bubble["translated_text"] = _translate_text_deepl(
                    text=text,
                    target_lang=target_lang,
                    source_lang=source_lang
                )

            if json_path.exists():
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        import json
                        json.dump(
                            [
                                {
                                    "bubble_id": b["id"],
                                    "bubble_type": b["type"],
                                    "text": b["text"],
                                    "translated_text": b.get("translated_text", ""),
                                    "bbox": b["bbox"],
                                    "confidence": b.get("confidence", 0.0)
                                }
                                for b in bubbles
                            ],
                            f,
                            indent=4,
                            ensure_ascii=False
                        )
                except Exception:
                    pass

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
            "translation": {
                "enabled": translate,
                "target_lang": target_lang,
                "provider": "deepl" if translate else ""
            },
            "artifacts": {
                "annotated_image": _build_asset_url(annotated_path),
                "input_image": _build_asset_url(input_copy_path),
                "ocr_json": _build_asset_url(json_path),
                "text_output": _build_asset_url(text_path) if text_path.exists() else "",
                "pipeline": _collect_pipeline_images(output_dir)
            }
        }

        _prune_items(API_OUTPUT_DIR, keep=3)
        _prune_items(UPLOAD_DIR, keep=3)

        return response
    except Exception as exc:
        with open("api_error.log", "w") as f:
            import traceback
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _get_ui_dir() -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    packaged = base_path / "frontend" / "dist"
    local = Path(__file__).resolve().parent / "frontend" / "dist"
    return packaged if packaged.exists() else local


UI_DIR = _get_ui_dir()

if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
else:
    @app.get("/")
    def ui_missing():
        return JSONResponse(
            {
                "error": "UI build not found.",
                "hint": "Run 'npm run build' inside the frontend folder."
            },
            status_code=503
        )
