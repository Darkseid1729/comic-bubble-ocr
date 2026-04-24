"""
evaluate.py - Simple OCR evaluation helper

Compare OCR output against ground-truth text using character and word accuracy.
"""

import argparse
from pathlib import Path

from ocr import character_accuracy, word_accuracy
from utils import load_json


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_text_from_json(path: Path) -> str:
    data = load_json(str(path))
    lines = []
    for item in data:
        text = item.get("text", "")
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR output.")
    parser.add_argument("--gt", required=True, help="Ground truth text file.")
    parser.add_argument("--ocr", help="OCR output text file.")
    parser.add_argument("--ocr_json", help="OCR JSON file (ocr_results.json).")

    args = parser.parse_args()

    if not args.ocr and not args.ocr_json:
        raise SystemExit("Provide --ocr or --ocr_json")

    gt_text = load_text(Path(args.gt))

    if args.ocr_json:
        pred_text = load_text_from_json(Path(args.ocr_json))
    else:
        pred_text = load_text(Path(args.ocr))

    char_acc = character_accuracy(pred_text, gt_text)
    word_acc = word_accuracy(pred_text, gt_text)

    print("Evaluation results")
    print("------------------")
    print(f"Character accuracy: {char_acc:.3f}")
    print(f"Word accuracy: {word_acc:.3f}")


if __name__ == "__main__":
    main()
