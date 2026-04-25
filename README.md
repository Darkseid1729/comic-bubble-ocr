# 🧠 Speech Bubble-Aware OCR System

**A Digital Image Processing (DIP) based system for extracting text from comic, manga, and manhwa images — bubble by bubble.**

---

## 📌 Problem Statement

Traditional OCR systems like Tesseract assume clean backgrounds, linear text flow, and standard fonts. When applied directly to comic images, they fail because:

- **Irregular layouts**: Comics have non-linear panel and bubble arrangements.
- **Multiple speech bubbles**: Text from adjacent bubbles gets merged, mixing dialogue from different speakers.
- **Stylized fonts**: Hand-drawn or decorative lettering confuses standard OCR models.
- **Embedded text**: Text is enclosed within graphical shapes (ellipses, spiky bursts, clouds).
- **Non-standard reading order**: Reading direction varies (left-to-right, right-to-left for manga).

**This system solves these problems** by detecting and isolating individual speech bubbles using classical DIP techniques *before* applying OCR, ensuring text from each bubble is extracted independently and accurately.

---

## 🏗️ System Architecture

```
Input Image
  → Preprocessing (Grayscale, Denoise, CLAHE, Binarize, Edge Detect, Morphology)
  → Segmentation (Contour Detection, Connected Components, Watershed)
  → Detection (Shape Analysis, Multi-Criteria Filtering, Type Classification)
  → ROI Extraction & Reading Order
  → OCR (Per-Bubble Tesseract with Fallback PSM Modes)
  → Post-Processing (Text Cleanup, Line Merging)
  → Output (Annotated Image + JSON + Text File)
```

---

## 📂 Project Structure

```
project2 ocr/
├── main.py              # Main pipeline orchestration + CLI
├── api.py               # FastAPI server for uploads + results
├── preprocessing.py     # Grayscale, denoising, contrast, binarization, edges, morphology
├── segmentation.py      # Contour detection, connected components, watershed
├── detection.py         # Shape analysis, bubble classification, ROI extraction
├── ocr.py               # Tesseract integration, text cleanup, evaluation metrics
├── utils.py             # Image I/O, visualization, JSON output, annotations
├── evaluate.py          # Accuracy evaluation helper
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── frontend/            # React UI (Vite)
├── samples/             # Sample comic images for testing
└── outputs/             # Generated outputs (annotated images, JSON, text)
```

---

## 🔬 DIP Techniques Used

### A. Preprocessing
| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| Grayscale Conversion | Reduce to single channel | Always (first step) |
| Gaussian Blur | General noise removal | Mild, uniform noise |
| Median Filter | Salt-and-pepper noise | Scanned pages with speckles |
| Bilateral Filter | Edge-preserving denoising | **Best for comics** — preserves bubble outlines |
| Histogram Equalization | Global contrast boost | Uniform lighting, low contrast |
| CLAHE | Adaptive contrast enhancement | **Preferred** — handles uneven scan lighting |

### B. Binarization
| Method | When to Use |
|--------|-------------|
| Global Threshold | Uniform lighting, known threshold |
| Adaptive Mean | Varying lighting, simple scenes |
| Adaptive Gaussian | **Best for comics** — smooth local adaptation |
| Otsu's Method | Bimodal histogram, unknown threshold |

### C. Edge Detection
| Method | Description |
|--------|-------------|
| Canny | Hysteresis-based, thin edges (primary) |
| Sobel | Gradient magnitude (comparison) |

### D. Morphological Operations
| Operation | Effect | Use Case |
|-----------|--------|----------|
| Erosion | Shrinks white regions | Separate touching objects |
| Dilation | Expands white regions | Fill gaps in outlines |
| Opening | Erosion → Dilation | Remove small noise |
| Closing | Dilation → Erosion | **Close broken bubble boundaries** |

### E. Segmentation
- **Contour Detection** (`cv2.findContours`)
- **Connected Component Analysis** (pixel-level region labeling)
- **Watershed Algorithm** (separate touching/overlapping bubbles)

### F. Shape Analysis
| Feature | Formula | Purpose |
|---------|---------|---------|
| Area | `cv2.contourArea` | Size filtering |
| Perimeter | `cv2.arcLength` | Shape complexity |
| Circularity | `C = 4πA / P²` | Circle-likeness (1.0 = circle) |
| Aspect Ratio | `W / H` | Shape elongation |
| Solidity | `Area / ConvexHullArea` | Convexity measure |
| Extent | `Area / BboxArea` | Fill ratio |

---

## 🧠 Bubble Type Detection

The system classifies detected bubbles into four types based on geometric features:

| Type | Characteristics | Visual |
|------|----------------|--------|
| **Speech** | High circularity, high solidity, good ellipse fit | Smooth oval |
| **Shout** | Many vertices, low solidity, jagged edges | Spiky burst |
| **Thought** | Moderate circularity, many vertices, rounded | Cloud-like |
| **Narration** | High rectangularity, few vertices, high solidity | Rectangular box |

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Tesseract OCR** installed on your system

#### Install Tesseract:

**Windows:**
```bash
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the System

```bash
# Single image
python main.py --input samples/comic_page.png

# With debug visualizations
python main.py --input samples/comic_page.png --debug

# Batch processing
python main.py --input_dir samples/ --output_dir results/

# Japanese manga
python main.py --input manga.jpg --lang jpn

# Custom minimum bubble area
python main.py --input page.png --min_area 1000

# Enable watershed for overlapping bubbles
python main.py --input page.png --watershed
```

### 4. Running the Standalone Executable
If you have built the application into a standalone `.exe` using PyInstaller:

1. Navigate to the `dist/` directory.
2. Double-click `app_launcher.exe`.
3. The backend will start silently, and your default web browser will open the UI.
4. **Closing the App:** Use the red **Close App** button in the UI header to safely terminate both the frontend and the background server process.

---

## 🛠️ How to Build (from source)

To create your own standalone executable:

### 1. Build the Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 2. Build the Python Bundle
Ensure your virtual environment is active, then run:
```bash
.\.venv\Scripts\pyinstaller --clean app_launcher.spec
```
The finished executable will be in the `dist/` folder.

---

## 🧩 Full Stack Setup (API + React UI)

### 1. Start the API server

```bash
uvicorn api:app --reload --port 8000
```

### 2. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Optional: set a custom API URL in [frontend/.env.example](frontend/.env.example)
by copying it to `frontend/.env` and editing the value.

---

## 🧪 Evaluation Helper

Compare OCR output to ground-truth text:

```bash
# Evaluate against a text file
python evaluate.py --gt ground_truth.txt --ocr outputs/your_image/extracted_text.txt

# Evaluate against JSON output
python evaluate.py --gt ground_truth.txt --ocr_json outputs/your_image/ocr_results.json
```

---

## 📦 Output Format

### Annotated Image
- Color-coded bounding boxes per bubble type
- Bubble IDs and type labels
- Text preview below each bubble

### JSON Output (`ocr_results.json`)
```json
[
    {
        "bubble_id": 1,
        "bubble_type": "speech",
        "text": "Hello! How are you?",
        "bbox": [120, 45, 200, 150]
    },
    {
        "bubble_id": 2,
        "bubble_type": "shout",
        "text": "WATCH OUT!",
        "bbox": [350, 60, 180, 120]
    }
]
```

### Text File (`extracted_text.txt`)
```
── Bubble #1 [speech] ──
Hello! How are you?

── Bubble #2 [shout] ──
WATCH OUT!
```

---

## 🐞 Debugging Guide

| Problem | Solution |
|---------|----------|
| Too many contours detected | Increase `--min_area`, tighten morphological closing |
| Bubbles not detected | Lower `--min_area`, check if image needs inversion |
| Adjacent bubbles merged | Enable `--watershed` flag |
| OCR output is garbage | Check binarization quality, try different CLAHE parameters |
| Dark background text missed | System auto-inverts, but check ROI preprocessing |
| Skewed text | Deskew is automatic via Hough Transform |

Run with `--debug` to save intermediate images at each stage for inspection.

---

## 📊 Evaluation

The system automatically compares:
1. **Baseline**: Raw OCR on the full image (text gets mixed)
2. **Bubble-Aware**: OCR per detected bubble (text stays separated)

Metrics available:
- Character Accuracy
- Word Accuracy
- Bubble Detection Count

---

## ⚠️ Edge Cases Handled

| Case | Handling |
|------|----------|
| Small text | ROI auto-upscaled to ~300 DPI equivalent |
| Dark backgrounds | Auto-inversion when mean intensity < 128 |
| Colored comics | Converted to grayscale / HSV as needed |
| Scan noise | Bilateral filtering + morphological opening |
| Broken bubble outlines | Morphological closing with elliptical kernel |
| Overlapping bubbles | Watershed algorithm (optional) |

---

## 🌍 Applications

- **Comic Translation**: Extract text → translate → re-insert
- **Accessibility**: Screen readers for visually impaired users
- **Digital Archiving**: Index and search comic collections
- **Content Analysis**: Study dialogue patterns, word frequency

### 🌟 Personal Future Vision: AI Manhwa Explainer Automation

The ultimate goal of this project is to integrate this OCR pipeline with a Large Language Model (LLM) agent (like Chatterbox) to fully automate the creation of "Manhwa Explanation" YouTube videos. 

**The Automated Workflow:**
1. **Upload:** User uploads a chapter of a manhwa.
2. **Extraction (This Project):** Extract dialogue, narration, and panel sequence.
3. **Analysis:** The LLM comprehends the plot, character interactions, and narrative tone.
4. **Video Generation:** The LLM generates a recap script, passes it to a Text-to-Speech (TTS) engine, and syncs the audio with panel highlights to automatically produce an engaging YouTube video.

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| OpenCV | Image processing & computer vision |
| NumPy | Numerical operations |
| Matplotlib | Debug visualization |
| Tesseract OCR | Text extraction |
| Pillow | Image format support |

---

## 📝 Notes

- This system uses **only classical DIP techniques** — no deep learning models.
- All techniques are **explainable and interpretable**.
- The modular design allows easy swapping of any pipeline stage.
- Parameters are **auto-tuned** based on image resolution but can be manually overridden.

---

## 📄 License

This project is developed for academic purposes as part of a Digital Image Processing course.
#   c o m i c - b u b b l e - o c r  
 