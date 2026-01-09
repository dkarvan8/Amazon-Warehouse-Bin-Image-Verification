# Amazon Bin Image Verification System

Automated warehouse bin verification using YOLOv8 object detection, CLIP embeddings, and OCR.

**Live Demo**: https://hridayx-amazon-warehouse-bin-image-verification-app-ykphri.streamlit.app/

---

## Overview

This system automates inventory verification in Amazon fulfillment centers by matching detected items in bin images against expected inventory metadata. It combines computer vision and multimodal learning to achieve robust verification despite packaging variations and occlusions.

**Pipeline**: YOLOv8 Detection → CLIP Visual Matching → OCR Text Extraction → Ensemble Validation

---

## Features

- **Multimodal Matching**: Combines CLIP visual embeddings (70%) and OCR text extraction (30%)
- **Quantity-Aware Validation**: Handles multiple quantities and filters duplicate detections
- **Confidence Scoring**: HIGH (both agree), MEDIUM (CLIP only), LOW (uncertain)
- **Dual Interface**: Pre-computed showcase results + live processing with reference scraping

---

## Technology Stack

- **Detection**: YOLOv8 (Ultralytics)
- **Visual Matching**: CLIP (OpenAI ViT-B/32)
- **Text Recognition**: EasyOCR
- **Web Scraping**: Selenium
- **Framework**: PyTorch, Streamlit

---

## Installation

```bash
git clone https://github.com/Hridayx/Amazon-Warehouse-Bin-Image-Verification.git
cd Amazon-Warehouse-Bin-Image-Verification
pip install -r requirements.txt
streamlit run app.py
```

**Requirements**: Python 3.11+, Chrome/Chromium, 4GB RAM

---

## Usage

### Showcase Mode
Select a bin from dropdown to view pre-computed verification results.

### Live Processing
1. Upload bin image and metadata JSON
2. Click "Process Bin"
3. System scrapes references, detects objects, matches with CLIP+OCR
4. Download results

---

## Performance

| Metric | Value |
|--------|-------|
| Average Accuracy | 38.89% |
| Precision | 100% |
| Recall | 38% |

**Analysis**: Zero false positives. Low recall due to missing references and packaging occlusions.

---

## Project Structure

```
├── app.py                    Streamlit interface
├── embeddings_matcher.py     CLIP + OCR matching
├── ensemble_validator.py     Validation logic
├── scraper.py                Reference scraper
├── best.pt                   YOLOv8 weights (89MB)
└── Display/                  Showcase results
```

---

## Methodology

1. **YOLOv8**: Detects objects with 0.25 confidence threshold
2. **CLIP**: 512-dim embeddings matched via cosine similarity
3. **OCR**: Text extraction with fuzzy matching (70% threshold)
4. **Ensemble**: Weighted voting with quantity filtering

---

## Limitations

- Requires pre-scraped reference images
- Struggles with heavily wrapped items
- YOLO may over-segment products

---

## Author

**Deekshitha**
- Github: [@dkarvan8](https://github.com/dkarvan8)


**Hridayx**  
- GitHub: [@Hridayx](https://github.com/Hridayx)

---

## License

Academic project. See repository for details.
