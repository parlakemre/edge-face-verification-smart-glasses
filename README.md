# Temporal Fusion and Quantization for Robust Edge-Based Face Recognition on Smart Glasses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Journal](https://img.shields.io/badge/Journal-Royal%20Society%20Open%20Science-orange.svg)](https://royalsocietypublishing.org/journal/rsos)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)

Official research code and reproducible benchmark pipelines for the manuscript:  
**"Temporal Fusion and Quantization for Robust Edge-Based Face Recognition on Smart Glasses"**.

---

## 📌 Overview

Deploying deep learning-based face verification models on wearable edge platforms (such as smart glasses) presents severe engineering bottlenecks due to strict thermal dissipation budgets, battery longevity constraints, and limited memory bandwidth. While post-training model quantization (INT8) slashes weight footprints by **74.4%** ($36.90\text{ MB} \to 9.44\text{ MB}$), single-frame INT8 inference induces representation drift that elevates the False Non-Match Rate (FNMR) from $32.96\%$ to $44.87\%$.

This repository provides an end-to-end, untethered edge verification architecture featuring:
1. **Integer Lookup Table (LUT) Gamma Correction:** An $\mathcal{O}(1)$ runtime lookup table ($\gamma = 1.8$) for zero-FLOP photometric enhancement.
2. **Offline Synthetic Occlusion Gallery:** Generating 7 canonical occlusion profiles (caps, medical masks, eyewear, headscarves) at enrollment with automated HSV modality detection.
3. **Top-$K$ Spatial Score Fusion:** A multi-candidate matching scheme ($K=3$) that eliminates imposter vulnerabilities from localized facial similarities.
4. **3-Phase Temporal State Machine:** Integrating Lucas-Kanade optical flow tracking with periodic burst consensus, sustaining **$>30\text{ FPS}$** throughput with **$<15\%$ CPU core utilization** while completely neutralizing quantization drift.

---

## 🔬 Key Empirical Results

### 1. LFW Benchmark (6,000 Pairs: 3,000 Genuine, 3,000 Imposter)
- **Equal Error Rate (EER):** $4.25\%$
- **Area Under Curve (AUC):** $0.993$ ($41.7\%$ relative error reduction over baseline single-shot $0.988$)
- **Zero Imposter Leakage:** $\text{FAR} = 0.0\%$ at operational threshold $\tau = 0.400$.

### 2. RFW Demographic Equity (24,000 Pairs across 4 Cohorts)
- Evaluated across African, Asian, Caucasian, and Indian demographic groups.
- Stable inter-cohort variance across precision regimes ($6.26\%$ under INT8 vs. $6.37\%$ under FP32), confirming that quantization and temporal smoothing preserve demographic fairness.

| Model / Architecture | Scope | Params | Model Size | Complexity | LFW Acc. | Edge Throughput | Occlusion Resilience |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MobileFaceNet | Recog. | 1.00M | 4.0 MB | 440 MFLOPs | 99.50% | 8--12 FPS | None (Single-Shot) |
| ShuffleFaceNet 1.0x | Recog. | 1.40M | 5.5 MB | 290 MFLOPs | 99.67% | 10--14 FPS | None (Single-Shot) |
| GhostFaceNet++ | Recog. | 1.70M | 7.5 MB | 52 MFLOPs | 99.63% | 12--16 FPS | None (Single-Shot) |
| **Proposed (FP32)** | **Full Edge** | **4.21M** | **15.9 MB** | **0.63 GFLOPs** | **99.40%** | **>30 FPS (<15% CPU)** | **Top-3 Spatial Fusion** |
| **Proposed (INT8)** | **Full Edge** | **4.21M** | **3.98 MB** | **0.16 GOPs** | **96.20%** | **>30 FPS (<12% CPU)** | **Spatial + Temporal FSM** |

---

## 📁 Repository Structure

```text
edge-face-verification-smart-glasses/
├── assets/
│   ├── Fig_cnn_architecture.pdf   # Architectural diagram of end-to-end pipeline
│   ├── far_frr_curve_top3.png     # LFW threshold operational curves
│   ├── occlusion_samples.png      # 7 canonical synthetic occlusion profiles
│   └── glasses.png                # Transparent eyewear template asset
├── src/
│   ├── __init__.py
│   ├── face_pipeline.py           # YuNet detection & SFace feature embedding wrapper
│   ├── passport_degradation.py    # Physics-based passport document degradation model
│   ├── synthetic_gallery.py       # LUT gamma, modality detection, 7 occlusion profiles
│   ├── temporal_fsm.py            # 3-phase temporal state machine & optical flow tracking
│   ├── quantize.py                # Post-training dynamic INT8 ONNX quantization
│   ├── evaluate_lfw.py            # Complete 6,000-pair LFW evaluation & FAR/FRR curves
│   └── evaluate_rfw.py            # 24,000-pair RFW demographic fairness benchmarking
├── demo.py                        # Standalone verification demo
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/parlakemre/edge-face-verification-smart-glasses.git
cd edge-face-verification-smart-glasses

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Standalone Demo
Test the synthetic gallery generation and passport degradation without external dataset downloads:
```bash
python demo.py
```

### 3. Reproducing LFW Benchmark
Download the [LFW Deep Funneled](http://vis-www.cs.umass.edu/lfw/) dataset and place pairs into `lfw_data/`:
```bash
python src/evaluate_lfw.py --lfw /path/to/lfw_data --output assets/far_frr_curve_top3.png
```

### 4. Reproducing RFW Demographic Evaluation
Download [Racial Faces in the Wild (RFW)](http://whdeng.cn/RFW/index.html):
```bash
python src/evaluate_rfw.py --rfw /path/to/rfw_data --threshold 0.400
```

### 5. Quantizing ONNX Models to INT8
```bash
python src/quantize.py --input models/face_recognition_sface_2021dec.onnx --output models/sface_int8.onnx
```

---

## 📖 Citation

If you find this research or code useful in your work, please cite:

```bibtex
@article{parlak2026temporal,
  author  = {Parlak, Emre and Kahraman, Nihan},
  title   = {Temporal Fusion and Quantization for Robust Edge-Based Face Recognition on Smart Glasses},
  journal = {Preprint / Under Review},
  year    = {2026},
  note    = {Submitted for peer review}
}
```

---

## 📄 License & Acknowledgments
- Code released under the [MIT License](LICENSE).
- The authors express sincere gratitude to Assoc. Prof. Dr. Nurullah Çalık (Istanbul Medeniyet University) for invaluable academic guidance.
