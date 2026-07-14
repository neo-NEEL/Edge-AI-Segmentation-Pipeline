
# Real-Time Python Edge AI Lane Segmentation Pipeline

A minimalist, high-performance Python implementation of an optimized computer vision pipeline. This project takes an optimized **UNet architecture** and executes it using the highly efficient **C++ NCNN edge runtime framework** to perform real-time lane tracking natively on the host CPU.

## 🛠️ The Implementation Journey & Engineering Fixes

This repository documents the step-by-step software engineering process required to deploy raw edge models successfully on a local Windows machine, moving past environment conflicts and file system bugs.

### 1. Environment Optimization & Bootstrapping
- **Windows Extension Fixes:** Resolved file extension anomalies (`.py.txt` and `.param.txt` double extensions) hidden by default Windows File Explorer settings by forcing clean rewrites natively via the Command Prompt.
- **Pip Engine Repair:** Successfully recovered a broken global environment by bootstrapping Python's native installation tools using `python -m ensurepip --default-pip` to force-inject the missing package management framework.

### 2. Core Vision & Array Setup
- Managed dependencies by binding core processing tools directly to the active executable workspace:
  ```bash
  python -m pip install opencv-python numpy ncnn

```

* **Automated Directory Scanner:** Replaced rigid, hardcoded dataset paths with a resilient Python filesystem crawler (`os.walk`) that automatically sweeps the directory tree to self-detect and load valid local image frames.

### 3. Custom Dynamic Filtering & Optimization (The Green Haze Fix)

* **Percentile Feature Isolation:** Addressed a critical issue where the raw uncalibrated model activation layer registered background pixels (sky, buildings) as lane elements, tinting the whole frame green.
* Implemented a dynamic matrix cutoff rule utilizing NumPy percentiles to strip away weak data artifacts and isolate true positive lane signatures:
```python
cutoff_val = np.percentile(output_mask, 90)
binary_mask = (output_mask > cutoff_val).astype(np.uint8) * 255

```


* **Target-Specific Overlay Blending:** Configured selective alpha blending using OpenCV matrix masking to map the bright green visualization layer *strictly* onto high-confidence lane coordinates, leaving the background terrain completely untouched.

---

## 🚀 Execution Data Flow

```
[Target Image File] 
       │
       ▼
[Auto Scan & Load] ──► [Scale & Normalize Matrix to 256x256]
                                      │
                                      ▼
                           [NCNN Forward Inference Layer]
                                      │
                                      ▼
                        [Extract Raw Activation Map]
                                      │
                                      ▼
                        [Apply 90th Percentile Cutoff Filter]
                                      │
                                      ▼
                        [Selective OpenCV Alpha Mask Blending]
                                      │
                                      ▼
                       [Live Computer Vision Dashboard]

```

## 📦 Running the System Locally

To test this minimalist Python setup on any machine, ensure `run_edge_inference.py`, `unet_jit.ncnn.param`, and `unet_jit.ncnn.bin` are sitting in the same folder, then run:

```bash
python run_edge_inference.py

```

```

```