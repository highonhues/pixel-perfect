# Microscopy Image Preprocessor

A web-based tool for enhancing biological microscopy images with real-time processing controls.

#### Quick Start

```bash
uv sync
uv run python app.py
```

Open browser to: http://localhost:7400

Demo Image
<img width="1897" height="968" alt="Screenshot 2025-10-18 131801" src="https://github.com/user-attachments/assets/94008430-cfb2-4ee2-9e50-d68997450902" />


## Processing Functions

#### CLAHE Enhancement
Improves local contrast in both dim and bright regions. Use 40-60 for subtle enhancement, 70-80 for dramatic results.

#### Illumination Correction
Corrects uneven lighting and background gradients across the image. Works well for fluorescence microscopy.

#### Background Subtraction
Removes background using morphological operations. Use for images with significant background noise.

#### Gaussian Smoothing
Reduces random noise while preserving edges. Keep intensity low (20-40) to avoid losing detail.

#### Median Filter
Removes salt-and-pepper noise effectively. Best for high-noise images.

#### Bilateral Filter
Edge-preserving smoothing. Reduces noise while maintaining sharp boundaries.

### Features

- Upload TIFF, PNG, JPG images (8-bit and 16-bit supported)
- Adjustable intensity sliders (0-100) for each function
- Real-time quality metrics (Contrast, SNR, Sharpness, Dynamic Range)
- Persistent settings - all controls remain set after processing
- Reset All button to clear all processing
- Download processed images

### Processing Pipeline

All processing starts from the original uploaded image - never builds on previous results. Check/uncheck any combination of functions and click "Process Image" to see results.


