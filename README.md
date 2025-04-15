# CUDA-Accelerated Canny Edge Detection

This project implements the Canny edge detection algorithm using CUDA for GPU acceleration and provides a comparison with a CPU implementation.

---

## Project Overview

The primary goal is to perform edge detection on PGM images using the Canny algorithm. It includes the following steps:

- Gaussian smoothing
- Gradient calculation
- Non-maximal suppression
- Hysteresis thresholding

GPU-accelerated (CUDA) and CPU versions are included for performance comparison.

---

## File Breakdown

- **`canny_edge.cu`**  
  Main program logic:
  - Parses command-line arguments (input image, sigma, thresholds)
  - Implements CPU (`gaussian_smooth_cpu`, `derrivative_x_y_cpu`) and GPU (`gaussian_smooth_gpu`, `derrivative_x_y_gpu`) steps using CUDA kernels
  - Calls functions for magnitude, non-max suppression, and hysteresis
  - Handles image I/O via `pgm_io.cu`

- **`hysteresis.cu`**  
  Implements:
  - `non_max_supp`
  - `apply_hysteresis`
  - `follow_edges`

- **`pgm_io.cu`**  
  - PGM I/O utilities: `read_pgm_image`, `write_pgm_image`
  - Some PPM utilities also included but unused in main logic

- **`Makefile`**  
  - Builds the project with `nvcc`
  - Contains targets for running and cleaning

- **`pics/`**  
  - Sample PGM images for testing

---

##  Build Instructions

Make sure you have CUDA installed.

```bash
make
```

Run: Execute the compiled program
```bash
make run
```

Alternatively : 
```bash
./canny <image_path.pgm> <sigma> <tlow> <thigh>
```
