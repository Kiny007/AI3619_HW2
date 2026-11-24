# AI3619 Homework 2 Programming Assignment

## Objective

Estimate camera intrinsics and per-image extrinsics from checkerboard images using Zhang's calibration method. The implementation uses NumPy for homography estimation, parameter recovery, and nonlinear refinement; OpenCV is used for image IO and corner detection. Calibration uses central-region corners, and reprojection errors are evaluated over both the central region and the full board.

## Reproduction

Run the following commands from the project directory with Python 3:

```bash
python -m pip install -r requirements.txt
python calibration.py --redetect
```

The input images must be in `figs/`, named `camera_calib1.jpg`, `camera_calib2.jpg`, etc. Results are saved to `output/`:

- `calibration_results.json`: camera parameters, per-image poses, and error statistics.
- `reprojection_errors.csv` and `reprojection_errors.svg`: per-image errors and their plot.
- `overlays/`: detected and reprojected corners drawn on each image.
- `corners/`: cached corner detections.
