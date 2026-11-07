# OpenCV Lab 1 – Advanced work with OpenCV

Laboratory work No. 1 for the course *Intelligent Information Systems* (Lviv Polytechnic National University, group ITM-11).
Twelve image and video analysis tasks with OpenCV 5.0, plus the individual task **Variant 1: cars, contours + tracking**.

Every task was tested on my own photos and videos, except the group photo for Task 8, which comes from a public news page.
The full report, with methods, tests and analysis, is submitted separately. This repository holds the code, the input images and the numeric results.

## Results at a glance

| Task | Topic | Main result |
|---|---|---|
| 1 | Automatic preprocessing | all six steps; Gaussian kernels 3/7/15 compared (sharpness, PSNR) |
| 2 | Adaptive thresholding | 5 binary images; adaptive Gaussian best under uneven light (left/right gap 5.5 vs 80–88 points) |
| 3 | Morphology | 15 results; opening 5×5 → closing 9×9 counts 9 of 9 objects |
| 4 | Contours | area, perimeter, box, centroid, aspect ratio, vertices for 9 objects; shape class 6 of 9 correct |
| 5 | Colour (HSV) | interactive trackbars; target marker isolated as one region |
| 6 | Histograms | CLAHE doubles local contrast; global equalisation widens the lighting gap |
| 7 | Hough lines and circles | circles 7 of 7; lines 3 of 5 (HoughLines) and 4 of 7 (HoughLinesP); parameter experiment |
| 8 | Haar face detection | 9-run scaleFactor × minNeighbors table; best 1.1 / 8 (9 of 9 faces, 1 false detection) |
| 9 | Object tracking | CSRT, KCF and MIL compared on one video |
| 10 | Motion detection | absdiff (2 variants) and MOG2 (2 histories); MOG2 with history 100 best |
| 11 | Complete CV system | video → preprocessing → motion → contours → filter → tracking → display, 26.7 FPS |
| 12 | Experimental study | Gaussian kernel k = 1…31 (8 runs); k = 3–11 give a perfect count |
| Var. 1 | Cars, contours + tracking | 15 of 15 cars counted at the line (precision and recall 1.0), 23 track IDs, 25.4 FPS |

## Files

| File | Task | Content |
|---|---|---|
| `Lib_Check.py`, `Opencv2.5_check.py`, `gui_test.py`, `check_trackers.py` | environment | library, OpenCV version, GUI and tracker checks (`gui_test.py` is identical to `Opencv2.5_check.py`) |
| `03_02_Labs.code-workspace`, `.vscode/settings.json` | all | VS Code configuration |
| `Lab_1_Jup.ipynb` | 1 | full Task 1 (steps 1–7, kernel experiments) |
| `task01_preprocess.py` | 1 | early version, steps 1–4 only |
| `Lab_1_Task02.ipynb` | 2 | five thresholding methods and experiments |
| `Lab_1_Task03.ipynb` | 3 | 15 morphology results, object counter, sensitivity |
| `Lab_1_Task04.ipynb` | 4 | contours, measurements, classification, ε experiment |
| `task05_hsv.py` | 5 | interactive HSV segmentation with trackbars |
| `Lab_1_Task06.ipynb` | 6 | histograms, equalisation, CLAHE, parameter experiment |
| `Lab_1_Task07.ipynb` | 7 | edge maps, Hough lines and circles, experiment |
| `Lab_1_Task08.ipynb` | 8 | Haar cascade face detection, 9-run experiment |
| `task09_tracking.py`, `Lab_1_Task09.ipynb` | 9 | tracking program (CSRT / KCF / MIL); comparison |
| `task10_motion.py`, `task10_annotate.py`, `Lab_1_Task10.ipynb` | 10 | motion pipeline; labelling helper (not used); video check and evaluation |
| `task11_system.py`, `Lab_1_Task11.ipynb` | 11 | full system; filter analysis and evaluation |
| `task12_annotate.py`, `Lab_1_Task12.ipynb` | 12 | ground-truth labelling; experiment |
| `var1_cars_system.py`, `var1_annotate.py`, `Lab_1_Variant1.ipynb` | Variant 1 | car counting and tracking system; ground-truth tool; video check, calibration, evaluation, experiment, figures |
| `data/` | inputs | input photos and the Haar cascade |
| `results/` | 1–12, Var. 1 | CSV / JSON results behind every number in the report (file prefix = task) |

The notebooks are saved **with their outputs**, so the figures and tables can be seen on GitHub without running anything.

## Setup

Tested with Python 3.14.3 on Windows.

```bash
python -m venv .venv01
.venv01\Scripts\activate          # Windows  (Linux/macOS: source .venv01/bin/activate)
pip install -r requirements.txt
python Lib_Check.py               # checks the libraries and the OpenCV version
python check_trackers.py          # checks that CSRT, KCF and MIL are available
```

`opencv-contrib-python` is required, not `opencv-python`: the trackers used in Task 9 are only in the contrib build.

## Running

Run everything **from the repository root**: the code reads from `data/` and writes to `results/`.

- Notebooks: open in VS Code or Jupyter, select the `.venv01` kernel, *Run All*.
- Scripts, for example: `python task05_hsv.py`, `python task09_tracking.py`, `python var1_cars_system.py`.

### Inputs not included

| File | Used by | Why missing / how to get it |
|---|---|---|
| `data/task09_track.mp4` | Task 9 | video, 50 MB |
| `data/task10_motion.mp4` | Tasks 10, 11 | video, 72 MB |
| `data/var1_cars.mp4` | Variant 1 | video, 146 MB (over GitHub's 100 MB file limit) |
| `data/task08_group.jpg` | Task 8 | not my photo; download it from the page in `data/Task08 source.txt` |

To run the video tasks, put your own videos in `data/` under these names. Variant 1 skips the first 28 s of its video (`START_S` in `var1_cars_system.py`); set it to 0 for another video, and redraw the road zone and counting line.

## Notes on OpenCV 5.0

- The Haar cascade is included in `data/cascades/` (from the OpenCV GitHub repository, 4.x branch) and loaded from there.
- All timings in `results/` were measured on one laptop (Intel Core i7-1255U) and vary between runs.

## Author

Aihiun Ateshohlu, group ITM-11, Lviv Polytechnic National University, 2026.
