# Migration / Backup Guide

What I did:

- Created Python packages and reorganized copies of core modules into the
  following new packages: `detection/`, `pipeline/`, `models/`, `config/`,
  and `weights/` (logical placeholder).
- Did NOT delete or modify any original top-level files — originals remain
  in place as a safe backup. This lets you test the new package layout before
  removing or replacing the old files.

Mapping (new location -> original file kept in place):

- `detection/ball_detector.py`  <- `ball_detector.py`
- `detection/ball_tracker.py`   <- `ball_tracker.py`
- `detection/person_detector.py`<- `person_detector.py`
- `detection/pose_detector.py`  <- `pose_estimator.py`
- `models/yolo_detect.py`       <- `yolo_detect.py`
- `pipeline/main_pipeline.py`  <- `detection_pipeline.py`
- `pipeline/preprocessing.py`   <- `frame_extractor.py`
- `pipeline/postprocessing.py`  <- `video_utils.py`
- `config/paths.py`             <- NEW central config (points to existing weights)

Important notes and next steps:

- I only *copied* modules into the new packages and updated their internal
  imports to reference the new package paths (so `pipeline/main_pipeline.py`
  imports `detection.*` modules). The original files are untouched.
- Large binary weights were NOT moved. Update `config/paths.py` or set the
  environment variables (`YOLO_BALL_WEIGHTS`, etc.) if you want the new code
  to reference moved weights.
- After you're happy with the restructure and tests pass, I can:
  - Remove the original top-level files (one-by-one or in a batch),
  - Move heavy weight files into `weights/` and update `config/paths.py`,
  - Run a search-and-replace to ensure no stray imports still reference old
    top-level modules.

How to run the pipeline using the new layout (quick test):

1. Run a small script that imports the new pipeline module, e.g.:

```powershell
python -c "from pipeline.main_pipeline import process_frames_pipeline; print('Imported OK')"
```

2. If the import above works, run your usual `main.py` — note `main.py` still
   imports the original modules; if you want I can update `main.py` to use the
   new package imports instead.
