# Hair4Face

Real-time face-shape analysis with MediaPipe facial thirds + optional CNN model fusion (model weight 0.2), and personalized hairstyle/beard suggestions from CSV lookups.

## App entry
- Streamlit entry: `codebase/app.py`

## Project layout
- Models: `codebase/codebase/men_faceshape_model.h5`, `codebase/codebase/women_faceshape_model.h5`
- Metrics utils: `codebase/codebase/metrics_utils.py`
- CNN inference: `codebase/codebase/inference.py`
- Suggestions CSVs: `codebase/resultmen.csv`, `codebase/resultwomen.csv`

## Run locally
```bash
pip install -r codebase/requirements.txt
streamlit run codebase/app.py
```

If TensorFlow is not installed, the app still runs with MediaPipe-only metrics; model fusion will downweight to metrics-only. To enable model fusion, install TensorFlow (may be heavy on some machines).

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app:
   - Repo: your GitHub repo
   - Branch: main (or default)
   - App file path: `codebase/app.py`
3. Add `codebase/requirements.txt` as build deps (auto-detected). If build timeouts due to TensorFlow, remove TensorFlow from requirements; model becomes optional.

## CSV schema
- Men: `face_shape,interocular_category,nose_category,hairstyle_suggestions,beard_suggestions`
- Women: `face_shape,interocular_category,nose_category,hairstyle_suggestions`

## Notes
- Facial thirds use brow (glabella) for mid-third; no image flipping.
- Fusion weights: metrics=0.8, model=0.2.
