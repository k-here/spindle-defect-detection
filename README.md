# Spindle Silver-Lining Detector

YOLO11n detector that flags silver linings on gold-thread spindles. Streamlit front end.

## Files
- `app.py` — Streamlit app (upload / camera, confidence slider, DEFECT/CLEAN verdict)
- `spindle_silver_best.pt` — trained weights (copy from Drive: `Colab spindle_project /spindle_silver_best.pt`)
- `requirements.txt`, `packages.txt` — deps (`libgl1` is needed on Streamlit Cloud)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo (including the `.pt` file) to GitHub.
2. share.streamlit.io -> New app -> select repo, branch, `app.py`.
3. Advanced settings -> pick Python 3.11 or 3.12.

## Notes
- Input preprocessing strips the capture-tool UI banner (green bars + session text) if present; raw camera frames pass through unchanged.
- Model trained on cropped frames, 1 class: `silver_defect`.
