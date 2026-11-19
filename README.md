# Spindle Silver-Lining Detector

YOLOv8n detector that flags silver linings on gold-thread spindles. Streamlit front end.

## Files
- `app.py` — Streamlit app (image upload, confidence slider, DEFECT/PASS verdict)
- `best.pt` — trained YOLOv8n weights (`silver_lining` class)
- `requirements.txt`, `packages.txt` — deps (`libgl1` and `libglib2.0-0` are needed on Streamlit Cloud)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo (including `best.pt`) to GitHub.
2. Go to share.streamlit.io -> New app -> select repo, branch, `app.py`.
3. Advanced settings -> pick Python 3.11 or 3.12.

## Notes
- Model: YOLOv8n (`best.pt`), 1 class: `silver_lining`.
