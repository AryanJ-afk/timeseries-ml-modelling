from io import StringIO, BytesIO
from uuid import uuid4
import os

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model_loader import load_model
from app.monitoring_service import (
    FEATURE_COLUMNS,
    append_to_history,
    run_feature_drift_detection,
    validate_feature_columns,
)

PREDICTIONS_DIR = "data/processed/predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

app = FastAPI(title="Time-Series Forecast API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/downloads", StaticFiles(directory="data/processed/predictions"), name="downloads")
templates = Jinja2Templates(directory="app/templates")

model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": None,
            "summary": None,
            "preview_table": None,
            "drift_details": None,
            "download_url": None,
        },
    )


@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a CSV file.")

        contents = await file.read()
        decoded = contents.decode("utf-8")
        df = pd.read_csv(StringIO(decoded))

        validate_feature_columns(df)
        df = df[FEATURE_COLUMNS].copy()

        predictions = model.predict(df).astype(float)

        result_df = df.copy()
        result_df["prediction"] = [float(x) for x in predictions]

        total_stored_rows = append_to_history(df)
        drift_info = run_feature_drift_detection()

        file_id = uuid4().hex
        output_filename = f"predictions_{file_id}.csv"
        output_path = os.path.join(PREDICTIONS_DIR, output_filename)
        result_df.to_csv(output_path, index=False)

        preview_html = result_df.head(20).to_html(
            classes="result-table",
            index=False,
            border=0,
            float_format=lambda x: f"{float(x):.4f}"
        )

        summary = {
            "total_uploaded_rows": int(len(df)),
            "total_stored_rows": int(total_stored_rows),
            "drift_checked": bool(drift_info["drift_checked"]),
            "drift_detected": bool(drift_info["drift_detected"]),
            "drift_score": float(drift_info["drift_score"]),
            "message": str(drift_info["message"]),
        }

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": None,
                "summary": summary,
                "preview_table": preview_html,
                "drift_details": drift_info.get("details", {}),
                "download_url": f"/downloads/{output_filename}",
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e),
                "summary": None,
                "preview_table": None,
                "drift_details": None,
                "download_url": None,
            },
        )
