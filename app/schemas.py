from pydantic import BaseModel


class DriftResponse(BaseModel):
    total_uploaded_rows: int
    total_stored_rows: int
    drift_checked: bool
    drift_detected: bool
    drift_score: float
    message: str
