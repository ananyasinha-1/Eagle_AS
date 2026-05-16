from fastapi import FastAPI
from apps.backend.tasks import analyze_sequence

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze/{track_id}")
async def trigger_analysis(track_id: int):
    """
    Trigger asynchronous VLM/LLM analysis for a given track.
    Returns immediately with a queued status.
    """
    analyze_sequence.delay(track_id)
    return {"status": "queued", "track_id": track_id}