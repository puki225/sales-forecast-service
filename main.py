"""
Thin FastAPI wrapper around pipeline.run(). Triggered on a schedule (n8n, or any HTTP-
capable scheduler) hitting POST /run - this service has no scheduler of its own. Runs
synchronously and returns a summary; a full catalog run is expected to take seconds to
low tens of seconds, not the kind of duration that needs a background job queue.
"""
import logging
import os

from fastapi import FastAPI, Header, HTTPException

import db
import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast_service")

app = FastAPI(title="Sales Forecast Service")

API_KEY = os.environ.get("API_KEY")  # optional - unset means /run has no auth, same
# opt-in convention as the Postgres-Access proxy this repo already runs.


def _check_auth(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/run")
def run(x_api_key: str = Header(default=None)):
    _check_auth(x_api_key)
    conn = db.get_connection()
    try:
        summary = pipeline.run(conn)
        logger.info("Forecast run complete: %s", summary)
        return {"ok": True, **summary}
    except Exception as e:
        logger.exception("Forecast run failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
