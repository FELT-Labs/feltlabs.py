"""Fast API server for exchanging models with ocean C2D provider."""
import os
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.responses import FileResponse

MODEL_DB = {}
app = FastAPI()


def remove_file(path: str) -> None:
    """Function to remove file from system."""
    os.unlink(path)


@app.get("/")
def read_root():
    """Index endpoint."""
    return {"FELToken": "model provider"}


@app.post("/upload_model")
async def upload_model(_id: str, file: UploadFile):
    """Endpoint for uploading model with given ID."""
    global MODEL_DB
    try:
        file_path = "model.joblib"
        MODEL_DB[_id] = file_path
        with open(file_path, "wb+") as f:
            f.write(file.file.read())

        return {"Status": "OK"}
    except Exception as e:
        print("FAIL", e)
        return {"Status": f"FAIL: {e}"}


@app.get("/model")
def main(background_tasks: BackgroundTasks, _id: Optional[str] = None):
    """Endpoint for getting model by given ID."""
    if not _id:
        return {"Status": "You need to provider model id."}

    if _id in MODEL_DB:
        path = MODEL_DB[_id]
        # path = MODEL_DB.pop(_id)
        # background_tasks.add_task(remove_file, path)
        return FileResponse(path=path, media_type="application/octet-stream")
    return {"Status": "File doesn't exist"}
