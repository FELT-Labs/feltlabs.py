import os
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.responses import FileResponse

# TODO: Separate the file provider

MODEL_DB = {}
app = FastAPI()


def remove_file(path: str) -> None:
    os.unlink(path)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload_model")
async def upload_model(_id: str, file: UploadFile):
    try:
        file_path = "model.joblib"
        MODEL_DB[_id] = file_path
        with open(file_path, "wb+") as f:
            f.write(file.file.read())

        return {"Status": "OK"}
    except Exception as e:
        return {"Status": f"FAIL: {e}"}


@app.get("/model")
def main(background_tasks: BackgroundTasks, _id: Optional[str] = None):
    if not _id:
        return {"Status": "You need to provider model id."}

    if _id in MODEL_DB:
        path = MODEL_DB.pop(_id)
        background_tasks.add_task(remove_file, path)
        return FileResponse(
            path=path, filename="model.joblib", media_type="application/octet-stream"
        )
    return {"Status": "File doesn't exist"}
