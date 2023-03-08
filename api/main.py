from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Running..."

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)