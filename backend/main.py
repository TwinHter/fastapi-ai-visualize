from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from api.metrics import router as metrics_router
from api.predict import router as predict_router
from api.image import router as image_router
import uvicorn

app = FastAPI()
origins = [
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(metrics_router)
app.include_router(predict_router)
app.include_router(image_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)