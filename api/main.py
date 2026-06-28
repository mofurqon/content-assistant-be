from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import content, health, ideas
from core.log import setup_logging

setup_logging()

app = FastAPI(title="Content Assistant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ideas.router, prefix="/v1")
app.include_router(content.router, prefix="/v1")
