from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "model": settings.MODEL_NAME,
    }