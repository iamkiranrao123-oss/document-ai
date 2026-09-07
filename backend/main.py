from fastapi import FastAPI

from backend.routes.upload import router as upload_router
from backend.routes.query import router as query_router
from backend.routes.documents import router as documents_router
from backend.routes.auth import router as auth_router

from backend.database.database import Base, engine


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Document AI",
    description=(
        "AI-powered document analysis and "
        "question answering system"
    ),
    version="1.0.0",
)


app.include_router(
    upload_router,
    tags=["Documents"]
)

app.include_router(
    query_router,
    tags=["Question Answering"]
)

app.include_router(
    documents_router,
    tags=["Documents"]
)

app.include_router(
    auth_router,
    tags=["Authentication"]
)


@app.get(
    "/",
    summary="API root",
    description="Returns a basic message confirming that the API is running."
)
def root():
    return {
        "message": "Document AI API is running"
    }


@app.get(
    "/health",
    summary="Health check",
    description="Checks whether the Document AI API is healthy."
)
def health_check():
    return {
        "status": "healthy"
    }