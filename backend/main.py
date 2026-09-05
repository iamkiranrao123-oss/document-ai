from fastapi import FastAPI

from backend.routes.upload import router as upload_router
from backend.routes.query import router as query_router
from backend.routes.documents import router as documents_router


app = FastAPI(
    title="Document AI",
    description="AI-powered document analysis and question answering system"
)


app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "message": "Document AI API is running"
    }