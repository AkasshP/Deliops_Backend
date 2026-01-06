# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .services.rag import ensure_index_ready
from .routes import items, chat, admin, auth, orders
from .routes.feedback import router as feedback_router

app = FastAPI(title="DeliOps FastAPI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(items.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(feedback_router)
app.include_router(orders.router)


@app.get("/")
def root():
    return {"message": "DeliOps API is running"}


@app.on_event("startup")
def _startup_refresh_index():
    """Initialize the vector index on application startup."""
    ensure_index_ready(startup=True)
    print("[startup] Vector index ready.")
