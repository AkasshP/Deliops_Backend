# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .services.rag import ensure_index_ready
from .routes import items, chat, admin, auth, orders, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    ensure_index_ready(startup=True)
    print("[startup] Vector index ready.")
    yield


app = FastAPI(title="DeliOps FastAPI Backend", lifespan=lifespan)

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
app.include_router(feedback.router)
app.include_router(orders.router)


@app.get("/")
def root():
    return {"message": "DeliOps API is running"}
