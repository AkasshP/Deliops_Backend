# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import items, chat, admin, auth
from .settings import settings
from .services.rag import ensure_index_ready  # NEW
from .routes.feedback import router as feedback_router  
from .routes import orders as orders_router

app = FastAPI(title="DeliOps FastAPI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(feedback_router)  
app.include_router(orders_router.router)

@app.get("/")
def root():
    return {"message": "DeliOps API is running 🚀"}

@app.on_event("startup")
def _startup_refresh_index():
    try:
        ensure_index_ready()
        print("[startup] vector index ready.")
    except Exception as e:
        # Don’t crash; chat will try again lazily
        print(f"[startup] index prep failed (will retry lazily): {e}")
