# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import items, chat, admin, auth

from .settings import settings

app = FastAPI(title="DeliOps FastAPI Backend")

# Allow frontend (localhost:3000)
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


@app.get("/")
def root():
    return {"message": "DeliOps API is running 🚀"}
