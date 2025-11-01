
# DeliOps FastAPI RAG Backend
## Setup
pip install -r requirements.txt
cp .env.example .env  # fill values
# (optional) ingest docs for RAG
python -m scripts.ingest --path ./docs
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
