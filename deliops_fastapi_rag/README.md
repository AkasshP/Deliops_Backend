
# DeliOps FastAPI RAG Backend

Backend API for Huskies Deli — handles inventory, chat (RAG-powered), ordering with Stripe payments, and an intelligent agent layer.

## Architecture

```
Client (Next.js frontend)
    │
    ▼
FastAPI ─── /items      Item CRUD (Postgres)
        ├── /chat       RAG chat: NLU → rules/name lookup/pgvector search → LLM polish
        ├── /agent/chat Agent chat: regex fast path OR RAG + OpenRouter LLM
        ├── /orders     Order creation + Stripe PaymentIntent
        ├── /admin      Admin management + pgvector reindex
        └── /auth       Admin authentication

Data store:
  Postgres  ─── all business data (items, orders, feedback)
              + pgvector embeddings (vector search)

External services:
  OpenAI    ─── embeddings (text-embedding-3-small), NLU intent classification, LLM polish
  OpenRouter ── agent LLM (Claude 3.5 Sonnet)
  Stripe    ─── payment processing
```

See [docs/agent.md](docs/agent.md) for detailed architecture diagrams.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill values

# Run the SQL migrations against your Postgres database
psql $DATABASE_URL -f migrations/001_create_item_embeddings.sql
psql $DATABASE_URL -f migrations/002_create_items_orders_feedback.sql

# Start the server (builds pgvector index on startup from Postgres items)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Postgres connection string (with pgvector extension) |
| `OPENAI_API_KEY` | Yes | — | Embeddings, NLU classification, LLM polish |
| `OPENROUTER_API_KEY` | For agent | — | Agent LLM path (OpenRouter) |
| `OPENROUTER_MODEL` | No | `anthropic/claude-3.5-sonnet` | Model for agent LLM |
| `STRIPE_SECRET_KEY` | For orders | — | Stripe payment processing |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated or JSON array |
| `OPENAI_MODEL` | No | `gpt-3.5-turbo` | Model for NLU + polish |
| `RAG_TOP_K` | No | `4` | Number of vector search results |
| `RAG_SIMILARITY_THRESHOLD` | No | `0.75` | Minimum cosine similarity for results |

## API Endpoints

### Chat
- `POST /chat` — Guest chat with RAG, NLU intent detection, and order placement

### Agent
- `POST /agent/chat` — Two-path agent (fast regex + normal RAG/LLM)
- `POST /agent/tools/retrieve` — Direct pgvector search
- `GET /agent/tools/inventory?query=...` — Direct inventory lookup (Postgres)

### Items
- `GET /items` — List public active items
- `GET /items/{id}` — Get single item
- `POST /items` — Create item (admin)
- `PATCH /items/{id}` — Update item (admin)
- `DELETE /items/{id}` — Delete item (admin)

### Orders
- `POST /orders/intent` — Create order + Stripe PaymentIntent
- `POST /orders/{id}/finalize` — Confirm payment + decrement stock
- `GET /orders` — List orders

### Admin
- `POST /admin/reindex` — Rebuild pgvector index from Postgres items
- `GET /admin/index-status` — Check pgvector index row count

## Project Structure

```
app/
├── main.py                  # FastAPI app, lifespan, router registration
├── settings.py              # Pydantic settings (env vars)
├── db.py                    # Async Postgres connection pool (asyncpg)
├── routes/
│   ├── chat.py              # /chat endpoint + session store
│   ├── items.py             # /items CRUD endpoints
│   ├── orders.py            # /orders endpoints
│   ├── admin.py             # /admin endpoints (reindex, index-status)
│   ├── auth.py              # /auth endpoints
│   └── feedback.py          # /feedback endpoints
├── services/
│   ├── items.py             # Item CRUD against Postgres
│   ├── orders.py            # Order logic + Stripe + Postgres transactions
│   ├── rag.py               # RAG pipeline: index build, search, answer
│   ├── nlu.py               # NLU intent classification (OpenAI)
│   ├── embeddings.py        # OpenAI embedding generation
│   └── feedback.py          # Feedback service (Postgres)
├── agent/
│   ├── agent_router.py      # /agent routes
│   ├── agent_runtime.py     # Two-path agent logic (fast/normal)
│   ├── schemas.py           # Agent request/response models
│   └── tools/
│       ├── lookup_inventory.py   # Name-based item lookup (Postgres)
│       ├── retrieve_knowledge.py # Vector similarity search (pgvector)
│       └── stubs.py              # Stub tools (add_to_cart, checkout)
├── vectorstore/
│   └── pgvector_store.py    # Postgres pgvector: upsert, query, delete
migrations/
├── 001_create_item_embeddings.sql  # pgvector table + IVFFlat index
└── 002_create_items_orders_feedback.sql  # items, orders, feedback tables
docs/
└── agent.md                 # Architecture diagrams
```
