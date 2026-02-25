# DeliOps Backend Architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                         │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  /items  │ │  /chat   │ │ /orders  │ │  /admin  │ │  /agent  │ │
│  │  routes  │ │  routes  │ │  routes  │ │  routes  │ │  routes  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │            │            │            │            │         │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐     │       ┌────┴─────┐  │
│  │  items   │ │   rag    │ │  orders  │     │       │  agent   │  │
│  │ service  │ │ service  │ │ service  │     │       │ runtime  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘     │       └────┬─────┘  │
│       │            │            │            │            │         │
│       │       ┌────┴─────┐     │            │       ┌────┴─────┐  │
│       │       │   nlu    │     │            │       │  tools   │  │
│       │       │ service  │     │            │       │(inv/rag) │  │
│       │       └────┬─────┘     │            │       └────┬─────┘  │
│       │            │           │            │            │         │
│  ┌────┴────────────┴───────────┴────────────┴────────────┴─────┐  │
│  │              pgvector_store.py (async via asyncpg)           │  │
│  └─────────────────────────────┬───────────────────────────────┘  │
└────────────────────────────────┼───────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                                                ▼
┌──────────────────────────────────┐   ┌──────────────────────┐
│         PostgreSQL               │   │   External APIs      │
│         + pgvector               │   │                      │
│                                  │   │  OpenAI (embed, NLU) │
│  items, orders, feedback tables  │   │  OpenRouter (agent)  │
│  item_embeddings + IVFFlat index │   │  Stripe (payments)   │
│                                  │   │                      │
└──────────────────────────────────┘   └──────────────────────┘
     source of truth + vectors            LLM + payments
```

### Data Storage

| Data          | Store              | Details                                        |
|---------------|--------------------|------------------------------------------------|
| Items         | Postgres           | `items` table, flat columns                     |
| Orders        | Postgres           | `orders` table, lines as JSONB, `SELECT FOR UPDATE` for stock |
| Feedback      | Postgres           | `feedback` table                                |
| Embeddings    | Postgres pgvector  | `item_embeddings` table, vector(1536)           |
| Sessions      | In-memory dict     | `SessionStore` with 1hr TTL, max 1000           |

---

## Chat Flow: `/chat` endpoint (RAG path)

The original chat endpoint used by the guest UI. Handles Q&A, small talk, and order placement.

```
User message
     │
     ▼
┌─────────────┐
│  NLU parse  │◄── OpenAI GPT-3.5 classifies intent
│ (parse_query│    (greeting, hours, deals, order, item, etc.)
│  in nlu.py) │
└──────┬──────┘
       │
       ▼
   ┌───────────┐  yes  ┌─────────────────┐
   │ is_order? │──────►│ GPT extracts    │
   └─────┬─────┘       │ {name, qty}     │
         │ no          │ from free text  │
         ▼             └────────┬────────┘
   ┌───────────┐               │
   │ rules     │  yes          ▼
   │ answer?   │──────► Return  ┌──────────────────┐
   │(hours,    │  text  │ Map names → itemIds     │
   │ greeting, │        │ via in-memory name_map  │
   │ deals...) │        └────────┬────────────────┘
   └─────┬─────┘                 │
         │ no                    ▼
         ▼              ┌────────────────────┐
   ┌────────────┐       │ create_order_with  │
   │ exact name │  yes  │ _intent (Stripe +  │
   │ lookup in  │──────►│  Postgres)         │
   │ _name_map  │ reply └────────┬───────────┘
   └─────┬──────┘                │
         │ no                    ▼
         ▼              Return mode="payment"
   ┌────────────┐       + clientSecret
   │ embed query│
   │ → pgvector │
   │ cosine     │
   │ search     │
   └─────┬──────┘
         │
         ▼
   ┌────────────┐
   │ format     │
   │ item reply │
   │ (+ optional│
   │ LLM polish)│
   └─────┬──────┘
         │
         ▼
   Return mode="chat"
   + message text
```

**Key files:** `routes/chat.py` → `services/nlu.py` → `services/rag.py` → `vectorstore/pgvector_store.py`

---

## Agent Flow: `/agent/chat` endpoint

The agent layer with a fast regex path (no LLM needed) and a normal RAG + OpenRouter LLM path.

```
User message + optional history
     │
     ▼
┌────────────────────┐
│ regex fast-pattern │   Patterns: "do you have", "how much",
│ check              │   "in stock", "price of", etc.
└─────────┬──────────┘
          │
     ┌────┴────┐
     │ match?  │
     └────┬────┘
     yes  │  no
     ▼    │
┌──────────────┐  │
│ lookup_      │  │
│ inventory()  │  │   Exact/contains name match
│ (name match  │  │   against _name_map + Postgres
│  + Postgres  │  │   for fresh qty/price
│  fresh data) │  │
└──────┬───────┘  │
       │          │
  ┌────┴────┐     │
  │ found?  │     │
  └────┬────┘     │
  yes  │  no      │
  ▼    └────┬─────┘
Format      │
inventory   ▼
reply   ┌──────────────┐
  │     │ retrieve_    │   Embed query via OpenAI
  │     │ knowledge()  │   → pgvector cosine search
  │     │ (pgvector    │   → filter by threshold
  │     │  search)     │
  │     └──────┬───────┘
  │            │
  │            ▼
  │     ┌──────────────┐
  │     │ Build context │   Format top-k results as text
  │     │ from results  │   lines for LLM prompt
  │     └──────┬───────┘
  │            │
  │            ▼
  │     ┌──────────────┐
  │     │ OpenRouter    │   System prompt (deli assistant)
  │     │ LLM call      │   + context + user question
  │     │ (Claude 3.5)  │   via httpx async POST
  │     └──────┬───────┘
  │            │
  ▼            ▼
┌────────────────────────┐
│ Return:                │
│  reply: str            │
│  used_tools: [...]     │
│  path: "fast"|"normal" │
└────────────────────────┘
```

**Key files:** `agent/agent_router.py` → `agent/agent_runtime.py` → `agent/tools/` → `vectorstore/pgvector_store.py`

---

## Full Request Lifecycle (startup → query)

```
                          APPLICATION STARTUP
                          ==================
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │  ensure_index_ready() │
                    │  (async, in lifespan) │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ build_index():        │
                    │ 1. list_items()       │
                    │    (all from Postgres │
                    │ 2. embed_texts()      │
                    │    (OpenAI API)       │
                    │ 3. pgvector_store     │
                    │    .upsert_items()    │
                    │ 4. delete_missing()   │
                    │    (cleanup stale)    │
                    │ 5. _rebuild_name_map()│
                    └───────────┬───────────┘
                                │
                                ▼
                     App ready, name_map in memory,
                     embeddings in Postgres pgvector

                          QUERY TIME
                          ==========
                                │
                    ┌───────────┴───────────┐
                    │ /chat or /agent/chat  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                   ▼
     ┌─────────────────┐                 ┌─────────────────┐
     │ /chat:          │                 │ /agent/chat:    │
     │ NLU → rules OR  │                 │ regex fast path │
     │ name lookup OR  │                 │ OR pgvector +   │
     │ pgvector search │                 │ OpenRouter LLM  │
     └────────┬────────┘                 └────────┬────────┘
              │                                   │
              │         ┌──────────────┐          │
              └────────►│  PostgreSQL  │◄─────────┘
                        │ (fresh data) │
                        └──────────────┘
```

---

## pgvector Database Schema

```sql
-- Table: item_embeddings
item_id      TEXT PRIMARY KEY      -- matches items.id
item_name    TEXT NOT NULL
category     TEXT
description  TEXT                  -- "Turkey | Type: prepared | Service: cold | ..."
price        NUMERIC(10, 2)
in_stock     BOOLEAN DEFAULT TRUE
updated_at   TIMESTAMPTZ
embedding    vector(1536)          -- OpenAI text-embedding-3-small

-- Index: IVFFlat for cosine distance (lists=10, suitable for <1000 items)
CREATE INDEX idx_item_embeddings_cosine
    ON item_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
```

Query pattern:
```sql
SELECT *, 1 - (embedding <=> $1::vector) AS similarity
FROM item_embeddings
WHERE in_stock = TRUE
ORDER BY embedding <=> $1::vector
LIMIT $2
```

---

## Agent Layer

The agent layer adds an intelligent chat interface with a **fast path** (regex + direct inventory lookup, no LLM) for simple queries and a **normal path** (pgvector search + OpenRouter LLM) for complex questions.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(none)* | Required — Postgres connection string |
| `OPENROUTER_API_KEY` | *(none)* | Required for the normal (LLM) path |
| `OPENROUTER_MODEL` | `anthropic/claude-3.5-sonnet` | Model ID on OpenRouter |
| `RAG_SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity for results |
| `RAG_TOP_K` | `4` | Max results returned |

## Endpoints

### `POST /agent/chat`

Main agent endpoint. Automatically selects fast or normal path.

**Request:**
```json
{
  "message": "do you have turkey?",
  "history": [{"role": "user", "content": "hi"}]
}
```

**Response:**
```json
{
  "reply": "Yes, we have Turkey! We have 5 in stock. It costs $8.99 plus tax.",
  "used_tools": ["lookup_inventory"],
  "path": "fast"
}
```

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"do you have turkey?"}'
```

### `POST /agent/tools/retrieve`

Direct access to pgvector search.

**Request:**
```json
{ "query": "cheese", "top_k": 4 }
```

**Response:**
```json
{
  "tool": "retrieve_knowledge",
  "query": "cheese",
  "results": [
    { "score": 0.82, "name": "Mac & Cheese", "type": "prepared", "price": 5.99, "in_stock": true }
  ],
  "count": 1
}
```

```bash
curl -X POST http://localhost:8000/agent/tools/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"cheese"}'
```

### `GET /agent/tools/inventory?query=...`

Direct inventory lookup by item name (Postgres, not pgvector).

**Response:**
```json
{
  "tool": "lookup_inventory",
  "query": "bagel",
  "found": true,
  "item": { "id": "item3", "name": "Bagel", "type": "prepared", "service": "cold", "qty": 10, "price": 2.49 }
}
```

```bash
curl 'http://localhost:8000/agent/tools/inventory?query=bagel'
```

### `POST /admin/reindex`

Rebuild the pgvector index from all Postgres items.

```bash
curl -X POST http://localhost:8000/admin/reindex
```

**Response:**
```json
{ "ok": true, "count": 42 }
```

## Fast Path Patterns

The fast path triggers on these regex patterns (case-insensitive):

| Pattern | Example |
|---|---|
| `do you have/carry/sell/stock` | "do you have turkey?" |
| `is there / are there / got any` | "is there any beef?" |
| `how much is/does/for` | "how much is a bagel?" |
| `what's the price` | "what's the price of mac & cheese?" |
| `in stock / available` | "is turkey available?" |
| `price of / cost of` | "cost of a bagel" |

If the fast path matches but the item isn't found by name, it falls through to the normal pgvector + LLM path.

## Tool Contracts

### `lookup_inventory(query) -> dict`
- **Input:** item name string
- **Output:** `{ tool, query, found: bool, item: {id, name, type, service, qty, price} | null }`

### `retrieve_knowledge(query, top_k?) -> dict`
- **Input:** search query, optional top_k (default from `RAG_TOP_K`)
- **Output:** `{ tool, query, results: [{score, name, type, price, in_stock}, ...], count }`
- Filters results below `RAG_SIMILARITY_THRESHOLD` (default 0.75)

### `add_to_cart(item_id, qty)` / `checkout()` — stubs
- Return `{ tool, status: "not_implemented", message }`
