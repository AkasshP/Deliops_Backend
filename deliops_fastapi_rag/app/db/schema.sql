-- DeliOps full Postgres schema
-- Run this to set up a fresh database:
--   psql $DATABASE_URL -f schema.sql

BEGIN;

-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- ITEMS
-- ============================================================
CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,
    name        TEXT        NOT NULL,
    type        TEXT        NOT NULL DEFAULT 'ingredient',
    service     TEXT        NOT NULL DEFAULT 'none',
    uom         TEXT        NOT NULL DEFAULT 'ea',
    category    TEXT,
    public      BOOLEAN     NOT NULL DEFAULT FALSE,
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    price_current   NUMERIC(10, 2),
    floor_qty       INT         NOT NULL DEFAULT 0,
    back_qty        INT         NOT NULL DEFAULT 0,
    total_qty       INT         NOT NULL DEFAULT 0,
    image_url       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_items_public_active
    ON items (public, active);

-- ============================================================
-- ORDERS
-- ============================================================
CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    status          TEXT        NOT NULL DEFAULT 'draft',
    customer_name   TEXT,
    customer_email  TEXT,
    lines           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    subtotal        NUMERIC(10, 2) NOT NULL DEFAULT 0,
    tax             NUMERIC(10, 2) NOT NULL DEFAULT 0,
    total           NUMERIC(10, 2) NOT NULL DEFAULT 0,
    currency        TEXT        NOT NULL DEFAULT 'USD',
    payment_provider TEXT,
    payment_intent_id TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_created_at
    ON orders (created_at DESC);

-- ============================================================
-- FEEDBACK
-- ============================================================
CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name        TEXT,
    email       TEXT,
    message     TEXT        NOT NULL,
    rating      INT         NOT NULL DEFAULT 0 CHECK (rating >= 0 AND rating <= 5),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_created_at
    ON feedback (created_at DESC);

-- ============================================================
-- ITEM EMBEDDINGS (pgvector)
-- ============================================================
CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id      TEXT PRIMARY KEY,
    item_name    TEXT        NOT NULL,
    category     TEXT,
    description  TEXT,
    price        NUMERIC(10, 2),
    in_stock     BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding    vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_item_embeddings_cosine
    ON item_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

CREATE INDEX IF NOT EXISTS idx_item_embeddings_in_stock
    ON item_embeddings (in_stock)
    WHERE in_stock = TRUE;

COMMIT;
