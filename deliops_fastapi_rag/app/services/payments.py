# app/services/payments.py
import os, json
import stripe
from .orders import mark_paid_and_decrement
from .firebase import ensure_firestore

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

def _to_cents(usd: float) -> int:
    return int(round(usd * 100))

def create_stripe_checkout_session(order_id: str, lines, total, currency, success_url, cancel_url):
    """
    Simplest path: pass explicit line items; price by unitPrice.
    """
    line_items = [{
        "quantity": li["qty"],
        "price_data": {
            "currency": currency.lower(),
            "unit_amount": _to_cents(li["unitPrice"]),
            "product_data": {"name": li["name"]},
        }
    } for li in lines]

    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=success_url + f"?orderId={order_id}",
        cancel_url=cancel_url + f"?orderId={order_id}",
        line_items=line_items,
        client_reference_id=order_id,
        # Optional: anti-duplication
        payment_intent_data={"metadata": {"orderId": order_id}},
        metadata={"orderId": order_id},
    )
    return {"id": session.id, "url": session.url}

async def handle_stripe_webhook(raw_body: bytes, signature: str | None):
    if not WEBHOOK_SECRET:  # dev fallback: no-op
        return

    event = stripe.Webhook.construct_event(
        payload=raw_body, sig_header=signature, secret=WEBHOOK_SECRET
    )

    typ = event["type"]
    data = event["data"]["object"]
    order_id = (data.get("client_reference_id")
                or (data.get("metadata") or {}).get("orderId")
                or (data.get("payment_intent") and stripe.PaymentIntent.retrieve(data["payment_intent"]).metadata.get("orderId")))

    # Persist raw payment event (optional)
    db = ensure_firestore()
    db.collection("payments").add({"type": typ, "orderId": order_id, "at": stripe.util.utcnow(), "payload": json.loads(raw_body.decode("utf-8"))})

    if typ in ("checkout.session.completed", "payment_intent.succeeded"):
        if order_id:
            # Atomically deduct stock and mark paid (idempotent)
            try:
                mark_paid_and_decrement(order_id)
            except Exception as e:
                # If insufficient stock at this moment => you can flag the order for manual resolution
                db.collection("orders").document(order_id).set({"status": "failed", "failure": str(e)}, merge=True)
