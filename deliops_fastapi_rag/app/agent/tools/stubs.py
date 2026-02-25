from __future__ import annotations

from typing import Dict, Any


def add_to_cart(item_id: str, qty: int = 1) -> Dict[str, Any]:
    """Stub: add an item to the user's cart."""
    return {
        "tool": "add_to_cart",
        "status": "not_implemented",
        "message": "Cart functionality is not yet available.",
    }


def checkout() -> Dict[str, Any]:
    """Stub: initiate checkout for the current cart."""
    return {
        "tool": "checkout",
        "status": "not_implemented",
        "message": "Checkout functionality is not yet available.",
    }
