"""Inventory management service."""
from database_config import DatabaseConfig


class InventoryService:
    """Manages product inventory and stock levels."""

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self._inventory = {}

    def add_product(self, product_id: str, name: str, quantity: int) -> dict:
        """Add a product to inventory."""
        self._inventory[product_id] = {
            "name": name,
            "quantity": quantity
        }
        return self._inventory[product_id]

    def get_stock(self, product_id: str) -> int:
        """Get current stock level for a product."""
        product = self._inventory.get(product_id)
        if product:
            return product["quantity"]
        return 0

    def update_stock(self, product_id: str, delta: int) -> bool:
        """Update stock level by adding a delta."""
        if product_id not in self._inventory:
            return False
        self._inventory[product_id]["quantity"] += delta
        return True

    def check_availability(self, product_id: str) -> bool:
        """Check if a product is in stock."""
        return self.get_stock(product_id) > 0
