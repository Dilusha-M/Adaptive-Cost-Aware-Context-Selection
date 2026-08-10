"""Payment processing service."""
from base_model import BaseModel
from user_repository import UserRepository
from database_config import DatabaseConfig


class PaymentService:
    """Handles payment processing and billing."""

    def __init__(self, user_repo: UserRepository, db_config: DatabaseConfig):
        self.user_repo = user_repo
        self.db_config = db_config
        self._transactions = []

    def process_payment(self, user_id: int, amount: float, currency: str = "USD") -> dict:
        """Process a payment transaction."""
        user = self.user_repo.find_by_id(user_id)
        if not user:
            return {"status": "failed", "message": "User not found"}
        transaction = {
            "user_id": user_id,
            "amount": amount,
            "currency": currency,
            "status": "completed"
        }
        self._transactions.append(transaction)
        return transaction

    def get_transactions(self, user_id: int) -> list:
        """Retrieve transactions for a user."""
        return [t for t in self._transactions if t["user_id"] == user_id]

    def refund(self, transaction_id: int) -> dict:
        """Refund a previous transaction."""
        return {"status": "refunded", "transaction_id": transaction_id}


class PaymentLog(BaseModel):
    """Model for logging payment events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_type = None
        self.description = None
