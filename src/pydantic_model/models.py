from pydantic import BaseModel, Field
from typing import Literal

class Transaction(BaseModel):
    step: int
    
    type: Literal[
        "PAYMENT", 
        "TRANSFER", 
        "CASH_OUT", 
        "DEBIT", 
        "CASH_IN"
    ]

    amount: float = Field(..., gt=0)  # must be > 0

    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)

    nameOrig: str
    nameDest: str