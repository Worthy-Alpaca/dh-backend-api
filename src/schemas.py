from pydantic import BaseModel
from typing import Optional


class DummyMachine(BaseModel):
    machine: str
    cph: int = 1
    nozHeads: int = 1
    SMD: bool = False
    offsets: Optional[dict] = None
