from typing import Literal, Optional, List, Dict

from pydantic import BaseModel

Label = Literal["motion", "brief", "deposition", "order", "exhibit"]


class DocResult(BaseModel):
    doc_id: str
    filename: Optional[str] = None
    label: Label
    confidence: float
    meta: Dict[str, str] = {}


class EmailReport(BaseModel):
    message_id: str
    subject: str
    from_addr: str
    to_addrs: List[str]
    results: List[DocResult]
