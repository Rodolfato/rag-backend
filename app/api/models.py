import datetime
from typing import List
from pydantic import BaseModel


class Query(BaseModel):
    query_text: str
    search_k: int


""" class Message(BaseModel):
    sender: str
    text: str
    timestamp: datetime


class MessageBatch(BaseModel):
    messages: List[Message] """
