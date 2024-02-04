from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class GPTTaskResponse(BaseModel):
    model: str = Field(min_length=1)
    choices: List[ResponseChoice]
    usage: Usage
