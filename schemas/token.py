from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[int] = None
    # Optional[int] == Union[int, None], int or None, -----can't Optional[int, None]-----
