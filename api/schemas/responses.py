from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str


class IdeasResponse(BaseModel):
    ideas: list[str]
