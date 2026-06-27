from pydantic import BaseModel


class DetectResponse(BaseModel):
    person: bool
    count: int
    confidence: float
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    coral: bool
    model: str
