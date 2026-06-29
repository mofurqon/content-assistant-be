from pydantic import BaseModel


class GenerateIdeasRequest(BaseModel):
    topic: str


class ContentCriteria(BaseModel):
    target_audience: str = "general readers"
    content_type: str = "Article"
    tone: str = "Professional and informative"


class PipelineStartRequest(BaseModel):
    selected_idea: str
    criteria: ContentCriteria = ContentCriteria()


class PipelineFeedbackRequest(BaseModel):
    human_feedback: str = ""
