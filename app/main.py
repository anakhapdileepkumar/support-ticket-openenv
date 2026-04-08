from fastapi import FastAPI
from pydantic import BaseModel

from app.env import SupportTicketEnv
from app.models import AgentAction, DifficultyLevel


app = FastAPI(
    title="Support Ticket Triage OpenEnv",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
env = SupportTicketEnv()


class ResetRequest(BaseModel):
    difficulty: DifficultyLevel


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset_env(request: ResetRequest):
    result = env.reset(request.difficulty)
    return result.model_dump()


@app.post("/step")
def step_env(action: AgentAction):
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def get_state():
    result = env.state()
    return result.model_dump()
