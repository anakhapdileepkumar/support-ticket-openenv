from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# -----------------------------
# Difficulty Levels
# -----------------------------
class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# -----------------------------
# Ticket Categories
# -----------------------------
class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    DELIVERY = "delivery"
    ACCOUNT = "account"
    REFUND = "refund"
    SPAM = "spam"


# -----------------------------
# Priority Levels
# -----------------------------
class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# -----------------------------
# Final Support Actions
# -----------------------------
class SupportAction(str, Enum):
    REPLY_WITH_SOLUTION = "reply_with_solution"
    REQUEST_MORE_INFO = "request_more_info"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    ESCALATE_TO_BILLING = "escalate_to_billing"
    MARK_AS_SPAM = "mark_as_spam"


# -----------------------------
# Action Types
# -----------------------------
class ActionType(str, Enum):
    CLASSIFY_TICKET = "classify_ticket"
    SET_PRIORITY = "set_priority"
    MARK_SPAM = "mark_spam"
    CHOOSE_ACTION = "choose_action"
    SUBMIT = "submit"


# -----------------------------
# Agent Action
# -----------------------------
class AgentAction(BaseModel):
    type: ActionType
    category: Optional[TicketCategory] = None
    priority: Optional[PriorityLevel] = None
    spam: Optional[bool] = None
    action: Optional[SupportAction] = None


# -----------------------------
# Ticket Task (Ground Truth)
# -----------------------------
class TicketTask(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    subject: str
    message: str
    true_category: TicketCategory
    true_priority: PriorityLevel
    true_spam: bool
    true_action: SupportAction
    max_steps: int = 5


# -----------------------------
# Observation (what agent sees)
# -----------------------------
class Observation(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    subject: str
    message: str
    history: List[AgentAction] = Field(default_factory=list)
    allowed_actions: List[ActionType] = Field(default_factory=list)
    steps_taken: int = 0
    steps_remaining: int = 0
    current_reward: float = 0.0
    done: bool = False
    feedback: str = ""


# -----------------------------
# Internal State (hidden)
# -----------------------------
class EnvironmentState(BaseModel):
    task: TicketTask
    history: List[AgentAction] = Field(default_factory=list)

    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[PriorityLevel] = None
    predicted_spam: Optional[bool] = None
    predicted_action: Optional[SupportAction] = None

    reward: float = 0.0
    done: bool = False
    steps_taken: int = 0
    last_feedback: str = ""


# -----------------------------
# Step Result
# -----------------------------
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


# -----------------------------
# Reset Result
# -----------------------------
class ResetResult(BaseModel):
    observation: Observation
    info: dict = Field(default_factory=dict)


# -----------------------------
# State Result
# -----------------------------
class StateResult(BaseModel):
    state: EnvironmentState


# -----------------------------
# Reward Breakdown (for grading)
# -----------------------------
class RewardBreakdown(BaseModel):
    category_score: float = 0.0
    priority_score: float = 0.0
    spam_score: float = 0.0
    action_score: float = 0.0
    total_score: float = 0.0
