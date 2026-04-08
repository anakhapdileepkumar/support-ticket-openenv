from app.env import SupportTicketEnv
from app.models import (
    AgentAction,
    ActionType,
    DifficultyLevel,
    TicketCategory,
    PriorityLevel,
    SupportAction,
)


def run_easy(env: SupportTicketEnv):
    env.reset(DifficultyLevel.EASY)
    env.step(
        AgentAction(type=ActionType.CLASSIFY_TICKET, category=TicketCategory.ACCOUNT)
    )
    result = env.step(AgentAction(type=ActionType.SUBMIT))
    return result.reward


def run_medium(env: SupportTicketEnv):
    env.reset(DifficultyLevel.MEDIUM)
    env.step(
        AgentAction(type=ActionType.CLASSIFY_TICKET, category=TicketCategory.BILLING)
    )
    env.step(AgentAction(type=ActionType.SET_PRIORITY, priority=PriorityLevel.HIGH))
    result = env.step(AgentAction(type=ActionType.SUBMIT))
    return result.reward


def run_hard(env: SupportTicketEnv):
    env.reset(DifficultyLevel.HARD)
    env.step(AgentAction(type=ActionType.CLASSIFY_TICKET, category=TicketCategory.SPAM))
    env.step(AgentAction(type=ActionType.SET_PRIORITY, priority=PriorityLevel.LOW))
    env.step(AgentAction(type=ActionType.MARK_SPAM, spam=True))
    env.step(
        AgentAction(type=ActionType.CHOOSE_ACTION, action=SupportAction.MARK_AS_SPAM)
    )
    result = env.step(AgentAction(type=ActionType.SUBMIT))
    return result.reward


if __name__ == "__main__":
    env = SupportTicketEnv()

    easy_score = run_easy(env)
    medium_score = run_medium(env)
    hard_score = run_hard(env)

    print("Baseline Results")
    print(f"Easy Score:   {easy_score:.2f}")
    print(f"Medium Score: {medium_score:.2f}")
    print(f"Hard Score:   {hard_score:.2f}")
