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
    task_name = "easy_1"
    print(f"[START] task={task_name}", flush=True)

    env.reset(DifficultyLevel.EASY)

    result1 = env.step(
        AgentAction(
            type=ActionType.CLASSIFY_TICKET,
            category=TicketCategory.ACCOUNT,
        )
    )
    print(f"[STEP] step=1 reward={result1.reward}", flush=True)

    result2 = env.step(AgentAction(type=ActionType.SUBMIT))
    print(f"[STEP] step=2 reward={result2.reward}", flush=True)

    print(f"[END] task={task_name} score={result2.reward} steps=2", flush=True)
    return result2.reward


def run_medium(env: SupportTicketEnv):
    task_name = "medium_1"
    print(f"[START] task={task_name}", flush=True)

    env.reset(DifficultyLevel.MEDIUM)

    result1 = env.step(
        AgentAction(
            type=ActionType.CLASSIFY_TICKET,
            category=TicketCategory.BILLING,
        )
    )
    print(f"[STEP] step=1 reward={result1.reward}", flush=True)

    result2 = env.step(
        AgentAction(
            type=ActionType.SET_PRIORITY,
            priority=PriorityLevel.HIGH,
        )
    )
    print(f"[STEP] step=2 reward={result2.reward}", flush=True)

    result3 = env.step(AgentAction(type=ActionType.SUBMIT))
    print(f"[STEP] step=3 reward={result3.reward}", flush=True)

    print(f"[END] task={task_name} score={result3.reward} steps=3", flush=True)
    return result3.reward


def run_hard(env: SupportTicketEnv):
    task_name = "hard_1"
    print(f"[START] task={task_name}", flush=True)

    env.reset(DifficultyLevel.HARD)

    result1 = env.step(
        AgentAction(
            type=ActionType.CLASSIFY_TICKET,
            category=TicketCategory.SPAM,
        )
    )
    print(f"[STEP] step=1 reward={result1.reward}", flush=True)

    result2 = env.step(
        AgentAction(
            type=ActionType.SET_PRIORITY,
            priority=PriorityLevel.LOW,
        )
    )
    print(f"[STEP] step=2 reward={result2.reward}", flush=True)

    result3 = env.step(
        AgentAction(
            type=ActionType.MARK_SPAM,
            spam=True,
        )
    )
    print(f"[STEP] step=3 reward={result3.reward}", flush=True)

    result4 = env.step(
        AgentAction(
            type=ActionType.CHOOSE_ACTION,
            action=SupportAction.MARK_AS_SPAM,
        )
    )
    print(f"[STEP] step=4 reward={result4.reward}", flush=True)

    result5 = env.step(AgentAction(type=ActionType.SUBMIT))
    print(f"[STEP] step=5 reward={result5.reward}", flush=True)

    print(f"[END] task={task_name} score={result5.reward} steps=5", flush=True)
    return result5.reward


if __name__ == "__main__":
    env = SupportTicketEnv()

    easy_score = run_easy(env)
    medium_score = run_medium(env)
    hard_score = run_hard(env)

    print(
        f"[END] task=summary score={(easy_score + medium_score + hard_score) / 3:.2f} steps=3",
        flush=True,
    )
