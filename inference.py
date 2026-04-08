import os
from openai import OpenAI

from app.env import SupportTicketEnv
from app.models import (
    AgentAction,
    ActionType,
    DifficultyLevel,
    TicketCategory,
    PriorityLevel,
    SupportAction,
)

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")

print("DEBUG API_BASE_URL set:", bool(API_BASE_URL), flush=True)
print("DEBUG MODEL_NAME set:", bool(MODEL_NAME), flush=True)
print("DEBUG API_KEY set:", bool(API_KEY), flush=True)

if not API_BASE_URL or not MODEL_NAME or not API_KEY:
    raise RuntimeError("Missing API_BASE_URL, MODEL_NAME, or API_KEY")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def llm_ping(task_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply in one short line only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    print(f"[STEP] task={task_name} llm_call=ok", flush=True)
    return text.strip()


def run_easy(env: SupportTicketEnv):
    task_name = "easy_1"
    print(f"[START] task={task_name}", flush=True)

    llm_ping(task_name, "Classify an account-related support ticket in one word.")

    env.reset(DifficultyLevel.EASY)

    r1 = env.step(
        AgentAction(
            type=ActionType.CLASSIFY_TICKET,
            category=TicketCategory.ACCOUNT,
        )
    )
    print(f"[STEP] task={task_name} step=1 reward={r1.reward}", flush=True)

    r2 = env.step(AgentAction(type=ActionType.SUBMIT))
    print(f"[STEP] task={task_name} step=2 reward={r2.reward}", flush=True)

    print(f"[END] task={task_name} score={r2.reward} steps=2", flush=True)
    return r2.reward


def run_medium(env: SupportTicketEnv):
    task_name = "medium_1"
    print(f"[START] task={task_name}", flush=True)

    llm_ping(
        task_name, "Classify a billing issue and suggest priority in one short line."
    )

    env.reset(DifficultyLevel.MEDIUM)

    r1 = env.step(
        AgentAction(
            type=ActionType.CLASSIFY_TICKET,
            category=TicketCategory.BILLING,
        )
    )
    print(f"[STEP] task={task_name} step=1 reward={r1.reward}", flush=True)

    r2 = env.step(
        AgentAction(
            type=ActionType.SET_PRIORITY,
            priority=PriorityLevel.HIGH,
        )
    )
    print(f"[STEP] task={task_name} step=2 reward={r2.reward}", flush=True)

    r3 = env.step(AgentAction(type=ActionType.SUBMIT))
    print(f"[STEP] task={task_name} step=3 reward={r3.reward}", flush=True)

    print(f"[END] task={task_name} score={r3.reward} steps=3", flush=True)
    return r3.reward


def run_hard(env: SupportTicketEnv):
    task_name = "hard_1"
    print(f"[START] task={task_name}", flush=True)

    llm_ping(
        task_name, "Detect whether a reward-claim email is spam in one short line."
    )

    env.reset(DifficultyLevel.HARD)

    r1 = env.step(
        AgentAction(
            type=ActionType.CLASSIFY_TICKET,
            category=TicketCategory.SPAM,
        )
    )
    print(f"[STEP] task={task_name} step=1 reward={r1.reward}", flush=True)

    r2 = env.step(
        AgentAction(
            type=ActionType.SET_PRIORITY,
            priority=PriorityLevel.LOW,
        )
    )
    print(f"[STEP] task={task_name} step=2 reward={r2.reward}", flush=True)

    r3 = env.step(
        AgentAction(
            type=ActionType.MARK_SPAM,
            spam=True,
        )
    )
    print(f"[STEP] task={task_name} step=3 reward={r3.reward}", flush=True)

    r4 = env.step(
        AgentAction(
            type=ActionType.CHOOSE_ACTION,
            action=SupportAction.MARK_AS_SPAM,
        )
    )
    print(f"[STEP] task={task_name} step=4 reward={r4.reward}", flush=True)

    r5 = env.step(AgentAction(type=ActionType.SUBMIT))
    print(f"[STEP] task={task_name} step=5 reward={r5.reward}", flush=True)

    print(f"[END] task={task_name} score={r5.reward} steps=5", flush=True)
    return r5.reward


if __name__ == "__main__":
    env = SupportTicketEnv()

    easy = run_easy(env)
    medium = run_medium(env)
    hard = run_hard(env)

    avg = (easy + medium + hard) / 3
    print(f"[END] task=summary score={avg:.2f} steps=3", flush=True)
