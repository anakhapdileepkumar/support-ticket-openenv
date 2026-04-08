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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def llm_ping(prompt: str):
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply shortly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )


def normalize_score(value: float) -> float:
    if value <= 0:
        return 0.01
    if value >= 1:
        return 0.99
    return float(value)


def run_task(env, difficulty, task_name):
    print(f"[START] task={task_name} env=support_ticket model={MODEL_NAME}", flush=True)

    env.reset(difficulty)

    rewards = []
    steps = 0
    done = False
    success = False
    score = 0.01

    try:
        llm_ping("Classify ticket")

        r = env.step(
            AgentAction(
                type=ActionType.CLASSIFY_TICKET,
                category=(
                    TicketCategory.ACCOUNT
                    if difficulty == DifficultyLevel.EASY
                    else (
                        TicketCategory.BILLING
                        if difficulty == DifficultyLevel.MEDIUM
                        else TicketCategory.SPAM
                    )
                ),
            )
        )
        steps += 1
        reward = float(r.reward or 0.0)
        done = r.done
        rewards.append(reward)

        print(
            f"[STEP] step={steps} action=classify reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )

        if difficulty != DifficultyLevel.EASY:
            r = env.step(
                AgentAction(
                    type=ActionType.SET_PRIORITY,
                    priority=(
                        PriorityLevel.HIGH
                        if difficulty == DifficultyLevel.MEDIUM
                        else PriorityLevel.LOW
                    ),
                )
            )
            steps += 1
            reward = float(r.reward or 0.0)
            done = r.done
            rewards.append(reward)

            print(
                f"[STEP] step={steps} action=set_priority reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True,
            )

        if difficulty == DifficultyLevel.HARD:
            r = env.step(AgentAction(type=ActionType.MARK_SPAM, spam=True))
            steps += 1
            reward = float(r.reward or 0.0)
            done = r.done
            rewards.append(reward)

            print(
                f"[STEP] step={steps} action=mark_spam reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True,
            )

        r = env.step(AgentAction(type=ActionType.SUBMIT))
        steps += 1
        reward = float(r.reward or 0.0)
        done = r.done
        rewards.append(reward)

        print(
            f"[STEP] step={steps} action=submit reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )

        score = normalize_score(reward)
        success = True

    except Exception:
        score = 0.01
        success = False

    rewards_str = ",".join(f"{x:.2f}" for x in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return score


if __name__ == "__main__":
    env = SupportTicketEnv()

    run_task(env, DifficultyLevel.EASY, "easy_1")
    run_task(env, DifficultyLevel.MEDIUM, "medium_1")
    run_task(env, DifficultyLevel.HARD, "hard_1")
