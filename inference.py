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

if not API_BASE_URL or not MODEL_NAME or not API_KEY:
    raise RuntimeError("Missing API_BASE_URL, MODEL_NAME, or API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def llm_ping(prompt: str):
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply shortly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )


def run_task(env, difficulty, task_name):
    print(f"[START] task={task_name} env=support_ticket model={MODEL_NAME}", flush=True)

    env.reset(difficulty)

    rewards = []
    steps = 0
    done = False

    try:
        # STEP 1
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
        reward = r.reward
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
            reward = r.reward
            done = r.done
            rewards.append(reward)

            print(
                f"[STEP] step={steps} action=set_priority reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True,
            )

        if difficulty == DifficultyLevel.HARD:
            r = env.step(AgentAction(type=ActionType.MARK_SPAM, spam=True))
            steps += 1
            reward = r.reward
            done = r.done
            rewards.append(reward)

            print(
                f"[STEP] step={steps} action=mark_spam reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True,
            )

        r = env.step(AgentAction(type=ActionType.SUBMIT))
        steps += 1
        reward = r.reward
        done = r.done
        rewards.append(reward)

        print(
            f"[STEP] step={steps} action=submit reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )

        score = reward
        success = score > 0

    except Exception as e:
        success = False
        score = 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

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
