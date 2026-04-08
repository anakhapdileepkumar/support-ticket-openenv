from app.models import (
    TicketTask,
    DifficultyLevel,
    TicketCategory,
    PriorityLevel,
    SupportAction,
)


easy_task = TicketTask(
    task_id="easy_1",
    difficulty=DifficultyLevel.EASY,
    subject="Unable to reset password",
    message="I am trying to reset my password but not receiving the email.",
    true_category=TicketCategory.ACCOUNT,
    true_priority=PriorityLevel.MEDIUM,
    true_spam=False,
    true_action=SupportAction.REPLY_WITH_SOLUTION,
)

medium_task = TicketTask(
    task_id="medium_1",
    difficulty=DifficultyLevel.MEDIUM,
    subject="Payment failed but money deducted",
    message="My payment failed but the amount was deducted from my account.",
    true_category=TicketCategory.BILLING,
    true_priority=PriorityLevel.HIGH,
    true_spam=False,
    true_action=SupportAction.ESCALATE_TO_BILLING,
)

hard_task = TicketTask(
    task_id="hard_1",
    difficulty=DifficultyLevel.HARD,
    subject="Congratulations! You won $1000",
    message="Click this link to claim your reward immediately.",
    true_category=TicketCategory.SPAM,
    true_priority=PriorityLevel.LOW,
    true_spam=True,
    true_action=SupportAction.MARK_AS_SPAM,
)

TASKS = [easy_task, medium_task, hard_task]


def get_task_by_difficulty(difficulty: DifficultyLevel):
    for task in TASKS:
        if task.difficulty == difficulty:
            return task
    return None


def get_task_by_id(task_id: str):
    for task in TASKS:
        if task.task_id == task_id:
            return task
    return None
