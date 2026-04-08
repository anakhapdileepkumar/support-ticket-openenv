from app.models import DifficultyLevel, EnvironmentState, RewardBreakdown


def grade_state(state: EnvironmentState) -> RewardBreakdown:
    task = state.task
    breakdown = RewardBreakdown()

    if task.difficulty == DifficultyLevel.EASY:
        if state.predicted_category == task.true_category:
            breakdown.category_score = 1.0

    elif task.difficulty == DifficultyLevel.MEDIUM:
        if state.predicted_category == task.true_category:
            breakdown.category_score = 0.6
        if state.predicted_priority == task.true_priority:
            breakdown.priority_score = 0.4

    elif task.difficulty == DifficultyLevel.HARD:
        if state.predicted_category == task.true_category:
            breakdown.category_score = 0.3
        if state.predicted_priority == task.true_priority:
            breakdown.priority_score = 0.2
        if state.predicted_spam == task.true_spam:
            breakdown.spam_score = 0.2
        if state.predicted_action == task.true_action:
            breakdown.action_score = 0.3

    breakdown.total_score = (
        breakdown.category_score
        + breakdown.priority_score
        + breakdown.spam_score
        + breakdown.action_score
    )

    return breakdown
