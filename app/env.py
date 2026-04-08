from typing import Optional

from app.models import (
    ActionType,
    AgentAction,
    DifficultyLevel,
    EnvironmentState,
    Observation,
    ResetResult,
    StateResult,
    StepResult,
)
from app.tasks import get_task_by_difficulty
from app.grader import grade_state


class SupportTicketEnv:
    def __init__(self):
        self._state: Optional[EnvironmentState] = None

    def _build_observation(self) -> Observation:
        if self._state is None:
            raise ValueError("Environment is not initialized. Call reset() first.")

        task = self._state.task
        return Observation(
            task_id=task.task_id,
            difficulty=task.difficulty,
            subject=task.subject,
            message=task.message,
            history=self._state.history,
            allowed_actions=[
                ActionType.CLASSIFY_TICKET,
                ActionType.SET_PRIORITY,
                ActionType.MARK_SPAM,
                ActionType.CHOOSE_ACTION,
                ActionType.SUBMIT,
            ],
            steps_taken=self._state.steps_taken,
            steps_remaining=max(task.max_steps - self._state.steps_taken, 0),
            current_reward=self._state.reward,
            done=self._state.done,
            feedback=self._state.last_feedback,
        )

    def reset(self, difficulty: DifficultyLevel) -> ResetResult:
        task = get_task_by_difficulty(difficulty)
        if task is None:
            raise ValueError(f"No task found for difficulty: {difficulty}")

        self._state = EnvironmentState(task=task)
        self._state.last_feedback = f"Loaded {difficulty.value} task."

        return ResetResult(
            observation=self._build_observation(),
            info={"message": "Environment reset successfully."},
        )

    def step(self, action: AgentAction) -> StepResult:
        if self._state is None:
            raise ValueError("Environment is not initialized. Call reset() first.")

        if self._state.done:
            return StepResult(
                observation=self._build_observation(),
                reward=self._state.reward,
                done=True,
                info={"message": "Episode already finished."},
            )

        self._state.history.append(action)
        self._state.steps_taken += 1

        if action.type == ActionType.CLASSIFY_TICKET:
            self._state.predicted_category = action.category
            self._state.last_feedback = "Ticket classification updated."

        elif action.type == ActionType.SET_PRIORITY:
            self._state.predicted_priority = action.priority
            self._state.last_feedback = "Priority updated."

        elif action.type == ActionType.MARK_SPAM:
            self._state.predicted_spam = action.spam
            self._state.last_feedback = "Spam decision updated."

        elif action.type == ActionType.CHOOSE_ACTION:
            self._state.predicted_action = action.action
            self._state.last_feedback = "Final support action selected."

        elif action.type == ActionType.SUBMIT:
            self._state.done = True
            self._state.last_feedback = "Submission received. Episode finished."

        else:
            self._state.last_feedback = "Unknown action."

        breakdown = grade_state(self._state)
        self._state.reward = breakdown.total_score

        if self._state.steps_taken >= self._state.task.max_steps:
            self._state.done = True
            self._state.last_feedback = "Maximum steps reached. Episode finished."

        return StepResult(
            observation=self._build_observation(),
            reward=self._state.reward,
            done=self._state.done,
            info={
                "reward_breakdown": breakdown.model_dump(),
                "message": self._state.last_feedback,
            },
        )

    def state(self) -> StateResult:
        if self._state is None:
            raise ValueError("Environment is not initialized. Call reset() first.")

        return StateResult(state=self._state)
