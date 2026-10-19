"""SimpleQA Verified as Epoch AI runs it, reconstructed from Epoch's published logs.

Epoch's source (`epoch-research/benchmarks`, commit 0bdf247, task
`bench/simpleqa_verified_anti_abstention`, version 0) is private. Every field here is
read from the 64 leaderboard logs Epoch publishes for it: the dataset copy and its
sample ids, the exact prompt, the grader template, the forced `submit` tool and its
grades, and the metric. Epoch's page says the anti-abstention instruction was developed
with Generality Labs; a review of this task must state that connection.

Dataset: the mirror `codelion/SimpleQA-Verified`, 1,000 rows, not shuffled, sample id
= row number + 1, target = `answer`, metadata as the mirror's columns. Google's own
release differs from the mirror in one row (original_index 55, latitude vs longitude).

Prompt: the question, a blank line, then the anti-abstention suffix below; no system
message; plain `generate()`.

Scorer: a grader model reads the question, the reference and the response and must
call `submit` with CORRECT, INCORRECT or NOT_ATTEMPTED; the score is C, I or N and
the headline metric is accuracy (share CORRECT; NOT_ATTEMPTED counts 0). Epoch's grader
is `epoch/gpt-5.6-terra` at medium reasoning; here it is bound through the `grader`
model role so a run can name whatever stands in for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, Sample, hf_dataset
from inspect_ai.model import ChatMessageUser, GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate
from inspect_ai.tool import ToolCall, ToolFunction, tool
from inspect_ai.util import json_schema

DATASET = "codelion/SimpleQA-Verified"
DATASET_REVISION = "5a913f57326d89935cbed0ac071494e7e624b876"

# verbatim from Epoch's logs: the user turn is the question, a blank line, then this
ANTI_ABSTENTION_SUFFIX = (
    "This is a guessing game, not a test. There's no penalty for being wrong, and an "
    'uncertain guess is far more useful to me than "I don\'t know". Give your single '
    "best guess."
)

GRADER_TEMPLATE = (Path(__file__).parent / "grader_template.txt").read_text()

Grade = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
GRADE_TO_SCORE = {"CORRECT": CORRECT, "INCORRECT": INCORRECT, "NOT_ATTEMPTED": NOANSWER}


def record_to_sample(record: dict, index: int) -> Sample:
    return Sample(
        id=index + 1,
        input=f"{record['problem']}\n\n{ANTI_ABSTENTION_SUFFIX}",
        target=str(record["answer"]),
        metadata={
            "original_index": record.get("original_index"),
            "topic": record.get("topic"),
            "answer_type": record.get("answer_type"),
            "multi_step": record.get("multi_step"),
            "requires_reasoning": record.get("requires_reasoning"),
            "urls": record.get("urls"),
        },
    )


def simpleqa_verified_dataset():
    ds = hf_dataset(
        path=DATASET,
        revision=DATASET_REVISION,
        split="train",
        sample_fields=FieldSpec(input="problem", target="answer", metadata=["original_index", "topic", "answer_type", "multi_step", "requires_reasoning", "urls"]),
    )
    # Epoch's ids are row numbers from 1, in the mirror's order; rebuild the prompt
    samples = []
    for i, s in enumerate(ds):
        samples.append(
            Sample(
                id=i + 1,
                input=f"{s.input}\n\n{ANTI_ABSTENTION_SUFFIX}",
                target=s.target,
                metadata=dict(s.metadata or {}),
            )
        )
    from inspect_ai.dataset import MemoryDataset

    return MemoryDataset(samples, name="simpleqa_verified")


@tool
def submit():
    async def execute(grade: Grade) -> str:
        """Submit the grade for the predicted answer.

        Args:
            grade: The grade - CORRECT, INCORRECT, or NOT_ATTEMPTED
        """
        return grade

    return execute


@scorer(metrics=[accuracy(), stderr()])
def simpleqa_scorer() -> Scorer:
    """Epoch's grader: the template filled in, the grade returned through a forced tool call."""

    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model(role="grader")
        prompt = GRADER_TEMPLATE.format(
            question=state.input_text,
            target=target.text,
            predicted_answer=state.output.completion,
        )
        result = await grader.generate(
            [ChatMessageUser(content=prompt)],
            tools=[submit()],
            tool_choice=ToolFunction(name="submit"),
        )
        grade = None
        for call in result.message.tool_calls or []:
            if call.function == "submit":
                grade = str((call.arguments or {}).get("grade", "")).strip().upper()
        if grade not in GRADE_TO_SCORE:
            # the grader did not use the tool: fall back to the first grade word in its text
            text = result.completion.upper()
            grade = next((g for g in ("NOT_ATTEMPTED", "INCORRECT", "CORRECT") if g in text), None)
        if grade is None:
            return Score(value=NOANSWER, answer=state.output.completion, explanation="The grader returned no grade")
        return Score(
            value=GRADE_TO_SCORE[grade],
            answer=state.output.completion,
            explanation=f"The grader model responded with: {grade}",
        )

    return score


@task(name="simpleqa_verified_anti_abstention")
def simpleqa_verified_anti_abstention() -> Task:
    return Task(
        dataset=simpleqa_verified_dataset(),
        solver=[generate()],
        scorer=simpleqa_scorer(),
        version=0,
    )
