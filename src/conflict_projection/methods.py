from __future__ import annotations

import re
import string
from collections.abc import Callable

from .config import ProjectionConfig
from .llm import ChatClient
from .projection import rank_with_bm25, rank_with_projection
from .prompts import TaskName, agent_prompt, aggregator_prompt, answer_prompt
from .runtime import ModelRuntime
from .schemas import Instance, MethodOutput, RankedUnit


Method = Callable[[Instance], MethodOutput]


def _format_units(
    units: tuple[RankedUnit, ...],
    *,
    annotate_weights: bool,
    expose_metadata: bool,
) -> str:
    maximum = max((unit.weight for unit in units), default=1.0) or 1.0
    blocks: list[str] = []
    for index, unit in enumerate(units, start=1):
        annotations: list[tuple[str, str]] = []
        if annotate_weights:
            annotations.append(("evidence_weight", f"{unit.weight / maximum:.2f}"))
        if expose_metadata and unit.source_domain:
            annotations.append(("source", unit.source_domain))
        if expose_metadata and unit.pub_date:
            annotations.append(("date", unit.pub_date))
        attributes = "".join(f' {key}="{value}"' for key, value in annotations)
        blocks.append(f"<document id=\"{index}\"{attributes}>\n{unit.text}\n</document>")
    return "\n\n".join(blocks)


def _all_units(instance: Instance) -> tuple[RankedUnit, ...]:
    return tuple(
        RankedUnit(
            text=document.text,
            weight=1.0 / len(instance.documents),
            doc_indices=(index,),
            source_domain=document.source_domain,
            pub_date=document.pub_date,
        )
        for index, document in enumerate(instance.documents)
    )


def answer_from_units(
    instance: Instance,
    units: tuple[RankedUnit, ...],
    client: ChatClient,
    task: TaskName,
    *,
    annotate_weights: bool = False,
    expose_metadata: bool = False,
) -> MethodOutput:
    context = _format_units(
        units, annotate_weights=annotate_weights, expose_metadata=expose_metadata
    )
    prompt = answer_prompt(
        task,
        instance.question,
        context,
        weights_visible=annotate_weights,
        metadata_visible=expose_metadata,
        options=instance.metadata.get("options") if task == "conflictbank" else None,
    )
    response = client.complete(prompt)
    return MethodOutput(
        text=response,
        selected_units=units,
        metadata={
            "annotate_weights": annotate_weights,
            "expose_metadata": expose_metadata,
        },
    )


def make_concat_method(client: ChatClient, task: TaskName) -> Method:
    def method(instance: Instance) -> MethodOutput:
        return answer_from_units(instance, _all_units(instance), client, task)

    return method


def make_bm25_method(client: ChatClient, task: TaskName, *, top_k: int = 5) -> Method:
    def method(instance: Instance) -> MethodOutput:
        return answer_from_units(instance, rank_with_bm25(instance, top_k), client, task)

    return method


def make_projected_method(
    client: ChatClient,
    runtime: ModelRuntime,
    task: TaskName,
    config: ProjectionConfig,
    *,
    annotate_weights: bool = False,
    expose_metadata: bool = False,
) -> Method:
    def method(instance: Instance) -> MethodOutput:
        ranking = rank_with_projection(instance, runtime, config)
        output = answer_from_units(
            instance,
            ranking.ranked_units,
            client,
            task,
            annotate_weights=annotate_weights,
            expose_metadata=expose_metadata,
        )
        diagnostics = {
            "converged": ranking.projection.converged,
            "iterations": ranking.projection.iterations,
            "max_violation": ranking.projection.max_violation,
            "feature_names": list(ranking.projection.feature_names),
            "lambdas": ranking.projection.lambdas.tolist(),
        }
        return MethodOutput(
            text=output.text,
            selected_units=output.selected_units,
            metadata={**output.metadata, "projection": diagnostics},
        )

    return method


def _normalize_agent_answer(response: str) -> str:
    match = re.search(r"Answer:\s*(.*?)(?:\s*Explanation:|$)", response, re.IGNORECASE | re.DOTALL)
    value = match.group(1) if match else ""
    value = value.lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(character for character in value if character not in string.punctuation)
    return " ".join(value.split())


def _agents_converged(previous: list[str], current: list[str]) -> bool:
    if len(previous) != len(current):
        return False
    old = [_normalize_agent_answer(response) for response in previous]
    new = [_normalize_agent_answer(response) for response in current]
    return all(before and after and before == after for before, after in zip(old, new))


def make_madam_method(
    client: ChatClient,
    task: TaskName,
    *,
    rounds: int = 3,
    max_agent_tokens: int = 256,
) -> Method:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    def call_agent(instance: Instance, document: str, history: str | None) -> str:
        return client.complete(
            agent_prompt(instance.question, document, history), max_tokens=max_agent_tokens
        )

    def aggregate(instance: Instance, responses: list[str]) -> str:
        joined = "\n".join(
            f"<agent id=\"{index}\">\n{response}\n</agent>"
            for index, response in enumerate(responses, start=1)
        )
        return client.complete(aggregator_prompt(task, instance.question, joined))

    def method(instance: Instance) -> MethodOutput:
        documents = [document.text for document in instance.documents]
        responses = [call_agent(instance, document, None) for document in documents]
        aggregation = aggregate(instance, responses)
        rounds_run = 1
        converged = False
        for round_index in range(2, rounds + 1):
            updated: list[str] = []
            for index, document in enumerate(documents):
                history = "\n".join(
                    f"Agent {other + 1}: {responses[other]}"
                    for other in range(len(responses))
                    if other != index
                )
                updated.append(call_agent(instance, document, history))
            converged = _agents_converged(responses, updated)
            responses = updated
            # Aggregate the current round even when it is the converged round.
            aggregation = aggregate(instance, responses)
            rounds_run = round_index
            if converged:
                break
        return MethodOutput(
            text=aggregation,
            metadata={"rounds_run": rounds_run, "converged": converged},
        )

    return method
