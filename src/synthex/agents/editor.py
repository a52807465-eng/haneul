"""Tool-using Editor agent with provider-neutral tool declarations."""

from __future__ import annotations

import json
from copy import deepcopy

from synthex.agents._feedback import describe_validation_error
from synthex.agents._normalize import normalize_plan
from synthex.agents.base import AgentContext
from synthex.agents.models import EditorRequest, EditorResult, ToolRecord
from synthex.llm import LLMRequest, Message, ThinkingLevel, ToolSpec
from synthex.prompts.editor_v1 import route_editor_initial_user_prompt, route_editor_system_prompt
from synthex.routejson import Route, plan_from_yaml, plan_to_yaml, render_plan

TOOLS = (
    ToolSpec(name="read_plan", description="Return the current lean RouteJSON plan as YAML."),
    ToolSpec(
        name="render_plan",
        description="Execute all graph edits and return the rendered plan.",
        parameters={
            "type": "object",
            "properties": {
                "up_to": {"type": "string"},
                "strip_maps": {"type": "boolean"},
            },
        },
    ),
    ToolSpec(name="read_edit_history", description="Describe prior edit iterations."),
    ToolSpec(
        name="revert_to",
        description="Revert to the original route for this pass.",
        parameters={
            "type": "object",
            "properties": {"iteration": {"type": "integer"}},
            "required": ["iteration"],
        },
    ),
    ToolSpec(
        name="edit_plan",
        description="Replace one exact, unique string in the plan YAML.",
        parameters={
            "type": "object",
            "properties": {
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["old_string", "new_string"],
        },
    ),
    ToolSpec(
        name="submit",
        description="Submit a renderable edited route for re-critique.",
        parameters={
            "type": "object",
            "properties": {"edit_log": {"type": "string"}},
            "required": ["edit_log"],
        },
    ),
)


class _EditSession:
    def __init__(
        self,
        route: Route,
        history: list[dict[str, object]],
        snapshots: dict[int, Route],
    ) -> None:
        self.original = deepcopy(route.to_op_plan())
        self.plan_yaml = plan_to_yaml(self.original)
        self.history = deepcopy(history)
        self.snapshots = {
            iteration: deepcopy(snapshot.to_op_plan()) for iteration, snapshot in snapshots.items()
        }
        if not self.snapshots:
            self.snapshots[0] = deepcopy(self.original)

    def _parsed(self) -> dict:
        plan = plan_from_yaml(self.plan_yaml)
        if not isinstance(plan, dict):
            raise ValueError("plan YAML must contain an object")
        return normalize_plan(plan)

    def route(self) -> Route:
        """Validate the edited plan as RouteJSON."""

        return Route.from_op_plan(self._parsed())

    def render_error(self) -> str | None:
        """Why the edited plan cannot be submitted, or None.

        Checks both halves of the gate. Renderability alone is not enough:
        ``render_plan`` ignores the RouteStep schema, so a plan with a step id of
        "S3a" renders happily and then fails Route validation at submit, which
        used to escape the turn loop and fail the whole stage.
        """

        try:
            render_plan(self._parsed())
        except Exception as exc:
            return str(exc)
        try:
            self.route()
        except Exception as exc:
            return describe_validation_error(exc)
        return None

    def dispatch(self, name: str, arguments: dict) -> str:
        try:
            if name == "read_plan":
                return self.plan_yaml
            if name == "read_edit_history":
                if not self.history:
                    return "No prior edit iterations."
                return json.dumps(self.history, indent=2)
            if name == "revert_to":
                iteration = arguments.get("iteration")
                if not isinstance(iteration, int) or iteration not in self.snapshots:
                    raise ValueError(
                        f"iteration must be one of {sorted(self.snapshots)}; got {iteration!r}"
                    )
                self.plan_yaml = plan_to_yaml(deepcopy(self.snapshots[iteration]))
                return f"OK: reverted to route snapshot {iteration}."
            if name == "render_plan":
                up_to = arguments.get("up_to")
                if isinstance(up_to, str) and up_to.isdigit():
                    up_to = int(up_to)
                rendered = render_plan(
                    self._parsed(),
                    up_to=up_to,
                    strip_frontier_maps=bool(arguments.get("strip_maps", False)),
                )
                return plan_to_yaml(rendered)
            if name == "edit_plan":
                old = arguments.get("old_string")
                new = arguments.get("new_string")
                if not isinstance(old, str) or not isinstance(new, str) or not old:
                    raise ValueError(
                        "old_string and new_string must be strings; old_string cannot be empty"
                    )
                count = self.plan_yaml.count(old)
                if count != 1:
                    raise ValueError(f"old_string must match exactly once; matched {count} times")
                candidate = self.plan_yaml.replace(old, new, 1)
                plan = plan_from_yaml(candidate)
                if not isinstance(plan, dict):
                    raise ValueError("edit did not produce a plan object")
                self.plan_yaml = candidate
                error = self.render_error()
                return "OK: edit applied" + (
                    f"; route currently does not render: {error}" if error else ""
                )
            return f"ERROR: unknown tool {name!r}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"


class Editor:
    name = "editor"
    prompt_version = "editor-v1"

    def __init__(
        self, temperature: float = 0.3, thinking_level: ThinkingLevel | None = "medium"
    ) -> None:
        self.temperature = temperature
        self.thinking_level = thinking_level

    async def run(self, request: EditorRequest, context: AgentContext) -> EditorResult:
        session = _EditSession(request.route, request.history, request.snapshots)
        rendered = plan_to_yaml(render_plan(request.route.to_op_plan()))
        initial = (
            route_editor_initial_user_prompt.replace("{{TARGET_SMILES}}", request.route.target)
            .replace("{{STRATEGY_PROMPT}}", request.strategy)
            .replace("{{FEEDBACK}}", json.dumps(request.critique.model_dump(), indent=2))
            .replace("{{RENDERED_PLAN_YAML}}", rendered.rstrip())
            .replace("{{ITERATION}}", str(request.iteration))
            .replace("{{MAX_EDIT_ITERATIONS}}", str(request.max_iterations))
        )
        messages: list[Message] = [Message(role="user", content=initial)]
        history: list[ToolRecord] = []
        no_call_turns = 0
        for turn in range(request.max_turns):
            response = await context.llm.generate(
                LLMRequest(
                    system=route_editor_system_prompt,
                    messages=tuple(messages),
                    temperature=self.temperature,
                    thinking_level=self.thinking_level,
                    tools=TOOLS,
                )
            )
            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                no_call_turns += 1
                if no_call_turns >= 3:
                    raise RuntimeError("Editor made no tool call for three consecutive turns")
                messages.append(
                    Message(role="user", content="Call an editing tool or submit the route.")
                )
                continue
            no_call_turns = 0
            turn_failed = False
            for call in response.tool_calls:
                if call.name == "submit":
                    error = session.render_error()
                    if turn_failed or error:
                        output = (
                            f"ERROR: submit refused; {error or 'an earlier tool failed this turn'}"
                        )
                        ok = False
                    else:
                        edit_log = str(call.arguments.get("edit_log", "")).strip()
                        result = session.route()
                        history.append(
                            ToolRecord(
                                turn=turn,
                                name=call.name,
                                arguments=call.arguments,
                                ok=True,
                                result=edit_log,
                            )
                        )
                        context.emit({"agent": self.name, "prompt_version": self.prompt_version})
                        return EditorResult(route=result, edit_log=edit_log, tool_history=history)
                else:
                    output = session.dispatch(call.name, call.arguments)
                    ok = not output.startswith("ERROR")
                    turn_failed = turn_failed or not ok
                history.append(
                    ToolRecord(
                        turn=turn, name=call.name, arguments=call.arguments, ok=ok, result=output
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=output,
                    )
                )
        raise RuntimeError(f"Editor did not submit within {request.max_turns} turns")


__all__ = ["Editor", "TOOLS"]
