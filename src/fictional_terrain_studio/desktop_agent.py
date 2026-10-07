from __future__ import annotations

from dataclasses import dataclass, field
import json
from queue import Queue
from threading import Event
from typing import Any

from .agent_tools import AgentPolicy, AgentSession, ToolSpec
from .app_controller import EditorController
from .ollama_agent import OllamaConversation, OllamaTerrainAgent, OllamaResult


@dataclass(slots=True)
class ApprovalRequest:
    tool_name: str
    description: str
    risk: str
    arguments: dict[str, Any]
    decision: Event = field(default_factory=Event)
    approved: bool = False


class DesktopAgentBridge:
    """Connect the desktop editor to the shared FTS agent/controller surface.

    The bridge contains no Tk calls, so model execution can live on a worker
    thread while the UI services approval requests on its main thread.
    """

    def __init__(self, controller: EditorController) -> None:
        self.controller = controller
        self.approval_requests: Queue[ApprovalRequest] = Queue()
        self.event_queue: Queue[dict[str, Any]] = Queue()
        self.allow_edits = True
        self.allow_final_bake = False
        self.prompt_for_risks: tuple[str, ...] = ("destructive", "expensive")
        self.session = AgentSession(
            controller,
            policy=self._policy(),
            approval_hook=self._request_approval,
            event_hook=self.event_queue.put,
            cancel_check=self._check_cancelled,
        )
        self.agent: OllamaTerrainAgent | None = None
        self.conversation: OllamaConversation | None = None
        self.model = "qwen3:30b"
        self.base_url = "http://127.0.0.1:11434"
        self.think = True
        self.cancel_event = Event()


    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise RuntimeError("Agent request cancelled")

    def _policy(self) -> AgentPolicy:
        return AgentPolicy(
            read_only=not self.allow_edits,
            allow_final_bake=self.allow_final_bake,
            approval_risks=self.prompt_for_risks,
        )

    def set_policy(self, *, allow_edits: bool, allow_final_bake: bool, prompt_for_risks: tuple[str, ...] | None = None) -> None:
        self.allow_edits = bool(allow_edits)
        self.allow_final_bake = bool(allow_final_bake)
        if prompt_for_risks is not None:
            self.prompt_for_risks = tuple(prompt_for_risks)
        self.session.policy = self._policy()

    def configure(self, *, model: str, base_url: str, think: bool) -> None:
        model = model.strip() or "qwen3:30b"
        base_url = base_url.strip() or "http://127.0.0.1:11434"
        changed = model != self.model or base_url.rstrip("/") != self.base_url.rstrip("/") or bool(think) != self.think
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = bool(think)
        if changed or self.agent is None:
            self.agent = OllamaTerrainAgent(
                self.session,
                model=self.model,
                base_url=self.base_url,
                think=self.think,
            )
            self.conversation = self.agent.conversation()

    def reset_conversation(self) -> None:
        if self.conversation is not None:
            self.conversation.reset()

    def test_connection(self) -> list[str]:
        self.configure(model=self.model, base_url=self.base_url, think=self.think)
        assert self.agent is not None
        return self.agent.list_models()

    def send(self, prompt: str, *, view_mode: str = "hillshade", include_editor_context: bool = True) -> OllamaResult:
        if self.controller.project is None:
            raise RuntimeError("Open a terrain project before using the embedded agent")
        self.cancel_event.clear()
        if self.conversation is None:
            self.configure(model=self.model, base_url=self.base_url, think=self.think)
        assert self.conversation is not None
        context = None
        if include_editor_context:
            context = json.dumps(self.session.editor_context(view_mode), indent=2, sort_keys=True)
        return self.conversation.send(prompt, context=context, cancel_event=self.cancel_event)


    def stop(self) -> None:
        """Request cancellation of the active agent loop.

        A request already blocked inside Ollama's HTTP response may only return
        when that request completes or times out; tool execution stops at the
        next cancellation boundary.
        """
        self.cancel_event.set()

    def _request_approval(self, tool: ToolSpec, arguments: dict[str, Any]) -> bool:
        req = ApprovalRequest(tool.name, tool.description, tool.risk, dict(arguments))
        self.approval_requests.put(req)
        # The Tk main thread sets the decision. A long timeout prevents a dead
        # model thread if the application is closed while a dialog is pending.
        if not req.decision.wait(timeout=600.0):
            return False
        return bool(req.approved)
