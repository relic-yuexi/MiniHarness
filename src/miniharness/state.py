"""Pure event reduction. Replaying this module never executes I/O."""

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class State:
    messages: list[dict] = field(default_factory=list)
    queue: dict[str, dict] = field(default_factory=dict)
    requests: dict[str, dict] = field(default_factory=dict)
    actions: dict[str, dict] = field(default_factory=dict)
    jobs: dict[str, dict] = field(default_factory=dict)
    notifications: dict[str, dict] = field(default_factory=dict)
    usage: dict[str, dict] = field(default_factory=dict)
    todo: dict = field(default_factory=lambda: {"revision": 0, "items": []})
    config_hash: str | None = None
    epoch: int = 0
    current_turn: str | None = None
    turn_requests: list[str] = field(default_factory=list)

    def apply(self, event: dict) -> None:
        p = deepcopy(event["payload"])
        kind = event["type"]
        if kind == "runtime.configured":
            self.config_hash = p["hash"]
        elif kind == "input.enqueued":
            request_id = p["request_id"]
            self.queue[request_id] = p
            self.requests[request_id] = p | {"status": "queued"}
        elif kind == "user.accepted":
            request_id = p["request_id"]
            self.queue.pop(request_id, None)
            self.requests[request_id]["status"] = "accepted"
            self.turn_requests.append(request_id)
            self.messages.append(p["message"])
        elif kind == "input.rejected":
            request_id = p["request_id"]
            self.queue.pop(request_id, None)
            self.requests[request_id].update(status="failed", result=p["result"])
        elif kind == "turn.started":
            self.current_turn = event["turn_id"]
            self.turn_requests = []
        elif kind == "turn.ended":
            for request_id in self.turn_requests:
                self.requests[request_id].update(status=p["status"], result=p)
            self.current_turn = None
            self.turn_requests = []
        elif kind == "assistant.committed":
            self.messages.append(p["message"])
        elif kind == "action.started":
            self.actions[event["action_id"]] = p | {"status": "started", "turn_id": event["turn_id"], "step_id": event["step_id"]}
        elif kind == "action.completed":
            action_id = event["action_id"]
            previous = self.actions.get(action_id, {})
            if previous.get("status") == "completed":
                raise ValueError("Action has already completed")
            self.actions[action_id] = previous | p | {"status": "completed"}
            if p.get("todo") is not None:
                if p["todo"]["revision"] != self.todo["revision"] + 1:
                    raise ValueError("Invalid todo revision")
                self.todo = p["todo"]
            if p.get("job") is not None:
                self.jobs[p["job"]["job_id"]] = p["job"] | {"status": "queued"}
        elif kind == "tools.committed":
            self.messages.extend(p["messages"])
        elif kind == "usage.recorded":
            self.usage[p["attempt_id"]] = p["usage"]
        elif kind == "compact.committed":
            self.messages = p["messages"]
            self.epoch = p["epoch"]
        elif kind == "job.completed":
            self.jobs[p["job_id"]].update(status=p["status"], result=p["result"])
            self.notifications[p["job_id"]] = p
        elif kind == "notification.delivered":
            self.notifications.pop(p["job_id"], None)
        elif kind == "notification.accepted":
            self.messages.append(p["message"])
            self.notifications.pop(p["job_id"], None)

    @property
    def total_tokens(self) -> int:
        return sum((u.get("input_tokens_total") or 0) + (u.get("output_tokens") or 0) for u in self.usage.values())


def replay(events: list[dict]) -> State:
    state = State()
    for event in events:
        state.apply(event)
    return state
