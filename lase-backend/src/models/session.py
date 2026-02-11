import json
import uuid
from datetime import datetime, timezone

from ..db import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(db.Model):
    __tablename__ = "session"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(32), nullable=False, default="idle")
    config_json = db.Column(db.Text, nullable=False, default="{}")
    workspace_path = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    tasks = db.relationship("Task", backref="session", lazy=True, cascade="all, delete-orphan")
    logs = db.relationship("LogEntry", backref="session", lazy=True, cascade="all, delete-orphan")

    def config(self) -> dict:
        try:
            return json.loads(self.config_json or "{}")
        except Exception:
            return {}

    def to_dict(self) -> dict:
        # Keep keys aligned with the current frontend expectations.
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "task_count": len(self.tasks),
            "workspace_path": self.workspace_path,
            "config": self.config(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Task(db.Model):
    __tablename__ = "task"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), db.ForeignKey("session.id"), nullable=False, index=True)

    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(32), nullable=False, default="queued")  # queued|running|completed|failed|stopped
    status_detail = db.Column(db.String(128), nullable=True) # More granular status (e.g., 'failed_recoverable_tool_execution')
    progress = db.Column(db.Float, nullable=False, default=0.0)
    last_output = db.Column(db.Text, nullable=True)
    execution_log = db.Column(db.Text, nullable=True) # Stores JSON string of LLM's steps
    context_snapshot = db.Column(db.Text, nullable=True) # Stores JSON string of LLM's internal state for resumption
    failure_reason = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "description": self.description,
            "status": self.status,
            "status_detail": self.status_detail,
            "progress": self.progress,
            "last_output": (self.last_output[:5000] + "... [TRUNCATED]") if self.last_output and len(self.last_output) > 5000 else self.last_output,
            "execution_log": json.loads(self.execution_log) if self.execution_log else None,
            "context_snapshot": json.loads(self.context_snapshot) if self.context_snapshot else None,
            "failure_reason": self.failure_reason,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class LogEntry(db.Model):
    __tablename__ = "log_entry"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(36), db.ForeignKey("session.id"), nullable=False, index=True)
    task_id = db.Column(db.String(36), nullable=True, index=True)

    event_type = db.Column(db.String(64), nullable=False)
    data_json = db.Column(db.Text, nullable=False, default="{}")
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    def data(self) -> dict:
        try:
            return json.loads(self.data_json or "{}")
        except Exception:
            return {}

    def to_dict(self) -> dict:
        d = self.data()
        # Truncate large output fields to prevent OOM during serialization/transport
        if "output" in d and isinstance(d["output"], str) and len(d["output"]) > 5000:
             d["output"] = d["output"][:5000] + "... [TRUNCATED DUE TO SIZE]"
        
        return {
            "id": self.id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "data": d,
            "timestamp": self.timestamp.isoformat(),
        }


class ApiUsage(db.Model):
    __tablename__ = "api_usage"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    provider = db.Column(db.String(32), nullable=False, index=True)
    model = db.Column(db.String(128), nullable=False, index=True)
    prompt_tokens = db.Column(db.Integer, nullable=False, default=0)
    completion_tokens = db.Column(db.Integer, nullable=False, default=0)
    total_tokens = db.Column(db.Integer, nullable=False, default=0)
    cost_usd = db.Column(db.Float, nullable=False, default=0.0)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)
