import os
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from sqlalchemy import func, text

from .config import load_settings, get_default_config, load_config_safely, save_config_safely
from .db import db
from .models.session import ApiUsage, LogEntry, Session, Task
from .core.orchestrator import AgentOrchestrator
from .core.code_quality_auditor import CodeQualityAuditor
from .core.predictive_analyzer import PredictiveAnalyzer
from .core.natural_language_interface import classify_intent, infer_category
from .core.toolbus import ToolBus
from .core import security
import shutil
import time
import json
import base64
import re
import tempfile
import subprocess
import hashlib
from datetime import datetime, timezone
import requests




def create_app() -> Flask:
    project_root = str(Path(__file__).resolve().parents[1])  # lase-backend/
    settings = load_settings(project_root)
    bgm_library_dir = os.getenv("LASE_BGM_LIBRARY_DIR", os.path.join(project_root, "bgm_library"))
    os.makedirs(bgm_library_dir, exist_ok=True)

    os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)
    os.makedirs(settings.workspaces_dir, exist_ok=True)
    os.makedirs(settings.generated_images_dir, exist_ok=True)
    os.makedirs(settings.logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.models_config_path), exist_ok=True)

    app = Flask(__name__, static_folder=settings.static_dir, static_url_path="")
    CORS(app)

    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{settings.database_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    with app.app_context():
        db.create_all()
        # Lightweight schema migration for existing SQLite DBs.
        try:
            cols = [r[1] for r in db.session.execute(text("PRAGMA table_info(task)")).fetchall()]
            if "category" not in cols:
                db.session.execute(text("ALTER TABLE task ADD COLUMN category VARCHAR(32)"))
                db.session.commit()
        except Exception:
            db.session.rollback()

        # Ensure models.yaml exists (matches legacy docs/deploy behavior).
        if not os.path.exists(settings.models_config_path):
            save_config_safely(settings.models_config_path, get_default_config())

    orchestrator = AgentOrchestrator(
        app=app,
        workspaces_dir=settings.workspaces_dir,
        generated_images_dir=settings.generated_images_dir,
        models_yaml_path=settings.models_config_path,
    )
    toolbus = ToolBus(workspaces_dir=settings.workspaces_dir, generated_images_dir=settings.generated_images_dir)
    auditor = CodeQualityAuditor()
    predictor = PredictiveAnalyzer()
    actual_usage_cache = {"ts": 0.0, "key": None, "data": None}

    def _is_within_workspace(workspace_path: str, candidate_path: str) -> bool:
        try:
            ws_real = os.path.realpath(workspace_path)
            cand_real = os.path.realpath(candidate_path)
            return os.path.commonpath([ws_real, cand_real]) == ws_real
        except Exception:
            return False

    def _infer_story_mood(lines: list[str]) -> str:
        blob = " ".join([str(x or "") for x in lines]).lower()
        if any(k in blob for k in ["fear", "dark", "haunted", "mystery", "shadow", "secret", "nightmare"]):
            return "mystery"
        if any(k in blob for k in ["battle", "quest", "adventure", "dragon", "hero", "journey", "epic"]):
            return "adventure"
        if any(k in blob for k in ["happy", "joy", "love", "friend", "magic", "dream", "peace"]):
            return "uplifting"
        return "calm"

    def _collect_bgm_candidates(mood: str) -> list[str]:
        exts = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
        mood_key = str(mood or "calm").strip().lower() or "calm"
        candidates: list[str] = []
        mood_dir = os.path.join(bgm_library_dir, mood_key)
        if os.path.isdir(mood_dir):
            for root, _, files in os.walk(mood_dir):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in exts:
                        candidates.append(os.path.join(root, fn))
        if candidates:
            return sorted(candidates)

        for root, _, files in os.walk(bgm_library_dir):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                low = fn.lower()
                if mood_key in low:
                    candidates.append(os.path.join(root, fn))
        if candidates:
            return sorted(candidates)

        # Fallback to any track in library.
        for root, _, files in os.walk(bgm_library_dir):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    candidates.append(os.path.join(root, fn))
        return sorted(candidates)

    def _pick_bgm_track(task_id: str, mood: str) -> str | None:
        candidates = _collect_bgm_candidates(mood)
        if not candidates:
            return None
        seed = hashlib.sha256(f"{task_id}:{mood}".encode("utf-8")).hexdigest()
        idx = int(seed[:8], 16) % len(candidates)
        return candidates[idx]

    def _to_bgm_url(track_path: str) -> str:
        rel = os.path.relpath(track_path, bgm_library_dir).replace("\\", "/")
        return f"/api/bgm/{rel}"

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/sessions")
    def list_sessions():
        sessions = Session.query.order_by(Session.updated_at.desc()).all()
        return jsonify([s.to_dict() for s in sessions])

    @app.post("/api/sessions")
    def create_session():
        print("DEBUG: Entered create_session", flush=True)
        payload = request.get_json(force=True, silent=True) or {}
        name = (payload.get("name") or "").strip() or "New Session"
        config = payload.get("config") or {}
        s = Session(name=name, config_json=json.dumps(config))
        db.session.add(s)
        print("DEBUG: About to commit session", flush=True)
        db.session.commit()
        print("DEBUG: Session committed", flush=True)

        # Create workspace eagerly so UI can show it immediately.
        os.makedirs(settings.workspaces_dir, exist_ok=True)
        ws = os.path.join(settings.workspaces_dir, s.id)
        os.makedirs(ws, exist_ok=True)
        s.workspace_path = ws
        db.session.commit()

        return jsonify(s.to_dict())

    def _delete_session_impl(session_id: str):
        sess = Session.query.get(session_id)
        if not sess:
            return jsonify({"error": "Session not found"}), 404

        running_count = Task.query.filter_by(session_id=session_id, status="running").count()
        if running_count > 0:
            return jsonify({"error": "Session has running tasks. Pause/stop them before deleting."}), 409

        workspace_path = sess.workspace_path
        db.session.delete(sess)
        db.session.commit()

        if workspace_path and os.path.exists(workspace_path):
            shutil.rmtree(workspace_path, ignore_errors=True)

        return jsonify({"status": "deleted", "session_id": session_id})

    @app.delete("/api/sessions/<session_id>")
    def delete_session(session_id: str):
        return _delete_session_impl(session_id)

    @app.post("/api/sessions/<session_id>/delete")
    def delete_session_compat(session_id: str):
        return _delete_session_impl(session_id)

    @app.get("/api/sessions/<session_id>/tasks")
    def list_session_tasks(session_id: str):
        tasks = Task.query.filter_by(session_id=session_id).order_by(Task.created_at.desc()).all()
        return jsonify([t.to_dict() for t in tasks])

    @app.get("/api/sessions/<session_id>/logs")
    def list_session_logs(session_id: str):
        limit = int(request.args.get("limit", "50"))
        limit = max(1, min(500, limit))
        logs = LogEntry.query.filter_by(session_id=session_id).order_by(LogEntry.timestamp.desc()).limit(limit).all()
        logs = list(reversed(logs))
        return jsonify([l.to_dict() for l in logs])

    @app.post("/api/tasks")
    def create_task():
        def _detect_requested_story_minutes(text: str) -> int | None:
            t = str(text or "").strip().lower()
            if not t:
                return None
            if "half hour" in t or "half-hour" in t:
                return 30
            m_hr = re.search(r"\b(\d{1,2})\s*(?:hours|hour|hrs|hr)\b", t)
            if m_hr:
                return max(1, min(120, int(m_hr.group(1)) * 60))
            m_min = re.search(r"\b(\d{1,3})\s*(?:minutes|minute|mins|min)\b", t)
            if m_min:
                return max(1, min(120, int(m_min.group(1))))
            m_compact = re.search(r"\b(\d{1,3})\s*m\b", t)
            if m_compact:
                return max(1, min(120, int(m_compact.group(1))))
            return None

        payload = request.get_json(force=True, silent=True) or {}
        session_id = payload.get("session_id")
        description = (payload.get("description") or "").strip()
        category = (payload.get("category") or "").strip().lower()
        inferred_category = infer_category(description)
        if inferred_category:
            category = inferred_category
        if category and category not in {"image", "website", "research", "story", "android_app", "python_app"}:
            return jsonify({"error": "invalid_category"}), 400
        story_options = payload.get("story_options") or {}
        if story_options and not isinstance(story_options, dict):
            return jsonify({"error": "invalid_story_options"}), 400
        generate_illustrations = bool(story_options.get("generate_illustrations", True))
        illustration_count_mode = str(story_options.get("illustration_count_mode", "auto") or "auto").strip().lower()
        if illustration_count_mode not in {"auto", "manual"}:
            illustration_count_mode = "auto"
        illustration_count = int(story_options.get("illustration_count", 10) or 10)
        illustration_count = max(0, min(10, illustration_count))
        illustration_style = str(story_options.get("illustration_style", "ghibli") or "ghibli").strip().lower()
        allowed_styles = {"ghibli", "storybook", "anime", "cinematic", "fantasy", "watercolor", "photorealistic"}
        if illustration_style not in allowed_styles:
            illustration_style = "ghibli"
        target_minutes = int(story_options.get("duration_minutes", 5) or 5)
        target_minutes = max(3, min(30, target_minutes))
        detected_minutes = _detect_requested_story_minutes(description)
        if detected_minutes is not None:
            target_minutes = max(3, min(30, detected_minutes))
        auto_start = bool(payload.get("auto_start", False))

        if not session_id or not Session.query.get(session_id):
            return jsonify({"error": "invalid_session_id"}), 400
        if not description:
            return jsonify({"error": "missing_description"}), 400

        initial_snapshot = {"category": category} if category else {}
        if category == "story":
            initial_snapshot["story_options"] = {
                "generate_illustrations": generate_illustrations,
                "illustration_count_mode": illustration_count_mode,
                "illustration_count": illustration_count,
                "illustration_style": illustration_style,
                "target_minutes": target_minutes,
            }
        t = Task(
            session_id=session_id,
            category=category or None,
            description=description,
            status="queued",
            progress=0.0,
            last_output=None,
            context_snapshot=json.dumps(initial_snapshot) if initial_snapshot else None,
        )
        db.session.add(t)
        db.session.commit()

        if auto_start:
            orchestrator.start_task(t.id)

        return jsonify(t.to_dict())

    @app.post("/api/tasks/<task_id>/resume")
    def resume_task(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        
        # Only allow resuming if the task is failed or stopped
        if task.status not in ["failed", "stopped"]:
            return jsonify({"error": "Task can only be resumed if failed or stopped"}), 400
        
        # Reset status to queued or running, orchestrator will pick it up
        task.status = "queued" # Orchestrator will set to running
        task.failure_reason = None # Clear previous failure reason
        task.status_detail = None # Clear previous status detail
        db.session.commit()
        
        # Trigger the orchestrator to start/resume the task
        orchestrator.start_task(task.id)
        
        return jsonify(task.to_dict())

    @app.post("/api/tasks/<task_id>/decision")
    def task_decision(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        payload = request.get_json(force=True, silent=True) or {}
        decision = str(payload.get("decision") or "").strip().lower()
        proposal = str(payload.get("proposal") or "").strip()
        if decision not in {"accept", "counter"}:
            return jsonify({"error": "decision must be 'accept' or 'counter'"}), 400
        if decision == "counter" and not proposal:
            return jsonify({"error": "proposal required for counter decision"}), 400

        messages = []
        if task.execution_log:
            try:
                messages = json.loads(task.execution_log) or []
            except Exception:
                messages = []

        if decision == "accept":
            decision_msg = (
                "User decision: ACCEPT.\n"
                "Proceed with the requested permission/improvement and continue execution."
            )
        else:
            decision_msg = (
                "User decision: COUNTER PROPOSAL.\n"
                f"Updated direction: {proposal}\n"
                "Follow this proposal and continue execution."
            )
        messages.append({"role": "user", "content": decision_msg})
        task.execution_log = json.dumps(messages)
        task.status = "queued"
        task.status_detail = "user_decision_received"
        task.failure_reason = None
        db.session.commit()

        log = LogEntry(
            session_id=task.session_id,
            task_id=task.id,
            event_type="user_decision_submitted",
            data_json=json.dumps({"decision": decision, "has_proposal": bool(proposal)}),
        )
        db.session.add(log)
        db.session.commit()

        orchestrator.start_task(task.id)
        return jsonify(task.to_dict())

    @app.post("/api/tasks/<task_id>/pause")
    def pause_task(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        force = str(request.args.get("force", "")).strip().lower() in {"1", "true", "yes", "on"}
        if task.status not in {"running", "queued"}:
            return jsonify({"error": "Task can only be paused when queued or running"}), 400

        is_active = orchestrator.is_task_active(task.id)
        orchestrator.request_stop(task.id)
        if task.status == "queued" or force or not is_active:
            task.status = "stopped"
            task.status_detail = "paused_by_user" if not force else "paused_forcefully"
        else:
            task.status_detail = "pause_requested"
        db.session.commit()
        return jsonify(task.to_dict())

    @app.post("/api/tasks/<task_id>/restart")
    def restart_task(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        if task.status == "running":
            return jsonify({"error": "Task is running. Pause first, then restart."}), 409

        task.status = "queued"
        task.progress = 0.0
        task.status_detail = "restart_requested"
        task.failure_reason = None
        task.last_output = None
        task.execution_log = None
        task.context_snapshot = None
        db.session.commit()

        orchestrator.start_task(task.id)
        return jsonify(task.to_dict())

    @app.delete("/api/tasks/<task_id>")
    def delete_task(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        force = str(request.args.get("force", "")).strip().lower() in {"1", "true", "yes", "on"}

        # Allow force-delete by first requesting stop and waiting briefly.
        if task.status == "running":
            if not force:
                return jsonify({"error": "Task is running. Pause or retry delete with force."}), 409
            orchestrator.request_stop(task.id)
            deadline = time.time() + 3.0
            while orchestrator.is_task_active(task.id) and time.time() < deadline:
                time.sleep(0.1)
            if orchestrator.is_task_active(task.id):
                return jsonify({"error": "Task is still stopping. Retry delete in a moment."}), 409

        # Remove timeline entries for this task for full deletion semantics.
        LogEntry.query.filter_by(task_id=task.id).delete(synchronize_session=False)
        db.session.delete(task)
        db.session.commit()
        return jsonify({"status": "deleted", "task_id": task_id})

    @app.post("/api/tasks/<task_id>/refine")
    def refine_task(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        payload = request.get_json(force=True, silent=True) or {}
        message = (payload.get("message") or "").strip()
        auto_start = bool(payload.get("auto_start", True))
        if not message:
            return jsonify({"error": "missing_message"}), 400

        description = (
            f"Refine task {task.id[:8]} based on user feedback:\n{message}\n\n"
            f"Original task:\n{task.description}"
        )
        new_task = Task(
            session_id=task.session_id,
            category=task.category,
            description=description,
            status="queued",
            progress=0.0,
            status_detail="refinement_requested",
            last_output=None,
            context_snapshot=task.context_snapshot if task.context_snapshot else None,
        )
        db.session.add(new_task)
        db.session.commit()

        log = LogEntry(
            session_id=task.session_id,
            task_id=new_task.id,
            event_type="task_refinement_requested",
            data_json=json.dumps(
                {
                    "parent_task_id": task.id,
                    "message": message,
                }
            ),
        )
        db.session.add(log)
        db.session.commit()

        if auto_start:
            orchestrator.start_task(new_task.id)

        return jsonify({"task": new_task.to_dict(), "parent_task_id": task.id})

    @app.route("/api/models/config", methods=["GET", "POST"])
    def manage_model_config():
        config_path = settings.models_config_path
        
        if request.method == "GET":
            # Return current config (masking keys)
            try:
                config = load_config_safely(config_path)
                defaults = get_default_config()
                for sec in [
                    "default_models",
                    "openai_settings",
                    "gemini_settings",
                    "anthropic_settings",
                    "deepseek_settings",
                    "ollama_settings",
                    "stability_settings",
                ]:
                    if sec not in config and sec in defaults:
                        config[sec] = defaults[sec]
                
                # Mask keys for security in UI (simple approach)
                # In a real app, never send keys back. 
                # Here we send them masked or blank to indicate they are set.
                def mask(val):
                    return "****" if val and "YOUR" not in val else val

                # Inject keys from keyring if available
                # If keyring is disabled, we rely on what's loaded from the file in 'config'
                def inject(sec, key_name):
                    k = security.get_api_key(key_name)
                    if not k:
                        env_map = {
                            "openai_api_key": "LASE_OPENAI_API_KEY",
                            "gemini_api_key": "LASE_GEMINI_API_KEY",
                            "anthropic_api_key": "LASE_ANTHROPIC_API_KEY",
                            "deepseek_api_key": "LASE_DEEPSEEK_API_KEY",
                            "stability_api_key": "LASE_STABILITY_API_KEY",
                        }
                        env_name = env_map.get(key_name)
                        if env_name:
                            k = os.getenv(env_name)
                    if k:
                        if sec not in config: config[sec] = {}
                        config[sec]["api_key"] = k

                inject("openai_settings", "openai_api_key")
                inject("gemini_settings", "gemini_api_key")
                inject("anthropic_settings", "anthropic_api_key")
                inject("deepseek_settings", "deepseek_api_key")
                inject("stability_settings", "stability_api_key")

                # Mask all keys
                for sec in ["openai_settings", "gemini_settings", "anthropic_settings", "deepseek_settings", "stability_settings"]:
                    if sec in config and "api_key" in config[sec]:
                        config[sec]["api_key"] = mask(config[sec]["api_key"])
                
                return jsonify(config)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        elif request.method == "POST":
            # Update config
            try:
                payload = request.get_json(force=True)
                # Load existing first to preserve unedited values
                current_config = load_config_safely(config_path)

                # Deep merge or selective update
                # For simplicity, we assume payload structure matches config structure
                # and we only update provided fields.
                
                # Helper to update settings dict and save keys to keyring
                def update_section(section):
                    if section not in payload: return
                    if section not in current_config: current_config[section] = {}
                    
                    # Special handling for api_key
                    if "api_key" in payload[section]:
                        val = payload[section]["api_key"]
                        
                        if val == "****":
                            # User sent back the mask, meaning no change. Remove from payload.
                            del payload[section]["api_key"]
                        elif val:
                            # User provided a new key
                            provider_key = f"{section.replace('_settings', '')}_api_key"
                            # Try secure storage
                            if security.set_api_key(provider_key, val):
                                # Success: Remove from payload so it doesn't get written to file
                                del payload[section]["api_key"]
                                # Also clean up from existing file config if present (migration)
                                if "api_key" in current_config[section]:
                                    del current_config[section]["api_key"]
                            else:
                                raise ValueError(f"Secure storage unavailable for {section}. API keys are not stored in plain text.")
                        else:
                            # Empty string implies clearing the key
                            provider_key = f"{section.replace('_settings', '')}_api_key"
                            security.set_api_key(provider_key, "")
                            if "api_key" in current_config[section]:
                                del current_config[section]["api_key"]
                            del payload[section]["api_key"]

                    for k, v in payload[section].items():
                        current_config[section][k] = v

                update_section("openai_settings")
                update_section("gemini_settings")
                update_section("anthropic_settings")
                update_section("deepseek_settings")
                update_section("stability_settings")
                update_section("ollama_settings")
                
                if "default_models" in payload:
                     current_config["default_models"] = payload["default_models"]

                save_config_safely(config_path, current_config)
                
                if hasattr(orchestrator, "llm"):
                    orchestrator.llm.reload_config()

                return jsonify({"status": "updated"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    @app.get("/api/models")
    def list_models():
        # Delegate to orchestrator's LLM adapter to get real available models
        if hasattr(orchestrator, "llm"):
            return jsonify(orchestrator.llm.list_available_models())
        
        # Fallback if something is wrong
        return jsonify([])

    @app.get("/api/tools")
    def list_tools():
        return jsonify(toolbus.list_tools())

    # Self-improvement API (minimal, design-aligned endpoints)
    @app.post("/api/nl/intent")
    def nl_intent():
        payload = request.get_json(force=True, silent=True) or {}
        text = (payload.get("text") or "").strip()
        intent = classify_intent(text)
        return jsonify({"task_type": intent.task_type, "confidence": intent.confidence})

    @app.post("/api/quality/audit")
    def quality_audit():
        payload = request.get_json(force=True, silent=True) or {}
        target_path = (payload.get("path") or "").strip() or str(Path(__file__).resolve().parents[1])
        return jsonify(auditor.run(target_path))

    @app.get("/api/predict")
    def predict():
        return jsonify(predictor.predict())

    @app.get("/api/usage/summary")
    def usage_summary():
        provider = (request.args.get("provider") or "openai").strip().lower()
        now = datetime.now(timezone.utc)
        last_hour = now.timestamp() - 3600

        q = ApiUsage.query.filter_by(provider=provider)
        total = q.with_entities(
            func.count(ApiUsage.id),
            func.coalesce(func.sum(ApiUsage.prompt_tokens), 0),
            func.coalesce(func.sum(ApiUsage.completion_tokens), 0),
            func.coalesce(func.sum(ApiUsage.total_tokens), 0),
            func.coalesce(func.sum(ApiUsage.cost_usd), 0.0),
        ).first()

        recent = (
            ApiUsage.query.filter(ApiUsage.provider == provider, ApiUsage.timestamp >= datetime.fromtimestamp(last_hour, tz=timezone.utc))
            .with_entities(
                func.count(ApiUsage.id),
                func.coalesce(func.sum(ApiUsage.total_tokens), 0),
                func.coalesce(func.sum(ApiUsage.cost_usd), 0.0),
            )
            .first()
        )

        by_model_rows = (
            ApiUsage.query.filter_by(provider=provider)
            .with_entities(
                ApiUsage.model,
                func.coalesce(func.sum(ApiUsage.total_tokens), 0).label("tokens"),
                func.coalesce(func.sum(ApiUsage.cost_usd), 0.0).label("cost"),
            )
            .group_by(ApiUsage.model)
            .order_by(func.sum(ApiUsage.cost_usd).desc())
            .limit(10)
            .all()
        )

        return jsonify(
            {
                "provider": provider,
                "totals": {
                    "requests": int(total[0] or 0),
                    "prompt_tokens": int(total[1] or 0),
                    "completion_tokens": int(total[2] or 0),
                    "total_tokens": int(total[3] or 0),
                    "cost_usd": round(float(total[4] or 0.0), 6),
                },
                "last_hour": {
                    "requests": int(recent[0] or 0),
                    "total_tokens": int(recent[1] or 0),
                    "cost_usd": round(float(recent[2] or 0.0), 6),
                },
                "by_model": [
                    {"model": r[0], "total_tokens": int(r[1] or 0), "cost_usd": round(float(r[2] or 0.0), 6)}
                    for r in by_model_rows
                ],
            }
        )

    @app.get("/api/usage/openai/actual")
    def usage_openai_actual():
        def _effective_key(v: str | None) -> str | None:
            if not v:
                return None
            s = str(v).strip()
            if not s or s.startswith("****") or "YOUR_" in s:
                return None
            return s

        def _iter_results(payload):
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, list):
                    for bucket in data:
                        if isinstance(bucket, dict):
                            results = bucket.get("results")
                            if isinstance(results, list):
                                for row in results:
                                    if isinstance(row, dict):
                                        yield row
            elif isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        yield row

        def _sum_cost_usd(payload) -> float:
            total = 0.0
            for row in _iter_results(payload):
                amount = row.get("amount")
                if isinstance(amount, dict) and isinstance(amount.get("value"), (int, float)):
                    total += float(amount["value"])
            return total

        def _sum_usage_fields(payload) -> tuple[int, int, int]:
            reqs = 0
            in_tok = 0
            out_tok = 0
            for row in _iter_results(payload):
                if isinstance(row.get("num_model_requests"), (int, float)):
                    reqs += int(row["num_model_requests"])
                if isinstance(row.get("input_tokens"), (int, float)):
                    in_tok += int(row["input_tokens"])
                if isinstance(row.get("output_tokens"), (int, float)):
                    out_tok += int(row["output_tokens"])
            return reqs, in_tok, out_tok

        def _fetch_paginated(url: str, headers: dict, params: dict, timeout: int = 30):
            merged = dict(params or {})
            merged.setdefault("limit", 100)
            rows = []
            next_page = None
            for _ in range(10):
                if next_page:
                    merged["page"] = next_page
                elif "page" in merged:
                    del merged["page"]
                resp = requests.get(url, headers=headers, params=merged, timeout=timeout)
                if not resp.ok:
                    return None, resp
                payload = resp.json()
                rows.append(payload)
                next_page = payload.get("next_page") if isinstance(payload, dict) else None
                if not next_page:
                    break
            return rows, None

        config = load_config_safely(settings.models_config_path)
        openai_settings = config.get("openai_settings") or {}
        base_url = (openai_settings.get("org_base_url") or "https://api.openai.com/v1").rstrip("/")
        organization_id = str(openai_settings.get("organization_id") or os.getenv("OPENAI_ORG_ID") or "").strip()
        project_id = str(openai_settings.get("project_id") or os.getenv("OPENAI_PROJECT_ID") or "").strip()
        api_key = (
            _effective_key(security.get_api_key("openai_api_key"))
            or _effective_key(os.getenv("LASE_OPENAI_API_KEY"))
            or _effective_key(openai_settings.get("api_key"))
        )
        if not api_key:
            return jsonify({"provider": "openai", "error": "openai_api_key_not_configured"}), 400

        try:
            window_days = int(request.args.get("window_days", "30"))
        except ValueError:
            window_days = 30
        window_days = max(1, min(90, window_days))
        now_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = now_ts - window_days * 86400

        cache_key = f"{api_key[:12]}:{window_days}:{base_url}"
        ttl_s = 60
        if actual_usage_cache["data"] and actual_usage_cache["key"] == cache_key and (time.time() - actual_usage_cache["ts"]) < ttl_s:
            return jsonify(actual_usage_cache["data"])

        headers = {"Authorization": f"Bearer {api_key}"}
        if organization_id:
            headers["OpenAI-Organization"] = organization_id
        if project_id:
            headers["OpenAI-Project"] = project_id
        out = {
            "provider": "openai",
            "window_days": window_days,
            "start_time": start_ts,
            "end_time": now_ts,
            "actual_spend_usd": None,
            "actual_requests": None,
            "actual_input_tokens": None,
            "actual_output_tokens": None,
            "source": "openai_org_api",
            "org_context": {
                "organization_id": organization_id or None,
                "project_id": project_id or None,
            },
        }

        # Costs endpoint (official org-level billing API)
        try:
            costs_pages, c_err = _fetch_paginated(
                f"{base_url}/organization/costs",
                headers,
                {"start_time": start_ts, "end_time": now_ts, "limit": 100},
                30,
            )
            if c_err is None and costs_pages is not None:
                out["actual_spend_usd"] = round(sum(_sum_cost_usd(p) for p in costs_pages), 6)
            else:
                out["cost_error"] = f"http_{c_err.status_code}"
                try:
                    out["cost_error_detail"] = c_err.json()
                except Exception:
                    out["cost_error_detail"] = (c_err.text or "")[:300]
        except Exception as e:
            out["cost_error"] = str(e)

        # Usage endpoint (token/request consumption)
        try:
            usage_pages, u_err = _fetch_paginated(
                f"{base_url}/organization/usage/completions",
                headers,
                {"start_time": start_ts, "end_time": now_ts, "limit": 100},
                30,
            )
            if u_err is None and usage_pages is not None:
                req_total = 0
                in_total = 0
                out_total = 0
                for p in usage_pages:
                    reqs, in_tok, out_tok = _sum_usage_fields(p)
                    req_total += reqs
                    in_total += in_tok
                    out_total += out_tok
                out["actual_requests"] = req_total
                out["actual_input_tokens"] = in_total
                out["actual_output_tokens"] = out_total
            else:
                out["usage_error"] = f"http_{u_err.status_code}"
                try:
                    out["usage_error_detail"] = u_err.json()
                except Exception:
                    out["usage_error_detail"] = (u_err.text or "")[:300]
        except Exception as e:
            out["usage_error"] = str(e)

        actual_usage_cache["ts"] = time.time()
        actual_usage_cache["key"] = cache_key
        actual_usage_cache["data"] = out
        return jsonify(out)

    # --- IDE / File System APIs ---

    @app.get("/api/fs/tree")
    def fs_tree():
        session_id = request.args.get("session_id")
        if not session_id: return jsonify({"error": "missing_session_id"}), 400
        
        sess = Session.query.get(session_id)
        if not sess or not sess.workspace_path: return jsonify({"error": "session_not_found"}), 404
        
        root_path = sess.workspace_path
        if not os.path.exists(root_path): return jsonify([])

        file_tree = []
        # Simple recursive walk or just use os.walk and build structure
        # Let's build a flat list of paths for simplicity or a nested structure
        # Flattened list with type is easier for some UI, nested for Tree View.
        # Let's return a list of nodes with {id, parent, text, type}
        
        def build_tree(directory):
            tree = []
            for entry in os.scandir(directory):
                if entry.name.startswith(".") and entry.name != ".env": continue # Skip hidden except .env
                if entry.name == "node_modules" or entry.name == "__pycache__" or entry.name == "venv": continue
                
                node = {
                    "id": entry.path, # Use full path as ID for simplicity
                    "text": entry.name,
                    "parent": directory if directory != root_path else "#",
                    "type": "dir" if entry.is_dir() else "file"
                }
                if entry.is_dir():
                    node["children"] = build_tree(entry.path)
                tree.append(node)
            return sorted(tree, key=lambda k: (k['type'] != 'dir', k['text']))

        return jsonify(build_tree(root_path))

    @app.get("/api/fs/content")
    def fs_get_content():
        session_id = request.args.get("session_id")
        path = request.args.get("path") # Full absolute path from ID
        if not session_id or not path: return jsonify({"error": "missing_args"}), 400

        # Security check: ensure path is within session workspace
        sess = Session.query.get(session_id)
        if not sess or not sess.workspace_path: return jsonify({"error": "session_not_found"}), 404
        
        if not _is_within_workspace(sess.workspace_path, path):
             return jsonify({"error": "forbidden_path"}), 403
        
        if not os.path.exists(path) or not os.path.isfile(path):
             return jsonify({"error": "file_not_found"}), 404

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return jsonify({"content": content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/fs/content")
    def fs_save_content():
        payload = request.get_json(force=True)
        session_id = payload.get("session_id")
        path = payload.get("path")
        content = payload.get("content")
        
        if not session_id or not path: return jsonify({"error": "missing_args"}), 400
        
        sess = Session.query.get(session_id)
        if not sess or not sess.workspace_path: return jsonify({"error": "session_not_found"}), 404
        
        if not _is_within_workspace(sess.workspace_path, path):
             return jsonify({"error": "forbidden_path"}), 403

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return jsonify({"status": "saved"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/chat/refine")
    def chat_refine():
        payload = request.get_json(force=True)
        session_id = payload.get("session_id")
        path = payload.get("path") # Context file path
        message = payload.get("message")
        
        if not session_id or not message: return jsonify({"error": "missing_args"}), 400
        
        # Create a new task for this refinement
        # We assume the orchestrator's context-aware planning will pick up the current state
        # We just need to phrase the prompt well.
        
        rel_path = os.path.basename(path) if path else "project"
        description = f"Refine {rel_path}: {message}"
        
        t = Task(session_id=session_id, description=description, status="queued", progress=0.0, last_output=None)
        db.session.add(t)
        db.session.commit()
        
        orchestrator.start_task(t.id)
        
        return jsonify(t.to_dict())

    @app.get("/api/images/<path:filename>")
    def serve_generated_image(filename: str):
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            return jsonify({"error": "invalid_filename"}), 400
        full = os.path.join(settings.generated_images_dir, safe_name)
        if not os.path.isfile(full):
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(settings.generated_images_dir, safe_name)

    @app.get("/api/story/<task_id>")
    def serve_story(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "task_not_found"}), 404
        try:
            snap = json.loads(task.context_snapshot) if task.context_snapshot else {}
        except Exception:
            snap = {}
        slides = (snap or {}).get("story_slides")
        if isinstance(slides, list) and slides:
            story_title = str((snap or {}).get("story_title") or "Story").strip() or "Story"
            html = orchestrator._build_story_html(task.id, story_title, slides)
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}
        html_path = (snap or {}).get("story_artifact_path")
        if not html_path or not os.path.isfile(html_path):
            return jsonify({"error": "story_artifact_not_found"}), 404
        workspace = task.session.workspace_path or ""
        if not _is_within_workspace(workspace, html_path):
            return jsonify({"error": "forbidden"}), 403
        return send_file(html_path, mimetype="text/html; charset=utf-8")

    @app.post("/api/story/<task_id>/slides/<int:slide_index>/regenerate-image")
    def regenerate_story_slide_image(task_id: str, slide_index: int):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "task_not_found"}), 404
        try:
            snap = json.loads(task.context_snapshot) if task.context_snapshot else {}
        except Exception:
            snap = {}
        if not isinstance(snap, dict):
            snap = {}

        slides = snap.get("story_slides") or []
        if not isinstance(slides, list) or not slides:
            return jsonify({"error": "story_slides_not_found"}), 400
        if slide_index < 0 or slide_index >= len(slides):
            return jsonify({"error": "invalid_slide_index"}), 400

        slide = slides[slide_index] if isinstance(slides[slide_index], dict) else {}
        opts = (snap.get("story_options") or {}) if isinstance(snap.get("story_options"), dict) else {}
        story_character_bible = str(snap.get("story_character_bible") or "").strip()
        style_key = str(opts.get("illustration_style") or "ghibli").strip().lower()
        style_text, style_preset = orchestrator._story_style_prompt_and_preset(style_key)
        stability = orchestrator.llm.config.get("stability_settings") or {}
        api_key = (
            security.get_api_key("stability_api_key")
            or os.getenv("LASE_STABILITY_API_KEY")
            or stability.get("api_key")
        )
        if not api_key:
            return jsonify({"error": "stability_api_key_not_configured"}), 400

        raw_prompt = (
            str(slide.get("image_prompt") or "").strip()
            or str(slide.get("narration") or "").strip()
            or str(task.description or "").strip()
        )
        style_augmented = orchestrator._build_story_illustration_prompt(
            raw_prompt,
            style_key,
            character_bible=story_character_bible,
            chapter_idx=slide_index + 1,
        )
        refined_prompt, refine_source = orchestrator._refine_image_prompt(style_augmented)

        def _gen(prompt: str):
            return orchestrator.toolbus.image_generate(
                prompt,
                negative_prompt=orchestrator._story_negative_prompt(),
                aspect_ratio="16:9",
                style_preset=style_preset or stability.get("default_style_preset", "") or None,
                output_format=stability.get("default_output_format", "png"),
                api_key=api_key,
                base_url=stability.get("base_url", "https://api.stability.ai"),
                timeout_s=int(stability.get("timeout", 120)),
            )

        res = _gen(refined_prompt)
        if not getattr(res, "ok", False):
            fallback_prompt = (
                str(slide.get("caption") or "").strip()
                or str(slide.get("title") or "").strip()
                or str(slide.get("narration") or "").strip()
                or raw_prompt
            )
            fallback_prompt = (
                f"{fallback_prompt}, {style_text}, "
                + (f"character bible: {story_character_bible}, " if story_character_bible else "")
                + "single coherent scene, natural anatomy, "
                "no extra limbs/fingers/tails, no text or watermark"
            )[:2000]
            res = _gen(fallback_prompt)

        if not getattr(res, "ok", False):
            return jsonify({"error": "image_regeneration_failed", "detail": str(getattr(res, "output", ""))}), 500

        image_path = (res.meta or {}).get("image_path", "")
        image_url = orchestrator._to_image_preview_url(image_path)
        if not image_url:
            return jsonify({"error": "invalid_generated_image"}), 500

        slide["image_url"] = image_url
        slides[slide_index] = slide
        snap["story_slides"] = slides
        task.context_snapshot = json.dumps(snap)
        db.session.commit()

        # Best-effort persist artifact file for compatibility.
        story_title = str(snap.get("story_title") or "Story").strip() or "Story"
        html_path = str(snap.get("story_artifact_path") or "").strip()
        if html_path:
            workspace = task.session.workspace_path or ""
            if _is_within_workspace(workspace, html_path):
                try:
                    html = orchestrator._build_story_html(task.id, story_title, slides)
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                except Exception:
                    pass

        try:
            orchestrator._log(
                session_id=task.session_id,
                task_id=task.id,
                event_type="story_slide_regenerated",
                data={
                    "slide_index": slide_index,
                    "image_url": image_url,
                    "prompt": refined_prompt,
                    "prompt_refinement": refine_source,
                    "character_bible": story_character_bible,
                },
            )
        except Exception:
            pass

        return jsonify(
            {
                "status": "ok",
                "slide_index": slide_index,
                "image_url": image_url,
                "slide": slide,
            }
        )

    @app.post("/api/story/translate")
    def translate_story_narrations():
        payload = request.get_json(force=True, silent=True) or {}
        language = str(payload.get("language") or "").strip() or "en-US"
        narrations = payload.get("narrations") or []
        if not isinstance(narrations, list):
            return jsonify({"error": "invalid_narrations"}), 400

        # Bound payload to keep runtime translation lightweight.
        source = [str(x or "")[:4000] for x in narrations[:20]]
        if not source:
            return jsonify({"language": language, "narrations": []})

        lang_low = language.lower()
        if lang_low.startswith("en"):
            return jsonify({"language": language, "narrations": source})

        model_info = orchestrator.llm.get_model_for_task("general")
        prompt = (
            "Translate each item in the input JSON array to the requested language.\n"
            "Rules:\n"
            "- Preserve array length and order exactly.\n"
            "- Return STRICT JSON array of strings only.\n"
            "- Do not include markdown, commentary, or code fences.\n"
            f"Target language: {language}\n"
            f"Input: {json.dumps(source, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": "You are a precise translation engine."},
            {"role": "user", "content": prompt},
        ]
        out = orchestrator.llm.call_model(model_info, messages) or ""
        raw = str(out).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()

        translated = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                translated = [str(x or "") for x in parsed]
        except Exception:
            translated = None

        if not translated:
            translated = source
        if len(translated) < len(source):
            translated = translated + source[len(translated):]
        translated = translated[:len(source)]
        return jsonify({"language": language, "narrations": translated})

    @app.post("/api/story/tts")
    def synthesize_story_tts():
        payload = request.get_json(force=True, silent=True) or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "missing_text"}), 400
        text = text[:1500]
        selected_voice_name = str(payload.get("selected_voice_name") or "").strip()
        selected_voice_lang = str(payload.get("selected_voice_lang") or payload.get("language") or "").strip()
        speech_rate_raw = payload.get("speech_rate", 1.0)
        try:
            speech_rate = max(0.5, min(2.0, float(speech_rate_raw)))
        except Exception:
            speech_rate = 1.0

        ffmpeg_bin = shutil.which("ffmpeg")
        ffprobe_bin = shutil.which("ffprobe")
        if not ffmpeg_bin or not ffprobe_bin:
            return jsonify({"error": "ffmpeg_not_available"}), 500

        def _pick_flite_voice(voice_name: str, voice_lang: str) -> str:
            nm = str(voice_name or "").lower()
            lang = str(voice_lang or "").lower()
            female_tokens = ["female", "woman", "girl", "zira", "aria", "samantha", "victoria", "siri female"]
            male_tokens = ["male", "man", "boy", "david", "mark", "guy", "daniel", "siri male", "alex"]
            if any(t in nm for t in female_tokens):
                return "slt"
            if any(t in nm for t in male_tokens):
                return "kal"
            if lang.startswith("en-gb"):
                return "rms"
            if lang.startswith("en"):
                return "slt"
            return "slt"

        flite_voice = _pick_flite_voice(selected_voice_name, selected_voice_lang)
        with tempfile.TemporaryDirectory(prefix="story_tts_", dir="/tmp") as td:
            text_file = os.path.join(td, "tts.txt")
            wav_path = os.path.join(td, "tts.wav")
            wav2_path = os.path.join(td, "tts_rate.wav")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)

            cp = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"flite=textfile={text_file}:voice={flite_voice}",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    wav_path,
                ],
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                return jsonify({"error": "tts_generation_failed", "detail": (cp.stderr or "")[-400:]}), 500

            rate_cp = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    wav_path,
                    "-af",
                    f"atempo={speech_rate:.3f}",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    wav2_path,
                ],
                capture_output=True,
                text=True,
            )
            out_path = wav2_path if rate_cp.returncode == 0 and os.path.isfile(wav2_path) else wav_path
            if not os.path.isfile(out_path):
                return jsonify({"error": "tts_output_missing"}), 500

            probe = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    out_path,
                ],
                capture_output=True,
                text=True,
            )
            try:
                duration_s = max(0.1, float((probe.stdout or "0.1").strip()))
            except Exception:
                duration_s = 0.1

            with open(out_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")
        return jsonify(
            {
                "status": "ok",
                "voice_used": flite_voice,
                "duration_s": duration_s,
                "audio_base64": audio_b64,
                "format": "wav",
            }
        )

    @app.get("/api/story/<task_id>/bgm")
    def get_story_bgm(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "task_not_found"}), 404
        try:
            snap = json.loads(task.context_snapshot) if task.context_snapshot else {}
        except Exception:
            snap = {}
        slides = (snap or {}).get("story_slides") or []
        narrations = [
            str((slides[i].get("narration") if i < len(slides) and isinstance(slides[i], dict) else "") or "")
            for i in range(len(slides))
        ]
        mood = _infer_story_mood(narrations)
        track_path = _pick_bgm_track(task_id, mood)
        if not track_path:
            return jsonify({"status": "fallback", "mood": mood, "source": "synth", "track_url": None})
        return jsonify(
            {
                "status": "ok",
                "mood": mood,
                "source": "library",
                "track_url": _to_bgm_url(track_path),
                "track_name": os.path.basename(track_path),
            }
        )

    @app.post("/api/story/<task_id>/video")
    def generate_story_video(task_id: str):
        task = Task.query.get(task_id)
        if not task:
            return jsonify({"error": "task_not_found"}), 404

        try:
            snap = json.loads(task.context_snapshot) if task.context_snapshot else {}
        except Exception:
            snap = {}
        slides = (snap or {}).get("story_slides") or []
        if not isinstance(slides, list) or not slides:
            return jsonify({"error": "story_slides_not_found"}), 400

        payload = request.get_json(force=True, silent=True) or {}
        narrations_in = payload.get("narrations") or []
        if narrations_in and not isinstance(narrations_in, list):
            return jsonify({"error": "invalid_narrations"}), 400
        selected_voice_name = str(payload.get("selected_voice_name") or "").strip()
        selected_voice_lang = str(payload.get("selected_voice_lang") or payload.get("language") or "").strip()
        speech_rate_raw = payload.get("speech_rate", 1.0)
        try:
            speech_rate = max(0.5, min(2.0, float(speech_rate_raw)))
        except Exception:
            speech_rate = 1.0
        auto_music = bool(payload.get("auto_music", True))
        narrations = [
            str((narrations_in[i] if i < len(narrations_in) else ((slides[i] or {}).get("narration") or "")) or "")[:700]
            for i in range(len(slides))
        ]

        ffmpeg_bin = shutil.which("ffmpeg")
        ffprobe_bin = shutil.which("ffprobe")
        if not ffmpeg_bin or not ffprobe_bin:
            return jsonify({"error": "ffmpeg_not_available"}), 500

        def _resolve_image_path(image_url: str) -> str | None:
            u = str(image_url or "").strip()
            if not u:
                return None
            if u.startswith("/api/images/"):
                base = os.path.basename(u)
                p = os.path.join(settings.generated_images_dir, base)
                return p if os.path.isfile(p) else None
            if os.path.isabs(u) and os.path.isfile(u):
                if _is_within_workspace(settings.generated_images_dir, u):
                    return u
            return None

        def _ffmpeg_escape_text(text: str) -> str:
            t = str(text or "")
            t = t.replace("\\", "\\\\")
            t = t.replace(":", "\\:")
            t = t.replace("'", "\\'")
            t = t.replace("%", "\\%")
            t = t.replace("\n", " ")
            return t.strip()[:180]

        def _ffmpeg_escape_filter_value(value: str) -> str:
            v = str(value or "")
            v = v.replace("\\", "\\\\")
            v = v.replace(":", "\\:")
            v = v.replace("'", "\\'")
            v = v.replace(",", "\\,")
            return v

        def _pick_flite_voice(voice_name: str, voice_lang: str) -> str:
            # FFmpeg flite commonly exposes: slt (female), kal (male), awb (male), rms (male).
            nm = str(voice_name or "").lower()
            lang = str(voice_lang or "").lower()
            female_tokens = ["female", "woman", "girl", "zira", "aria", "samantha", "victoria", "siri female"]
            male_tokens = ["male", "man", "boy", "david", "mark", "guy", "daniel", "siri male", "alex"]
            if any(t in nm for t in female_tokens):
                return "slt"
            if any(t in nm for t in male_tokens):
                return "kal"
            if lang.startswith("en-gb"):
                return "rms"
            if lang.startswith("en"):
                return "slt"
            # Non-English text is still synthesized with available flite voice.
            return "slt"

        def _infer_story_mood(lines: list[str]) -> str:
            blob = " ".join(lines).lower()
            if any(k in blob for k in ["fear", "dark", "haunted", "mystery", "shadow", "secret", "nightmare"]):
                return "mystery"
            if any(k in blob for k in ["battle", "quest", "adventure", "dragon", "hero", "journey", "epic"]):
                return "adventure"
            if any(k in blob for k in ["happy", "joy", "love", "friend", "magic", "dream", "peace"]):
                return "uplifting"
            return "calm"

        def _bgm_expr_for_mood(mood: str) -> str:
            m = str(mood or "calm").lower()
            if m == "mystery":
                return (
                    "0.11*(0.55+0.45*sin(2*PI*2*t))*sin(2*PI*220*t)+"
                    "0.07*(0.50+0.50*sin(2*PI*2*t+0.8))*sin(2*PI*261.63*t)+"
                    "0.05*(0.45+0.55*sin(2*PI*4*t))*sin(2*PI*329.63*t)"
                )
            if m == "adventure":
                return (
                    "0.12*(0.55+0.45*sin(2*PI*2*t))*sin(2*PI*261.63*t)+"
                    "0.08*(0.50+0.50*sin(2*PI*2*t+0.6))*sin(2*PI*329.63*t)+"
                    "0.06*(0.50+0.50*sin(2*PI*4*t))*sin(2*PI*392*t)"
                )
            if m == "uplifting":
                return (
                    "0.11*(0.55+0.45*sin(2*PI*2*t))*sin(2*PI*293.66*t)+"
                    "0.08*(0.50+0.50*sin(2*PI*2*t+0.9))*sin(2*PI*369.99*t)+"
                    "0.05*(0.50+0.50*sin(2*PI*4*t))*sin(2*PI*440*t)"
                )
            return (
                "0.10*(0.55+0.45*sin(2*PI*2*t))*sin(2*PI*220*t)+"
                "0.07*(0.50+0.50*sin(2*PI*2*t+0.7))*sin(2*PI*277.18*t)+"
                "0.05*(0.45+0.55*sin(2*PI*4*t))*sin(2*PI*329.63*t)"
            )

        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        draw_font = f"fontfile={font_path}:" if os.path.isfile(font_path) else ""
        story_mood = _infer_story_mood(narrations)
        transition_s = 2.0
        flite_voice = _pick_flite_voice(selected_voice_name, selected_voice_lang)

        with tempfile.TemporaryDirectory(prefix="story_video_", dir="/tmp") as td:
            segment_paths: list[str] = []
            segment_durations: list[float] = []
            for i, slide in enumerate(slides):
                slide_obj = slide if isinstance(slide, dict) else {}
                text = narrations[i] or f"Chapter {i + 1}"
                text_file = os.path.join(td, f"text_{i}.txt")
                subtitle_file = os.path.join(td, f"subtitle_{i}.txt")
                audio_path = os.path.join(td, f"audio_{i}.wav")
                seg_path = os.path.join(td, f"seg_{i}.mp4")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                with open(subtitle_file, "w", encoding="utf-8") as f:
                    f.write(text)

                flite_filter = f"flite=textfile={text_file}:voice={flite_voice}"
                cp = subprocess.run(
                    [ffmpeg_bin, "-y", "-f", "lavfi", "-i", flite_filter, "-ar", "24000", "-ac", "1", audio_path],
                    capture_output=True,
                    text=True,
                )
                if cp.returncode != 0:
                    return jsonify({"error": "tts_generation_failed", "detail": (cp.stderr or "")[-400:]}), 500

                probe = subprocess.run(
                    [
                        ffprobe_bin,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        audio_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                try:
                    duration = max(1.0, float((probe.stdout or "1").strip()))
                except Exception:
                    duration = 2.0
                seg_duration = max(1.4, duration + transition_s)
                fade_out_start = max(0.2, seg_duration - transition_s)

                subtitle_file_escaped = _ffmpeg_escape_filter_value(subtitle_file)
                vf = (
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                    "drawbox=x=0:y=850:w=1920:h=230:color=black@0.55:t=fill,"
                    f"drawtext={draw_font}textfile='{subtitle_file_escaped}':fontcolor=white:fontsize=56:x=60:y=940:"
                    "line_spacing=6:reload=0:"
                    "fix_bounds=1:"
                    "text_shaping=1,"
                    "setsar=1,"
                    f"fade=t=in:st=0:d={transition_s:.3f},"
                    f"fade=t=out:st={fade_out_start:.3f}:d={transition_s:.3f}"
                )
                af = (
                    f"atempo={speech_rate:.3f},"
                    f"afade=t=in:st=0:d={transition_s:.3f},"
                    f"afade=t=out:st={fade_out_start:.3f}:d={transition_s:.3f}"
                )

                image_path = _resolve_image_path(slide_obj.get("image_url", ""))
                if image_path:
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-loop",
                        "1",
                        "-i",
                        image_path,
                        "-i",
                        audio_path,
                        "-t",
                        f"{seg_duration:.3f}",
                        "-vf",
                        vf,
                        "-af",
                        af,
                        "-r",
                        "30",
                        "-c:v",
                        "libx264",
                        "-aspect",
                        "16:9",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-shortest",
                        seg_path,
                    ]
                else:
                    fallback_vf = (
                        "drawbox=x=0:y=850:w=1920:h=230:color=black@0.55:t=fill,"
                        f"drawtext={draw_font}textfile='{subtitle_file_escaped}':fontcolor=white:fontsize=56:x=60:y=940:"
                        "line_spacing=6:reload=0:"
                        "fix_bounds=1:"
                        "text_shaping=1,"
                        "setsar=1"
                    )
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c=#111111:s=1920x1080:r=30:d={seg_duration:.3f}",
                        "-i",
                        audio_path,
                        "-vf",
                        f"{fallback_vf},fade=t=in:st=0:d={transition_s:.3f},fade=t=out:st={fade_out_start:.3f}:d={transition_s:.3f}",
                        "-af",
                        af,
                        "-c:v",
                        "libx264",
                        "-aspect",
                        "16:9",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-shortest",
                        seg_path,
                    ]

                seg_cp = subprocess.run(cmd, capture_output=True, text=True)
                if seg_cp.returncode != 0:
                    return jsonify({"error": "segment_video_failed", "detail": (seg_cp.stderr or "")[-500:]}), 500
                segment_paths.append(seg_path)
                segment_durations.append(float(seg_duration))

            if not segment_paths:
                return jsonify({"error": "no_segments_generated"}), 500

            os.makedirs(settings.generated_images_dir, exist_ok=True)
            out_name = f"story_{task_id}_{int(time.time())}.mp4"
            out_path = os.path.join(settings.generated_images_dir, out_name)
            if len(segment_paths) == 1:
                shutil.copy2(segment_paths[0], out_path)
            else:
                cmd = [ffmpeg_bin, "-y"]
                for p in segment_paths:
                    cmd.extend(["-i", p])

                fc_parts: list[str] = []
                for i in range(len(segment_paths)):
                    fc_parts.append(f"[{i}:v]settb=AVTB,format=yuv420p[v{i}]")
                    fc_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")

                v_prev = "v0"
                a_prev = "a_concat"
                cumulative = max(0.1, float(segment_durations[0]))
                for i in range(1, len(segment_paths)):
                    v_out = f"vx{i}"
                    offset = max(0.0, cumulative - transition_s)
                    fc_parts.append(
                        f"[{v_prev}][v{i}]xfade=transition=fade:duration={transition_s:.3f}:offset={offset:.3f}[{v_out}]"
                    )
                    v_prev = v_out
                    cumulative = cumulative + max(0.1, float(segment_durations[i])) - transition_s
                concat_audio_inputs = "".join([f"[a{i}]" for i in range(len(segment_paths))])
                fc_parts.append(f"{concat_audio_inputs}concat=n={len(segment_paths)}:v=0:a=1[{a_prev}]")

                cmd.extend(
                    [
                        "-filter_complex",
                        ";".join(fc_parts),
                        "-map",
                        f"[{v_prev}]",
                        "-map",
                        f"[{a_prev}]",
                        "-c:v",
                        "libx264",
                        "-aspect",
                        "16:9",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-movflags",
                        "+faststart",
                        out_path,
                    ]
                )
                final_cp = subprocess.run(cmd, capture_output=True, text=True)
                if final_cp.returncode != 0:
                    return jsonify({"error": "video_transition_merge_failed", "detail": (final_cp.stderr or "")[-700:]}), 500

            bgm_status = "skipped"
            bgm_source = "none"
            bgm_track_name = ""
            if auto_music:
                probe_total = subprocess.run(
                    [
                        ffprobe_bin,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        out_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                try:
                    total_duration = max(1.0, float((probe_total.stdout or "1").strip()))
                except Exception:
                    total_duration = 1.0

                bgm_path = os.path.join(td, "bgm.wav")
                mix_path = os.path.join(td, "story_mix.mp4")
                track_path = _pick_bgm_track(task_id, story_mood)
                bgm_cp = None
                if track_path:
                    bgm_track_name = os.path.basename(track_path)
                    bgm_source = "library"
                    bgm_cp = subprocess.run(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-stream_loop",
                            "-1",
                            "-i",
                            track_path,
                            "-t",
                            f"{total_duration:.3f}",
                            "-af",
                            (
                                "aresample=24000,pan=mono|c0=0.5*c0+0.5*c1,"
                                f"afade=t=in:st=0:d={transition_s:.3f},"
                                f"afade=t=out:st={max(0.2, total_duration - transition_s):.3f}:d={transition_s:.3f},"
                                "volume=0.85"
                            ),
                            "-ar",
                            "24000",
                            "-ac",
                            "1",
                            bgm_path,
                        ],
                        capture_output=True,
                        text=True,
                    )
                if (not bgm_cp) or bgm_cp.returncode != 0:
                    # Fallback: tonal synth bed.
                    expr = _bgm_expr_for_mood(story_mood)
                    bgm_source = "synth"
                    bgm_track_name = ""
                    bgm_cp = subprocess.run(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-f",
                            "lavfi",
                            "-i",
                            f"aevalsrc={expr}:s=24000:d={total_duration:.3f}",
                            "-af",
                            (
                                "lowpass=f=2200,highpass=f=60,"
                                f"afade=t=in:st=0:d={transition_s:.3f},"
                                f"afade=t=out:st={max(0.2, total_duration - transition_s):.3f}:d={transition_s:.3f},"
                                "volume=1.1"
                            ),
                            bgm_path,
                        ],
                        capture_output=True,
                        text=True,
                    )

                if bgm_cp.returncode == 0 and os.path.isfile(bgm_path):
                    mix_cp = subprocess.run(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-i",
                            out_path,
                            "-i",
                            bgm_path,
                            "-filter_complex",
                            "[0:a]volume=0.92[a0];[1:a]volume=0.65[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                            "-map",
                            "0:v:0",
                            "-map",
                            "[a]",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            "20",
                            "-pix_fmt",
                            "yuv420p",
                            "-c:a",
                            "aac",
                            "-shortest",
                            mix_path,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if mix_cp.returncode == 0 and os.path.isfile(mix_path):
                        os.replace(mix_path, out_path)
                        bgm_status = "added"
                    else:
                        bgm_status = "mix_failed"
                else:
                    bgm_status = "generation_failed"
        return jsonify(
            {
                "status": "ok",
                "download_url": f"/api/videos/{out_name}",
                "format": "mp4",
                "mood": story_mood,
                "background_music": bgm_status,
                "background_music_source": bgm_source,
                "background_music_track": bgm_track_name,
                "transition": "fade",
                "voice_used": flite_voice,
            }
        )

    @app.get("/api/videos/<path:filename>")
    def serve_generated_video(filename: str):
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            return jsonify({"error": "invalid_filename"}), 400
        full = os.path.join(settings.generated_images_dir, safe_name)
        if not os.path.isfile(full):
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(settings.generated_images_dir, safe_name)

    @app.get("/api/bgm/<path:filename>")
    def serve_bgm_track(filename: str):
        rel = str(filename or "").strip().replace("\\", "/")
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            return jsonify({"error": "invalid_filename"}), 400
        full = os.path.realpath(os.path.join(bgm_library_dir, rel))
        bgm_real = os.path.realpath(bgm_library_dir)
        if os.path.commonpath([bgm_real, full]) != bgm_real:
            return jsonify({"error": "forbidden"}), 403
        if not os.path.isfile(full):
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(os.path.dirname(full), os.path.basename(full))

    # Static frontend serving (prod build copied into src/static)
    @app.get("/")
    def spa_root():
        return _serve_spa(app)

    @app.get("/<path:path>")
    def spa_any(path: str):
        if path.startswith("api/"):
            return jsonify({"error": "not_found"}), 404
        static_dir = app.static_folder or ""
        full = os.path.join(static_dir, path)
        if static_dir and os.path.isfile(full):
            return send_from_directory(static_dir, path)
        return _serve_spa(app)

    # --- SocketIO Setup ---
    from flask_socketio import SocketIO, emit
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    def stream_logs():
        """Background thread to push logs to clients."""
        last_log_id = 0
        while True:
            socketio.sleep(0.5)
            try:
                # Access the toolbus via the orchestrator/app context
                # Since orchestrator is local to create_app, we rely on closure
                if hasattr(orchestrator, 'toolbus'):
                    pm = orchestrator.toolbus.process_manager
                    pids = pm.get_active_pids()
                    for pid in pids:
                        lines = pm.get_logs(pid)
                        if lines:
                            socketio.emit('process_log', {'pid': pid, 'lines': lines})
                with app.app_context():
                    rows = (
                        LogEntry.query.filter(LogEntry.id > last_log_id)
                        .order_by(LogEntry.id.asc())
                        .limit(200)
                        .all()
                    )
                    for row in rows:
                        payload = row.data()
                        detail = payload.get("output") or payload.get("error") or payload.get("plan") or json.dumps(payload)
                        detail = str(detail).replace("\n", " ")
                        if len(detail) > 360:
                            detail = detail[:360] + "... [TRUNCATED]"
                        task_tag = (row.task_id or "session")[:8]
                        line = f"[{row.event_type}] {detail}\n"
                        socketio.emit("process_log", {"pid": f"task:{task_tag}", "lines": [line]})
                        last_log_id = row.id
            except Exception:
                # Streaming failures should not break the app loop.
                continue

    watchdog_retries: dict[str, int] = {}

    def task_watchdog():
        """Recover tasks marked running when no active worker is alive."""
        stale_no_worker_s = int(os.getenv("LASE_STALE_NO_WORKER_SECONDS", "15"))
        while True:
            socketio.sleep(5)
            try:
                with app.app_context():
                    now = datetime.now(timezone.utc)
                    running = Task.query.filter_by(status="running").all()
                    for task in running:
                        if orchestrator.is_task_active(task.id):
                            continue
                        age_s = (now - task.updated_at).total_seconds() if task.updated_at else 0
                        if age_s < stale_no_worker_s:
                            continue

                        retries = watchdog_retries.get(task.id, 0)
                        if retries < 1:
                            watchdog_retries[task.id] = retries + 1
                            task.status = "queued"
                            task.status_detail = "watchdog_auto_retry"
                            task.failure_reason = None
                            db.session.commit()
                            le = LogEntry(
                                session_id=task.session_id,
                                task_id=task.id,
                                event_type="task_auto_retry",
                                data_json=json.dumps({"reason": "stale_running_no_worker", "retry": retries + 1}),
                            )
                            db.session.add(le)
                            db.session.commit()
                            orchestrator.start_task(task.id)
                        else:
                            task.status = "failed"
                            task.status_detail = "stalled_no_worker"
                            task.failure_reason = "Task became stale (running without active worker thread)."
                            db.session.commit()
                            le = LogEntry(
                                session_id=task.session_id,
                                task_id=task.id,
                                event_type="task_failed",
                                data_json=json.dumps({"error": task.failure_reason}),
                            )
                            db.session.add(le)
                            db.session.commit()
            except Exception:
                continue

    # Start the background task
    socketio.start_background_task(stream_logs)
    socketio.start_background_task(task_watchdog)

    @socketio.on('connect')
    def handle_connect():
        emit("connected", {"ok": True})

    return app, socketio


if __name__ == "__main__":
    # We need to load settings first to pass to create_app if needed, 
    # but create_app loads them internally.
    # However, to be consistent with previous logic:
    app, socketio = create_app()
    project_root = str(Path(__file__).resolve().parents[1])
    settings = load_settings(project_root)
    
    print(f" * Running on http://127.0.0.1:{settings.port} (SocketIO enabled)")
    socketio.run(app, host=settings.host, port=settings.port, debug=settings.debug, allow_unsafe_werkzeug=True)
