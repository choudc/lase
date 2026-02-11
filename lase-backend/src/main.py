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
from datetime import datetime, timezone
import requests




def create_app() -> Flask:
    project_root = str(Path(__file__).resolve().parents[1])  # lase-backend/
    settings = load_settings(project_root)

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
                
                # Force reload in adapter? Adapter reloads on init usually.
                # Ideally Orchestrator's LLMAdapter should reload.
                # For now, simplistic approach: adapter reads from file or we re-init.
                # To be proper: orchestrator.llm.reload_config()
                if hasattr(orchestrator, "llm"):
                     orchestrator.llm.config = current_config # weak reload

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
        html_path = (snap or {}).get("story_artifact_path")
        if not html_path or not os.path.isfile(html_path):
            return jsonify({"error": "story_artifact_not_found"}), 404
        workspace = task.session.workspace_path or ""
        if not _is_within_workspace(workspace, html_path):
            return jsonify({"error": "forbidden"}), 403
        return send_file(html_path, mimetype="text/html; charset=utf-8")

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
