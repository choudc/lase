import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass

from ..db import db
from ..models.session import LogEntry, Session, Task
from .llm_adapter import LLMAdapter
from .natural_language_interface import classify_intent
from . import security
from .toolbus import ToolBus


@dataclass(frozen=True)
class Step:
    name: str
    tool: str | None = None
    args: dict | None = None


class AgentOrchestrator:
    """
    Runs a lightweight Plan -> Execute -> Observe loop for tasks.
    This implementation is intentionally small but preserves the design intent:
    - progress updates
    - last_output capture
    - event logging ("progress_update" etc.)
    """

    def __init__(self, *, app, workspaces_dir: str, generated_images_dir: str, models_yaml_path: str):
        self.app = app
        self.workspaces_dir = workspaces_dir
        self.generated_images_dir = generated_images_dir
        self.toolbus = ToolBus(workspaces_dir=workspaces_dir, generated_images_dir=generated_images_dir)
        self.llm = LLMAdapter(models_yaml_path)
        self._threads: dict[str, threading.Thread] = {}
        self._stop_requested: set[str] = set()

    def start_task(self, task_id: str) -> bool:
        if task_id in self._threads and self._threads[task_id].is_alive():
            return False

        t = threading.Thread(target=self._run_task, args=(task_id,), daemon=True)
        self._threads[task_id] = t
        t.start()
        return True

    def is_task_active(self, task_id: str) -> bool:
        t = self._threads.get(task_id)
        return bool(t and t.is_alive())

    def request_stop(self, task_id: str) -> None:
        self._stop_requested.add(task_id)

    def _log(self, *, session_id: str, task_id: str | None, event_type: str, data: dict) -> None:
        le = LogEntry(session_id=session_id, task_id=task_id, event_type=event_type, data_json=json.dumps(data))
        db.session.add(le)
        db.session.commit()

    def _json_default(self, obj):
        # Allow dataclasses/objects like ModelInfo inside snapshots.
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    def _update_task(self, task: Task, *, status: str | None = None, progress: float | None = None, last_output: str | None = None, status_detail: str | None = None, execution_log: list[dict] | None = None, context_snapshot: dict | None = None, failure_reason: str | None = None) -> None:
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = max(0.0, min(1.0, float(progress)))
        if last_output is not None:
            task.last_output = last_output
        if status_detail is not None:
            task.status_detail = status_detail
        if execution_log is not None:
            task.execution_log = json.dumps(execution_log, default=self._json_default)
        if context_snapshot is not None:
            task.context_snapshot = json.dumps(context_snapshot, default=self._json_default)
        if failure_reason is not None:
            task.failure_reason = failure_reason
        db.session.commit()

    def _ensure_workspace(self, sess: Session) -> str:
        if sess.workspace_path and os.path.isdir(sess.workspace_path):
            return sess.workspace_path
        os.makedirs(self.workspaces_dir, exist_ok=True)
        ws = os.path.join(self.workspaces_dir, sess.id)
        os.makedirs(ws, exist_ok=True)
        sess.workspace_path = ws
        db.session.commit()
        return ws

    def _refine_image_prompt(self, raw_prompt: str) -> tuple[str, str]:
        """
        Refine a short user request into a richer image-generation prompt.
        Falls back safely when no chat model is available.
        """
        prompt = (raw_prompt or "").strip()
        if not prompt:
            return prompt, "none"
        model_info = self.llm.get_model_for_task("reasoning")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer for image generation. "
                    "Rewrite the user prompt to be vivid, specific, and visually coherent. "
                    "Include scene composition, lighting, mood, style hints, and quality cues. "
                    "Return only a single prompt string with no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            refined = (self.llm.call_model(model_info, messages) or "").strip()
        except Exception:
            refined = ""
        if not refined or refined.lower().startswith("error") or "unsupported chat provider" in refined.lower():
            # Deterministic fallback enhancement when LLM refinement is unavailable.
            fallback = (
                f"{prompt}, serene composition, natural lighting, cinematic atmosphere, "
                "high detail, balanced colors, depth of field, ultra quality"
            )
            return fallback[:2000], "fallback"
        refined = refined.strip().strip('"').strip("'")
        if refined.lower() == prompt.lower():
            refined = (
                f"{prompt}, tranquil setting, soft ambient light, realistic textures, "
                "cohesive composition, premium quality"
            )
            return refined[:2000], "fallback"
        return refined[:2000] if refined else prompt, "llm"

    def _to_image_preview_url(self, image_path: str) -> str | None:
        try:
            if not image_path:
                return None
            if not os.path.commonpath([os.path.realpath(self.generated_images_dir), os.path.realpath(image_path)]) == os.path.realpath(self.generated_images_dir):
                return None
            return f"/api/images/{os.path.basename(image_path)}"
        except Exception:
            return None

    def _to_web_preview_url(self, result) -> str | None:
        try:
            meta = getattr(result, "meta", {}) if hasattr(result, "meta") else {}
            port = meta.get("port")
            if port:
                return f"http://localhost:{int(port)}"
            output = getattr(result, "output", "") if hasattr(result, "output") else ""
            m = re.search(r"\bhttps?://[^\s]+", str(output))
            if m:
                return m.group(0)
        except Exception:
            pass
        return None

    def _extract_plan_steps(self, plan_text: str) -> list[str]:
        steps: list[str] = []
        for line in (plan_text or "").splitlines():
            m = re.match(r"^\s*(\d+)\.\s+(.*)$", line.strip())
            if m:
                step_name = m.group(2).strip()
                if step_name:
                    steps.append(step_name)
        return steps

    def _extract_completed_step_index(self, response_text: str) -> int | None:
        if not response_text:
            return None
        m = re.search(r"\[COMPLETED\]\s*Step\s*(\d+)", response_text, re.IGNORECASE)
        if not m:
            return None
        try:
            return max(0, int(m.group(1)) - 1)
        except Exception:
            return None

    def _needs_user_decision(self, response_text: str) -> bool:
        text = (response_text or "").strip().lower()
        if not text:
            return False
        if len(text) > 4000:
            return False
        patterns = [
            "need approval",
            "need your approval",
            "need permission",
            "please confirm",
            "do you want me to",
            "would you like me to",
            "i can proceed if you approve",
            "shall i proceed",
            "counter proposal",
            "suggest further improvement",
            "suggestion:",
            "proposed improvement",
        ]
        return any(p in text for p in patterns)

    def _build_plan(self, task: Task) -> list[Step]:
        intent = classify_intent(task.description)
        # We perform a dynamic loop, so the "plan" is just a high-level intent declaration for the UI.
        return [
            Step(name=f"Classify intent: {intent.task_type} (confidence {intent.confidence:.2f})"),
            Step(name="Architect: Analyse & Propose Features"),
            Step(name="Dynamic Execution Loop"),
            Step(name="Finalize"),
        ]

    def _get_workspace_structure(self, workspace_path: str) -> str:
        """
        Returns a simplified file tree string of the workspace.
        """
        if not os.path.exists(workspace_path):
            return ""
        
        structure = []
        ignore_dirs = {".git", "node_modules", "__pycache__", "venv", ".next", "dist", "build", ".expo"}
        
        for root, dirs, files in os.walk(workspace_path):
            # Modify dirs in-place to skip ignored
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            level = root.replace(workspace_path, "").count(os.sep)
            indent = "  " * level
            base = os.path.basename(root)
            if base:
                structure.append(f"{indent}- {base}/")
            
            subindent = "  " * (level + 1)
            for f in files:
                structure.append(f"{subindent}- {f}")
                
        return "\n".join(structure)

    def _generate_plan(self, task_description: str, model_info, workspace_structure: str = "") -> str:
        context_str = ""
        if workspace_structure:
            context_str = f"""
# CURRENT PROJECT STRUCTURE
{workspace_structure}

NOTE: The user likely wants to MODIFY or IMPROVE this existing project. 
Refactor or edit existing files instead of creating new ones unless necessary.
"""

        prompt = f"""You are an expert software architect.
The user has requested: "{task_description}"
{context_str}

Your goal is to:
1. Analyze the request.
2. Propose a set of key features/functionalities (Scope).
3. Create a detailed, step-by-step implementation plan.

Output ONLY the plan in the following format:

# Feature Proposal
- Feature 1
- Feature 2...

# Implementation Plan
1. [Setup] ...
2. [Backend] ...
3. [Frontend] ...
...
"""
        messages = [{"role": "user", "content": prompt}]
        # Use a distinct call to getting the plan
        return self.llm.call_model(model_info, messages)

    def _run_task(self, task_id: str) -> None:
        ctx = self.app.app_context()
        ctx.push()
        try:
            task = Task.query.get(task_id)
            if not task:
                return
            sess = Session.query.get(task.session_id)
            if not sess:
                return

            workspace = self._ensure_workspace(sess)
            
            # Initialize messages from execution_log if resuming
            messages = []
            snapshot = {}
            if task.execution_log:
                try:
                    messages = json.loads(task.execution_log)
                    self._log(session_id=sess.id, task_id=task.id, event_type="task_resumed", data={"from_log": True})
                except json.JSONDecodeError:
                    self._log(session_id=sess.id, task_id=task.id, event_type="error", data={"error": "Failed to decode execution_log, starting fresh."})
                    messages = []
            if task.context_snapshot:
                try:
                    snapshot = json.loads(task.context_snapshot) or {}
                except json.JSONDecodeError:
                    snapshot = {}

            resumed_from_log = bool(messages)
            
            # New task (or resumed task that continues via latest state below)
            self._update_task(
                task,
                status="running",
                progress=task.progress if resumed_from_log else 0.0,
                last_output=task.last_output if resumed_from_log else None,
                status_detail="resumed" if resumed_from_log else None,
                failure_reason=None,
            )
            self._log(session_id=sess.id, task_id=task.id, event_type="task_started", data={"workspace": workspace, "resumed": resumed_from_log})
            if not resumed_from_log:

                # Identify model
                intent = classify_intent(task.description)

                # Fast path: handle image generation directly to avoid requiring a planner/chat model.
                if intent.task_type == "image_generation":
                    if task.id in self._stop_requested:
                        self._stop_requested.discard(task.id)
                        self._update_task(task, status="stopped", progress=task.progress, status_detail="paused_by_user")
                        self._log(session_id=sess.id, task_id=task.id, event_type="task_stopped", data={"reason": "pause_requested"})
                        return
                    image_plan_steps = [
                        "Refine prompt for image quality",
                        "Generate image with Stability AI",
                        "Finalize output and preview link",
                    ]
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_initialized",
                        data={"steps": image_plan_steps},
                    )
                    stability = self.llm.config.get("stability_settings") or {}
                    api_key = (
                        security.get_api_key("stability_api_key")
                        or os.getenv("LASE_STABILITY_API_KEY")
                        or stability.get("api_key")
                    )
                    refined_prompt, refine_source = self._refine_image_prompt(task.description)
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_step_update",
                        data={"index": 0, "status": "completed", "step": image_plan_steps[0]},
                    )
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="prompt_refined",
                        data={"original": task.description, "refined": refined_prompt, "source": refine_source},
                    )
                    result = self.toolbus.image_generate(
                        refined_prompt,
                        aspect_ratio=stability.get("default_aspect_ratio", "1:1"),
                        style_preset=stability.get("default_style_preset", "") or None,
                        output_format=stability.get("default_output_format", "png"),
                        api_key=api_key,
                        base_url=stability.get("base_url", "https://api.stability.ai"),
                        timeout_s=int(stability.get("timeout", 120)),
                    )
                    output = result.output if hasattr(result, "output") else str(result)
                    ok = bool(getattr(result, "ok", False))
                    meta = getattr(result, "meta", {}) if hasattr(result, "meta") else {}
                    preview_url = self._to_image_preview_url(meta.get("image_path", ""))
                    if ok and preview_url:
                        output = f"{output}\nPreview URL: {preview_url}\nPrompt used: {refined_prompt}\nPrompt refinement: {refine_source}"
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="tool_result",
                        data={"tool": "image.generate", "output": output, "preview_url": preview_url, "prompt_used": refined_prompt, "prompt_refinement": refine_source},
                    )
                    if ok:
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="plan_step_update",
                            data={"index": 1, "status": "completed", "step": image_plan_steps[1]},
                        )
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="plan_step_update",
                            data={"index": 2, "status": "completed", "step": image_plan_steps[2]},
                        )
                        self._update_task(
                            task,
                            status="completed",
                            progress=1.0,
                            status_detail="image_generated",
                            last_output=output,
                        )
                        self._log(session_id=sess.id, task_id=task.id, event_type="task_completed", data={})
                    else:
                        self._update_task(
                            task,
                            status="failed",
                            progress=1.0,
                            status_detail="image_generation_failed",
                            failure_reason=output,
                            last_output=output,
                        )
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="plan_step_update",
                            data={"index": 1, "status": "error", "step": image_plan_steps[1], "error": output},
                        )
                        self._log(session_id=sess.id, task_id=task.id, event_type="task_failed", data={"error": output})
                    return

                model_info = self.llm.get_model_for_task(intent.task_type)

                # --- Generative Planning Phase ---
                self._log(session_id=sess.id, task_id=task.id, event_type="thinking", data={"phase": "planning"})
                
                # Get workspace context
                ws_structure = self._get_workspace_structure(workspace)
                
                plan = self._generate_plan(task.description, model_info, ws_structure)
                self._log(session_id=sess.id, task_id=task.id, event_type="plan_generated", data={"plan": plan})
                plan_steps = self._extract_plan_steps(plan)
                if plan_steps:
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_initialized",
                        data={"steps": plan_steps},
                    )
                
                # Show plan to user
                self._update_task(task, last_output=f"**Architect Plan**:\n{plan}\n\nStarting execution...")

                # Prepare tools
                tools = self.toolbus.list_tools()
                tools_desc = json.dumps(tools, indent=2)

                # System Prompt
                system_prompt = f"""You are an autonomous AI agent running in a local environment.
Workspace Directory: {workspace}
You have access to the following tools:
{tools_desc}

Your goal is to complete the user's task by strictly following the plan below.

# EXECUTION PLAN
{plan}

You must think step-by-step.
When you complete a step from the plan, output a message starting with: "[COMPLETED] Step X: ..."

To use a tool, you MUST output a JSON block strictly following this format:
```json
{{
  "action": "tool_name",
  "args": {{ ... }}
}}
```
Examples:
```json
{{
  "action": "shell.run",
  "args": {{ "command": "ls -la" }}
}}
```
```json
{{
  "action": "fs.write",
  "args": {{ "path": "hello.txt", "content": "Hello world" }}
}}
```

If you are done, output ONLY the word "DONE".
If you need to define a plan, just state your plan in text before acting.
If a tool execution fails, you receive a FAILURE message. You MUST analyze the error and attempt to fix it in the next step.
Do NOT output DONE if the last step failed.

# CAPABILITIES & SAFETY
- **Web**: Use `web.search` to find documentation or solutions. Use `web.read` to read pages.
- **Dependencies**: You can install dependencies using `shell.run`.
  - Python: `pip install <package>` (or `venv/bin/pip`)
  - Node: `npm install <package>`
- **Risky Actions**: `fs.delete` is available but RISKY. 
  - If you need to delete a file/folder or perform a destructive action, verify it is safe.
  - If it is a major deletion (e.g. entire folder), ASK THE USER for confirmation in a new task first using the `notify_user` concept (by just stating "I need approval to delete X"). 
  - Actually, since you are autonomous, use your best judgement. If unsure, stop and ask.
"""
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task.description}"}
                ]
            
            # Identify model (even if resuming, we need current model settings)
            intent = classify_intent(task.description) # Re-classify intent as task.description is the source of truth
            model_info = self.llm.get_model_for_task(intent.task_type)
            tools = self.toolbus.list_tools() # Tools might have changed

            max_steps = 15
            current_plan_step_idx = 0
            step_retry_count: dict[int, int] = {}
            start_step = 0
            if resumed_from_log:
                try:
                    start_step = int(snapshot.get("current_step", -1)) + 1
                except Exception:
                    start_step = 0
                if start_step < 0:
                    start_step = 0
                if start_step >= max_steps:
                    start_step = max_steps - 1

            for i in range(start_step, max_steps):
                if task.id in self._stop_requested:
                    self._stop_requested.discard(task.id)
                    self._update_task(task, status="stopped", progress=task.progress, status_detail="paused_by_user", execution_log=messages, context_snapshot={"current_step": i, "model_info": model_info, "tools": tools})
                    self._log(session_id=sess.id, task_id=task.id, event_type="task_stopped", data={"reason": "pause_requested"})
                    return
                # 1. Update progress roughly
                progress = (i + 1) / max_steps
                self._update_task(task, progress=progress)

                # 2. Call LLM
                try:
                    self._log(session_id=sess.id, task_id=task.id, event_type="thinking", data={"step": i+1})
                    response_text = self.llm.call_model(model_info, messages, tools)

                    # Hard-stop on provider/API errors so the task doesn't falsely progress.
                    if isinstance(response_text, str) and (
                        response_text.startswith("Error calling OpenAI:")
                        or response_text.startswith("Error from OpenAI:")
                        or response_text.startswith("Error calling")
                    ):
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="error",
                            data={"error": response_text, "provider": model_info.provider},
                        )
                        self._update_task(
                            task,
                            status="failed",
                            status_detail="llm_provider_error",
                            failure_reason=response_text,
                            last_output=response_text,
                            execution_log=messages,
                            context_snapshot={"current_step": i, "model_info": model_info, "tools": tools},
                        )
                        return
                    
                    # Save execution log and context snapshot after each LLM call
                    self._update_task(task, last_output=response_text, execution_log=messages, context_snapshot={"current_step": i, "model_info": model_info, "tools": tools})
                    
                    # Add assistant response to history
                    messages.append({"role": "assistant", "content": response_text})
                    completed_idx = self._extract_completed_step_index(response_text)
                    if completed_idx is not None:
                        current_plan_step_idx = max(current_plan_step_idx, completed_idx + 1)
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="plan_step_update",
                            data={"index": completed_idx, "status": "completed"},
                        )

                    # 3. Check for DONE
                    if "DONE" in response_text and len(response_text) < 20:
                        break
                    
                    # 4. Parse Tool Call
                    tool_call = self._parse_tool_call(response_text)
                    
                    if tool_call:
                        tool_name = tool_call.get("action")
                        tool_args = tool_call.get("args") or {}
                        
                        self._log(session_id=sess.id, task_id=task.id, event_type="tool_execution", 
                                  data={"tool": tool_name, "args": tool_args})
                        
                        # Execute
                        result = None
                        if tool_name == "shell.run":
                            result = self.toolbus.shell_run(tool_args.get("command", ""), cwd=workspace)
                        elif tool_name == "fs.read":
                            path = tool_args.get("path", "")
                            if not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.fs_read(path)
                        elif tool_name == "fs.write":
                            path = tool_args.get("path", "")
                            if not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.fs_write(path, tool_args.get("content", ""))
                        elif tool_name == "fs.ls":
                            path = tool_args.get("path", "")
                            if not path or path == ".":
                                path = workspace
                            elif not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.fs_ls(path)
                        elif tool_name == "fs.mkdir":
                            path = tool_args.get("path", "")
                            if not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.fs_mkdir(path)
                        elif tool_name == "project.init":
                            name = tool_args.get("name", "")
                            ptype = tool_args.get("type", "web")
                            result = self.toolbus.project_init(ptype, name, workspace)
                        elif tool_name == "project.run":
                            path = tool_args.get("path", "")
                            ptype = tool_args.get("type", "web")
                            port = int(tool_args.get("port", 3000))
                            if not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.project_run(ptype, path, port)
                        elif tool_name == "project.stop":
                            pid = tool_args.get("pid", "")
                            result = self.toolbus.project_stop(pid)
                        elif tool_name == "project.build":
                            ptype = tool_args.get("type", "android")
                            path = tool_args.get("path", "")
                            platform = tool_args.get("platform", "android")
                            if not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.project_build(ptype, path, platform)
                        elif tool_name == "vision.analyze":
                            image_path = tool_args.get("path", "")
                            prompt = tool_args.get("prompt", "Analyze this image.")
                            if not os.path.isabs(image_path):
                                image_path = os.path.join(workspace, image_path)
                            
                            if not os.path.exists(image_path):
                                result = {"ok": False, "output": f"Image not found at {image_path}", "meta": {}}
                            else:
                                try:
                                    # 1. Get Vision Model
                                    vision_model = self.llm.get_model_for_task("vision")
                                    
                                    # 2. Read Image
                                    with open(image_path, "rb") as f:
                                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                                    
                                    # 3. Construct Message
                                    # Detect mime type roughly
                                    mime = "image/png"
                                    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
                                        mime = "image/jpeg"
                                        
                                    vision_messages = [
                                        {"role": "user", "content": [
                                            {"type": "text", "text": prompt},
                                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                                        ]}
                                    ]
                                    
                                    # 4. Call Model
                                    # We don't pass tools to vision model usually, just get text back
                                    analysis = self.llm.call_model(vision_model, vision_messages)
                                    result = {"ok": True, "output": f"Vision Analysis:\n{analysis}", "meta": {}}
                                except Exception as e:
                                    result = {"ok": False, "output": f"Vision analysis failed: {str(e)}", "meta": {}}

                        elif tool_name == "web.search":
                            result = self.toolbus.web_search(tool_args.get("query", ""))
                        elif tool_name == "web.read":
                            result = self.toolbus.web_read(tool_args.get("url", ""))
                        elif tool_name == "browser.screenshot":
                            result = self.toolbus.browser_screenshot(tool_args.get("url", ""))
                        elif tool_name == "fs.delete":
                            path = tool_args.get("path", "")
                            if not os.path.isabs(path):
                                path = os.path.join(workspace, path)
                            result = self.toolbus.fs_delete(path)
                        elif tool_name == "image.generate":
                            stability = self.llm.config.get("stability_settings") or {}
                            raw_prompt = tool_args.get("prompt", "")
                            refined_prompt, refine_source = self._refine_image_prompt(raw_prompt)
                            api_key = security.get_api_key("stability_api_key") or os.getenv("LASE_STABILITY_API_KEY") or stability.get("api_key")
                            result = self.toolbus.image_generate(
                                refined_prompt,
                                negative_prompt=tool_args.get("negative_prompt", ""),
                                aspect_ratio=tool_args.get("aspect_ratio", stability.get("default_aspect_ratio", "1:1")),
                                seed=int(tool_args.get("seed", 0) or 0),
                                style_preset=tool_args.get("style_preset", stability.get("default_style_preset", "")) or None,
                                output_format=tool_args.get("output_format", stability.get("default_output_format", "png")),
                                api_key=api_key,
                                base_url=stability.get("base_url", "https://api.stability.ai"),
                                timeout_s=int(stability.get("timeout", 120)),
                            )
                            self._log(
                                session_id=sess.id,
                                task_id=task.id,
                                event_type="prompt_refined",
                                data={"original": raw_prompt, "refined": refined_prompt, "source": refine_source},
                            )
                        else:
                            result = {"ok": False, "output": f"Unknown tool: {tool_name}", "meta": {}}

                        # Format result
                        # Check success status
                        is_ok = True
                        output_str = ""
                        
                        if isinstance(result, dict):
                            is_ok = result.get("ok", False)
                            output_str = result.get("output", str(result))
                        elif hasattr(result, "ok") and hasattr(result, "output"):
                             is_ok = result.ok
                             output_str = result.output
                             if tool_name == "image.generate":
                                 preview_url = self._to_image_preview_url((result.meta or {}).get("image_path", ""))
                                 if preview_url:
                                     output_str = f"{output_str}\nPreview URL: {preview_url}"
                             if tool_name == "project.run":
                                 preview_url = self._to_web_preview_url(result)
                                 if preview_url:
                                     output_str = f"{output_str}\nPreview URL: {preview_url}"
                        else:
                             output_str = str(result)

                        # Add tool result to history
                        if not is_ok:
                             self._log(
                                 session_id=sess.id,
                                 task_id=task.id,
                                 event_type="plan_step_update",
                                 data={"index": current_plan_step_idx, "status": "error", "error": output_str},
                             )
                             retries = step_retry_count.get(current_plan_step_idx, 0)
                             messages.append({"role": "user", "content": f"Tool Execution FAILED.\nOutput: {output_str}\n\nYou must analyze this error and attempt to fix it in the next step."})
                             if retries < 1:
                                 step_retry_count[current_plan_step_idx] = retries + 1
                                 messages.append({
                                     "role": "user",
                                     "content": (
                                         "Before moving to any next plan step, retry and fix this failed step now. "
                                         "Use tools as needed and only proceed once the step succeeds."
                                     ),
                                 })
                                 self._update_task(
                                     task,
                                     status="running",
                                     status_detail="recovering_failed_step",
                                     failure_reason=None,
                                     execution_log=messages,
                                     context_snapshot={"current_step": i, "model_info": model_info, "tools": tools},
                                 )
                                 continue
                             # Save execution log and context snapshot after repeated tool failure
                             self._update_task(task, status="failed", status_detail="tool_execution_failed", failure_reason=output_str, execution_log=messages, context_snapshot={"current_step": i, "model_info": model_info, "tools": tools})
                             return
                        else:
                             messages.append({"role": "user", "content": f"Tool Output: {output_str}"})
                             # Save execution log and context snapshot after successful tool execution
                             if current_plan_step_idx in step_retry_count:
                                 self._log(
                                     session_id=sess.id,
                                     task_id=task.id,
                                     event_type="plan_step_update",
                                     data={"index": current_plan_step_idx, "status": "completed_after_retry"},
                                 )
                                 del step_retry_count[current_plan_step_idx]
                             self._update_task(task, last_output=output_str, execution_log=messages, context_snapshot={"current_step": i, "model_info": model_info, "tools": tools})
                        
                        self._log(session_id=sess.id, task_id=task.id, event_type="tool_result", 
                                    data={"output": output_str})
                    else:
                        # Human-in-the-loop: pause when model asks permission or proposes optional improvements.
                        if self._needs_user_decision(response_text):
                            self._update_task(
                                task,
                                status="stopped",
                                status_detail="awaiting_user_decision",
                                last_output=response_text,
                                execution_log=messages,
                                context_snapshot={
                                    "current_step": i,
                                    "model_info": model_info,
                                    "tools": tools,
                                    "awaiting_user_decision": True,
                                },
                            )
                            self._log(
                                session_id=sess.id,
                                task_id=task.id,
                                event_type="user_decision_requested",
                                data={"message": response_text},
                            )
                            return

                except Exception as e:
                    error_message = str(e)
                    self._log(session_id=sess.id, task_id=task.id, event_type="error", data={"error": error_message})
                    
                    # Save execution log and context snapshot before failing
                    self._update_task(task, status="failed", progress=1.0, last_output=f"Task Failed: {error_message}", status_detail="llm_execution_error", failure_reason=error_message, execution_log=messages, context_snapshot={"current_step": i, "model_info": model_info, "tools": tools})
                    return

            self._update_task(task, status="completed", progress=1.0, status_detail="success", execution_log=messages, context_snapshot={"final_step": max_steps, "model_info": model_info, "tools": tools})
            self._log(session_id=sess.id, task_id=task.id, event_type="task_completed", data={})
        finally:
            ctx.pop()
            self._stop_requested.discard(task_id)
            if task_id in self._threads and not self._threads[task_id].is_alive():
                self._threads.pop(task_id, None)

    def _parse_tool_call(self, text: str) -> dict | None:
        """
        Extracts JSON block from text.
        """
        try:
            # Look for ```json ... ```
            import re
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Fallback: look for just { ... } if it looks like the only thing
            clean = text.strip()
            if clean.startswith("{") and clean.endswith("}"):
                return json.loads(clean)
        except Exception:
            pass
        return None
