import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote

from ..db import db
from ..models.session import LogEntry, Session, Task
from .llm_adapter import LLMAdapter
from .natural_language_interface import Intent, classify_intent
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

    def _trim_text(self, text: str, max_len: int = 2000) -> str:
        s = str(text or "").strip()
        if len(s) <= max_len:
            return s
        return s[:max_len] + "... [TRUNCATED]"

    def _latest_user_message(self, messages: list[dict]) -> str:
        for m in reversed(messages or []):
            if str(m.get("role") or "") != "user":
                continue
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(str(p.get("text") or ""))
                return "\n".join(parts).strip()
            return str(content or "").strip()
        return ""

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
            # Preserve stable metadata across snapshot updates unless explicitly overwritten.
            try:
                existing = json.loads(task.context_snapshot) if task.context_snapshot else {}
                if isinstance(existing, dict) and isinstance(context_snapshot, dict):
                    merged = dict(existing)
                    merged.update(context_snapshot)
                    context_snapshot = merged
            except Exception:
                pass
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
                    "Keep the output tightly aligned to the exact scenario/situation in the input. "
                    "Include subject identity/count, action, environment, camera framing, lighting, mood, style hints, and quality cues. "
                    "Do not introduce unrelated objects/characters/events. "
                    "Enforce anatomically correct results (no extra limbs/fingers/tails, natural hands/faces). "
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

    def _task_category(self, task: Task) -> str | None:
        if getattr(task, "category", None):
            return str(task.category).strip().lower()
        try:
            if not task.context_snapshot:
                return None
            snap = json.loads(task.context_snapshot)
            if isinstance(snap, dict):
                v = str(snap.get("category") or "").strip().lower()
                # Backfill persisted category for legacy tasks.
                if v and not getattr(task, "category", None):
                    task.category = v
                    db.session.commit()
                return v or None
        except Exception:
            return None
        return None

    def _story_options(self, task: Task) -> dict:
        try:
            if not task.context_snapshot:
                return {
                    "generate_illustrations": True,
                    "illustration_count_mode": "auto",
                    "illustration_count": 10,
                    "illustration_style": "ghibli",
                    "target_minutes": 5,
                }
            snap = json.loads(task.context_snapshot)
            if not isinstance(snap, dict):
                return {
                    "generate_illustrations": True,
                    "illustration_count_mode": "auto",
                    "illustration_count": 10,
                    "illustration_style": "ghibli",
                    "target_minutes": 5,
                }
            opts = snap.get("story_options") or {}
            if not isinstance(opts, dict):
                opts = {}
            count_mode = str(opts.get("illustration_count_mode", "auto") or "auto").strip().lower()
            if count_mode not in {"auto", "manual"}:
                count_mode = "auto"
            style = str(opts.get("illustration_style", "ghibli") or "ghibli").strip().lower()
            if style not in {"ghibli", "storybook", "anime", "cinematic", "fantasy", "watercolor", "photorealistic"}:
                style = "ghibli"
            target_minutes = int(opts.get("target_minutes", 5) or 5)
            return {
                "generate_illustrations": bool(opts.get("generate_illustrations", True)),
                "illustration_count_mode": count_mode,
                "illustration_count": max(0, min(10, int(opts.get("illustration_count", 10) or 10))),
                "illustration_style": style,
                "target_minutes": max(3, min(30, target_minutes)),
            }
        except Exception:
            return {
                "generate_illustrations": True,
                "illustration_count_mode": "auto",
                "illustration_count": 10,
                "illustration_style": "ghibli",
                "target_minutes": 5,
            }

    def _auto_story_illustration_count(self, slides: list[dict]) -> int:
        if not slides:
            return 0
        # Ensure each chapter/slide gets at least one illustration target.
        return len(slides)

    def _story_style_prompt_and_preset(self, style: str) -> tuple[str, str | None]:
        s = str(style or "ghibli").strip().lower()
        mapping: dict[str, tuple[str, str | None]] = {
            "ghibli": (
                "Studio Ghibli-inspired hand-painted animation style, warm magical atmosphere, expressive characters, whimsical but grounded scenery",
                "anime",
            ),
            "storybook": (
                "classic children storybook illustration, painterly brush strokes, charming composition, soft color palette",
                "digital-art",
            ),
            "anime": (
                "high-quality anime illustration, clean linework, expressive characters, vivid but balanced colors",
                "anime",
            ),
            "cinematic": (
                "cinematic concept art, dramatic lighting, strong depth and framing, visually coherent scene",
                "cinematic",
            ),
            "fantasy": (
                "fantasy illustration art, epic environment, enchanted mood, rich atmospheric detail",
                "fantasy-art",
            ),
            "watercolor": (
                "watercolor illustration style, soft edges, paper texture, elegant color blending",
                None,
            ),
            "photorealistic": (
                "photorealistic scene, natural lighting, physically plausible materials, highly detailed rendering",
                "photographic",
            ),
        }
        return mapping.get(s, mapping["ghibli"])

    def _build_story_illustration_prompt(
        self,
        raw_prompt: str,
        style: str,
        character_bible: str | None = None,
        chapter_idx: int | None = None,
    ) -> str:
        base = (raw_prompt or "").strip()
        style_text, _ = self._story_style_prompt_and_preset(style)
        chapter_hint = ""
        if chapter_idx and int(chapter_idx) > 0:
            chapter_hint = f"chapter {int(chapter_idx)} scene only, "
        character_block = ""
        if character_bible:
            cb = str(character_bible).strip()
            character_block = (
                "character bible for continuity (must remain consistent across chapters): "
                f"{cb}. keep face, hair, body type, age impression, outfit signature, and colors consistent unless explicitly changed by the chapter narration, "
            )
        constraints = (
            f"single coherent {chapter_hint}explicit subject count, "
            "clear action-state and emotion matching the situation, "
            "consistent character identity/outfit across chapters, anatomically correct humans/animals, "
            "natural face and hands, two hands only per person, five fingers per hand, no extra limbs or tails, "
            "consistent perspective and scale, clear subject focus, high detail, clean edges, "
            "no text, no logo, no watermark, no unrelated elements"
        )
        return f"{base}, {style_text}, {character_block}{constraints}"[:2000]

    def _derive_story_character_bible(self, task_description: str, slides: list[dict], model_info) -> str:
        narrations = [
            str((s or {}).get("narration") or "").strip()
            for s in (slides or [])
            if isinstance(s, dict) and str((s or {}).get("narration") or "").strip()
        ][:24]
        if not narrations:
            return ""

        prompt = (
            "Extract recurring character continuity notes from the story.\n"
            "Return STRICT JSON array only. Each item keys:\n"
            "name, role, visual_traits, outfit_signature, color_palette.\n"
            "Rules:\n"
            "- Include only recurring characters (appearing across multiple chapters) when possible.\n"
            "- Keep each field concise (5-18 words).\n"
            "- Max 5 characters.\n"
            "- If no recurring characters are identifiable, return [].\n"
            f"Story request: {task_description}\n"
            f"Narration chapters: {json.dumps(narrations, ensure_ascii=False)}"
        )
        raw = self.llm.call_model(model_info, [{"role": "user", "content": prompt}]) or ""
        raw = str(raw).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()

        try:
            data = json.loads(raw)
        except Exception:
            data = []
        if not isinstance(data, list):
            return ""

        lines = []
        for item in data[:5]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            role = str(item.get("role") or "").strip()
            traits = str(item.get("visual_traits") or "").strip()
            outfit = str(item.get("outfit_signature") or "").strip()
            palette = str(item.get("color_palette") or "").strip()
            if not (name and (traits or outfit)):
                continue
            line = (
                f"{name}"
                + (f" ({role})" if role else "")
                + (f": {traits}" if traits else "")
                + (f"; outfit {outfit}" if outfit else "")
                + (f"; colors {palette}" if palette else "")
            )
            lines.append(line)
        if not lines:
            return ""
        return " | ".join(lines)[:900]

    def _story_negative_prompt(self) -> str:
        return (
            "low quality, blurry, distorted anatomy, deformed body, malformed hands, bad face, crossed eyes, "
            "extra fingers, fused fingers, missing fingers, extra arms, extra legs, extra tails, extra heads, "
            "duplicate person, duplicate limbs, dislocated joints, twisted limbs, text, logo, watermark, "
            "cropped, out of frame, jpeg artifacts"
        )

    def _intent_for_task(self, task: Task) -> Intent:
        category = self._task_category(task)
        category_map = {
            "image": "image_generation",
            "website": "coding",
            "research": "reasoning",
            "story": "general",
            "android_app": "coding",
            "python_app": "coding",
        }
        if category in category_map:
            return Intent(task_type=category_map[category], confidence=0.99)
        return classify_intent(task.description)

    def _category_instruction(self, category: str | None) -> str:
        if category == "image":
            return "CATEGORY: IMAGE. Focus on high-quality prompt refinement and image generation output."
        if category == "website":
            return (
                "CATEGORY: WEBSITE. You must produce a runnable web app, create ./start.sh and ./deploy.sh in project root, "
                "launch the site, and provide Preview URL before finishing."
            )
        if category == "research":
            return "CATEGORY: RESEARCH. Prioritize web.search/web.read, structured findings, and source-backed conclusions."
        if category == "story":
            return (
                "CATEGORY: STORY. Produce coherent creative writing with clear structure and polished narrative. "
                "If story options request illustrations, generate them and provide a slideshow preview."
            )
        if category == "android_app":
            return "CATEGORY: ANDROID APP. Prioritize Android/Expo setup, run instructions, and build viability."
        if category == "python_app":
            return "CATEGORY: PYTHON APP. Produce runnable Python project with entrypoint and start script."
        return "CATEGORY: AUTO. Infer best approach from task description."

    def _is_website_task(self, description: str) -> bool:
        t = (description or "").lower()
        markers = ["website", "web app", "webapp", "frontend", "react", "vite", "landing page", "html/css"]
        return any(m in t for m in markers)

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

    def _generate_story_slides(self, task: Task, model_info, story_opts: dict | None = None) -> tuple[str, list[dict]]:
        def _sanitize_slide(item: dict, idx: int, fallback_text: str) -> dict:
            title = str(item.get("title") or f"Chapter {idx + 1}").strip() or f"Chapter {idx + 1}"
            caption = str(item.get("caption") or title).strip()
            narration = str(item.get("narration") or fallback_text).strip() or fallback_text
            image_prompt = str(item.get("image_prompt") or narration).strip() or narration
            return {
                "title": title,
                "caption": caption,
                "narration": narration,
                "image_prompt": image_prompt,
            }

        def _split_sentences_local(text: str) -> list[str]:
            t = str(text or "").strip()
            if not t:
                return []
            parts = re.split(r"(?<=[\.\!\?。！？])\s+", t)
            out = [p.strip() for p in parts if p and p.strip()]
            return out or [t]

        def _expand_to_target_count(slides_in: list[dict], target_count: int) -> list[dict]:
            slides_local = [_sanitize_slide(s if isinstance(s, dict) else {}, i, task.description or "Story") for i, s in enumerate(slides_in or [])]
            if not slides_local:
                slides_local = [
                    {
                        "title": "Chapter 1",
                        "caption": "Opening",
                        "narration": (task.description or "A story begins.").strip(),
                        "image_prompt": (task.description or "A story begins.").strip(),
                    }
                ]

            while len(slides_local) < target_count:
                # Split the longest narration first for natural expansion.
                best_idx = 0
                best_len = -1
                for i, s in enumerate(slides_local):
                    w = len(str(s.get("narration") or "").split())
                    if w > best_len:
                        best_idx = i
                        best_len = w
                base = slides_local[best_idx]
                sents = _split_sentences_local(base.get("narration", ""))

                if len(sents) >= 2:
                    mid = max(1, len(sents) // 2)
                    n1 = " ".join(sents[:mid]).strip()
                    n2 = " ".join(sents[mid:]).strip()
                else:
                    words = str(base.get("narration") or "").split()
                    if len(words) >= 16:
                        midw = max(6, len(words) // 2)
                        n1 = " ".join(words[:midw]).strip()
                        n2 = " ".join(words[midw:]).strip()
                    else:
                        n1 = str(base.get("narration") or "").strip()
                        n2 = f"Continuation: {n1}".strip()

                p1 = str(base.get("image_prompt") or n1).strip()
                p2 = str(base.get("image_prompt") or n2).strip()
                left = {
                    "title": str(base.get("title") or f"Chapter {best_idx + 1}").strip(),
                    "caption": str(base.get("caption") or "").strip() or "Part 1",
                    "narration": n1 or str(base.get("narration") or "").strip(),
                    "image_prompt": p1 or n1,
                }
                right = {
                    "title": f"{str(base.get('title') or f'Chapter {best_idx + 1}').strip()} (Cont.)",
                    "caption": "Continuation",
                    "narration": n2 or n1,
                    "image_prompt": p2 or n2 or n1,
                }
                slides_local = slides_local[:best_idx] + [left, right] + slides_local[best_idx + 1 :]

            return slides_local[:target_count]

        opts = story_opts or {}
        target_minutes = max(3, min(30, int(opts.get("target_minutes", 5) or 5)))
        target_words = int(target_minutes * 130)
        # Enforce pacing target: 1 chapter per 20 seconds => 3 chapters per minute.
        chapter_count = max(3, min(180, int(target_minutes * 3)))
        words_per_chapter = max(28, int(round(target_words / max(1, chapter_count))))
        prompt = (
            "Write a story based on the user prompt.\n"
            "Return STRICT JSON array only. Each item must include keys: "
            "title, caption, narration, image_prompt.\n"
            f"Target length: about {target_minutes} minutes total narration at normal speaking pace (~130 words/min).\n"
            f"Create exactly {chapter_count} chapters/slides (about 20 seconds per chapter).\n"
            f"Each narration should be around {words_per_chapter} words (about +/-20%).\n"
            "Keep cumulative narration length near target (+/-15%).\n"
            "For each chapter, image_prompt must describe one clear scene from that exact chapter.\n"
            "Include character count, key action, setting, time-of-day/lighting, camera framing.\n"
            "Do NOT add unrelated characters/objects/events not in that chapter.\n"
            "Avoid distortion by explicitly preferring natural anatomy and realistic hands/faces.\n"
            f"User prompt: {task.description}"
        )
        raw = self.llm.call_model(model_info, [{"role": "user", "content": prompt}]) or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()
        slides = []
        story_title = "Story"
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                if data.get("story_title"):
                    story_title = str(data.get("story_title") or "Story").strip() or "Story"
                maybe_slides = data.get("slides")
                if isinstance(maybe_slides, list):
                    data = maybe_slides
                else:
                    data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        slides.append(
                            {
                                "title": str(item.get("title") or "Story"),
                                "caption": str(item.get("caption") or ""),
                                "narration": str(item.get("narration") or ""),
                                "image_prompt": str(item.get("image_prompt") or ""),
                            }
                        )
        except Exception:
            slides = []

        if not slides:
            narration = (task.description or "A short story").strip()
            slides = [
                {
                    "title": "Opening",
                    "caption": "Story opening",
                    "narration": narration,
                    "image_prompt": narration,
                }
            ]

        # If model under-produces chapters, retry once with strict expansion prompt.
        if len(slides) < chapter_count:
            retry_prompt = (
                "Expand the following story slides.\n"
                f"Return STRICT JSON array with EXACTLY {chapter_count} items.\n"
                "Each item keys: title, caption, narration, image_prompt.\n"
                "Preserve continuity while splitting/expanding chapters.\n"
                f"Input slides: {json.dumps(slides, ensure_ascii=False)}"
            )
            retry_raw = self.llm.call_model(model_info, [{"role": "user", "content": retry_prompt}]) or ""
            retry_raw = retry_raw.strip()
            if retry_raw.startswith("```"):
                retry_raw = re.sub(r"^```[a-zA-Z]*\n", "", retry_raw).rstrip("`").strip()
            try:
                retry_data = json.loads(retry_raw)
                if isinstance(retry_data, list) and retry_data:
                    slides = retry_data
            except Exception:
                pass

        slides = _expand_to_target_count(slides, chapter_count)
        if slides and (not story_title or story_title == "Story"):
            first = str(slides[0].get("title") or "").strip()
            if first:
                story_title = first
        return story_title, slides[:chapter_count]

    def _build_story_youtube_metadata(self, title: str, slides: list[dict]) -> dict:
        story_title = str(title or "Story").strip() or "Story"
        narrations = [
            str((s or {}).get("narration") or "").strip()
            for s in (slides or [])
            if isinstance(s, dict)
        ]
        joined = " ".join(narrations).strip()
        summary = joined[:900].strip()
        if len(joined) > 900:
            summary += "..."

        words = re.findall(r"[A-Za-z0-9]+", story_title)
        derived_tags = [f"#{w[:24]}" for w in words[:4] if len(w) > 2]
        base_tags = ["#StoryTime", "#AIStory", "#Animation", "#AIGenerated", "#Narration"]
        hashtags = []
        for t in base_tags + derived_tags:
            if t not in hashtags:
                hashtags.append(t)
        hashtags = hashtags[:8]

        keyword_base = [
            "ai story",
            "animated story",
            "story narration",
            "children story",
            "bedtime story",
            "storybook video",
        ]
        keyword_derived = [w.lower() for w in words if len(w) > 2][:8]
        keywords = []
        for k in keyword_base + keyword_derived:
            if k not in keywords:
                keywords.append(k)

        yt_title = f"{story_title} | AI Animated Story"
        yt_description = (
            f"{story_title}\n\n"
            f"{summary}\n\n"
            "Generated with LASE (Local Autonomous Software Engineer).\n"
            "If you enjoyed this story, like, comment, and subscribe for more.\n\n"
            f"{' '.join(hashtags)}"
        ).strip()

        return {
            "title": yt_title[:100],
            "description": yt_description[:5000],
            "hashtags": " ".join(hashtags),
            "keywords": ", ".join(keywords[:20]),
        }

    def _build_story_html(self, task_id: str, title: str, slides: list[dict]) -> str:
        payload = json.dumps(slides)
        youtube_meta_payload = json.dumps(self._build_story_youtube_metadata(title, slides))
        safe_title = (title or "Story Slideshow").replace("<", "").replace(">", "")
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    body {{ margin:0; font-family:Arial,sans-serif; background:#111; color:#fff; }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:16px; }}
    .grid {{ display:grid; grid-template-columns:2fr 1fr; gap:14px; }}
    .card {{ background:#1b1b1b; border:1px solid #333; border-radius:12px; overflow:hidden; }}
    .image-wrap {{ position:relative; }}
    .img {{ width:100%; height:500px; object-fit:cover; background:#000; transition: opacity 1s ease; opacity: 1; }}
    .img-caption-tag {{
      position:absolute; left:12px; top:12px; padding:6px 10px;
      background:rgba(15,23,42,.75); border:1px solid rgba(148,163,184,.5);
      color:#dbeafe; border-radius:999px; font-size:12px;
    }}
    .img-caption {{
      position:absolute; left:0; right:0; bottom:0; padding:10px 14px;
      background:linear-gradient(transparent, rgba(0,0,0,.78));
      color:#f8fafc; font-size:28px; line-height:1.45; font-weight:700; text-align:center;
      text-shadow:0 2px 8px rgba(0,0,0,.55);
    }}
    .subtitle-word.active {{
      background: rgba(59, 130, 246, 0.45);
      border-radius: 6px;
      padding: 0 3px;
      box-shadow: 0 0 0 1px rgba(147, 197, 253, 0.35) inset;
    }}
    .txt {{ padding:16px; }}
    .controls {{ display:flex; gap:8px; padding:12px 16px 16px; flex-wrap:wrap; }}
    button {{ background:#2563eb; color:#fff; border:0; border-radius:8px; padding:8px 12px; cursor:pointer; }}
    button.secondary {{ background:#374151; }}
    select {{
      background:#111827; color:#e5e7eb; border:1px solid #374151;
      border-radius:8px; padding:8px 10px; min-width:180px;
    }}
    .script {{ max-height:690px; overflow:auto; padding:10px; }}
    .script h3 {{ margin:6px 0 10px; font-size:16px; }}
    .script ol {{ margin:0; padding-left:20px; }}
    .script li {{
      margin:0 0 10px;
      line-height:1.5;
      padding:8px;
      border-radius:8px;
      background:#151515;
      border:1px solid #2a2a2a;
    }}
    .script li.active {{
      background:#1e3a8a33;
      border-color:#3b82f6;
    }}
    .meta {{
      margin-top: 12px;
      border-top: 1px solid #2a2a2a;
      padding-top: 12px;
    }}
    .meta h4 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: #bfdbfe;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .meta-row {{ margin: 0 0 10px; }}
    .meta-label {{
      display:block;
      font-size:11px;
      color:#93c5fd;
      margin-bottom:4px;
      text-transform: uppercase;
      letter-spacing:.04em;
    }}
    .meta-input, .meta-textarea {{
      width:100%;
      background:#0f172a;
      color:#e2e8f0;
      border:1px solid #334155;
      border-radius:8px;
      padding:8px 10px;
      font-size:12px;
      font-family: Arial, sans-serif;
    }}
    .meta-textarea {{
      min-height:140px;
      resize:vertical;
      line-height:1.45;
    }}
    .meta-copy {{
      margin-top:6px;
      background:#1d4ed8;
      font-size:12px;
      padding:6px 10px;
    }}
    .small {{ color:#93c5fd; font-size:12px; margin-top:4px; }}
    .status-wrap {{ display:none; width:100%; max-width:520px; }}
    .status-text {{ color:#93c5fd; font-size:12px; margin-bottom:4px; }}
    .progress-outer {{
      width:100%;
      height:8px;
      border-radius:999px;
      background:#0f172a;
      border:1px solid #334155;
      overflow:hidden;
    }}
    .progress-inner {{
      height:100%;
      width:0%;
      background:linear-gradient(90deg,#2563eb,#38bdf8);
      transition:width .25s ease;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns:1fr; }}
      .img {{ height:360px; }}
      .script {{ max-height:300px; }}
      .img-caption {{ font-size:22px; }}
    }}
  </style>
</head>
  <body>
  <div class="wrap">
    <div class="grid">
      <div class="card">
        <div class="image-wrap">
          <img id="slideImage" class="img" alt="Illustration" />
          <div id="slideCaptionOverlay" class="img-caption"></div>
        </div>
        <div class="txt">
          <p id="slideNarration"></p>
        </div>
        <div class="controls">
          <button class="secondary" id="prevBtn" onclick="prev()">Prev</button>
          <button id="playBtn" onclick="playStory()">Play Story</button>
          <button id="readBtn" onclick="readCurrent()">Read Current</button>
          <button class="secondary" id="pauseBtn" onclick="pauseStory()">Pause</button>
          <button class="secondary" id="nextBtn" onclick="next()">Next</button>
          <button class="secondary" id="regenBtn" onclick="regenerateCurrentImage()">Regenerate Image</button>
          <button class="secondary" id="videoBtn" onclick="generateStoryVideo()">Generate Video</button>
          <select id="chapterSelect" title="Chapter"></select>
          <select id="voiceSelect" title="Narration voice"></select>
        </div>
        <div class="controls">
          <a id="videoDownload" style="display:none; color:#93c5fd;" download="story.mp4">Download Story Video</a>
          <div id="translationStatusWrap" class="status-wrap">
            <div id="translationStatus" class="status-text">Translating subtitles...</div>
            <div class="progress-outer"><div id="translationProgressBar" class="progress-inner"></div></div>
          </div>
          <div id="videoStatusWrap" class="status-wrap">
            <div id="videoStatus" class="status-text">Generating video...</div>
            <div class="progress-outer"><div id="videoProgressBar" class="progress-inner"></div></div>
          </div>
          <span id="regenStatus" style="color:#93c5fd; font-size:12px; display:none;">Regenerating image...</span>
        </div>
      </div>
      <div class="card script">
        <h3>Complete Script</h3>
        <ol id="scriptList"></ol>
        <div class="meta">
          <h4>YouTube Metadata</h4>
          <div class="meta-row">
            <label class="meta-label" for="ytTitle">Title</label>
            <input id="ytTitle" class="meta-input" readonly />
            <button class="meta-copy" onclick="copyText(document.getElementById('ytTitle').value)">Copy Title</button>
          </div>
          <div class="meta-row">
            <label class="meta-label" for="ytDescription">Description</label>
            <textarea id="ytDescription" class="meta-textarea" readonly></textarea>
            <button class="meta-copy" onclick="copyText(document.getElementById('ytDescription').value)">Copy Description</button>
          </div>
          <div class="meta-row">
            <label class="meta-label" for="ytHashtags">Hashtags</label>
            <input id="ytHashtags" class="meta-input" readonly />
            <button class="meta-copy" onclick="copyText(document.getElementById('ytHashtags').value)">Copy Hashtags</button>
          </div>
          <div class="meta-row">
            <label class="meta-label" for="ytKeywords">Keywords</label>
            <input id="ytKeywords" class="meta-input" readonly />
            <button class="meta-copy" onclick="copyText(document.getElementById('ytKeywords').value)">Copy Keywords</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const taskId = {json.dumps(task_id)};
    const slides = {payload};
    const youtubeMeta = {youtube_meta_payload};
    let idx = 0;
    let playing = false;
    let selectedVoiceName = 'Google UK English Female';
    let narrationRunId = 0;
    let translationRunId = 0;
    const CHAPTER_TRANSITION_MS = 2000;
    let translationPromise = Promise.resolve();
    let isTranslating = false;
    let bgmCtx = null;
    let bgmMaster = null;
    let bgmNodes = [];
    let bgmTrackAudio = null;
    let bgmScheduler = null;
    let bgmNextTime = 0;
    let bgmStep = 0;
    let activeAudioEl = null;
    let subtitleTicker = null;
    let videoProgressTicker = null;
    let storyPlaybackEndResolver = null;
    const sourceNarrations = slides.map((s) => String((s && s.narration) || '').trim());
    const narrationByLang = {{ source: sourceNarrations }};
    let activeLanguage = 'source';

    function getNarration(s) {{
      const i = slides.indexOf(s);
      if (i < 0) return String((s && s.narration) || '');
      const arr = narrationByLang[activeLanguage] || sourceNarrations;
      return String(arr[i] || sourceNarrations[i] || '');
    }}

    function getCurrentLanguageTag() {{
      if (activeLanguage && activeLanguage !== 'source') return String(activeLanguage);
      const v = getPreferredVoice();
      const lang = String((v && v.lang) || '').trim();
      return lang || 'en-US';
    }}

    function splitSentences(text) {{
      const t = String(text || '').trim();
      if (!t) return [''];
      const lang = getCurrentLanguageTag();
      if (typeof Intl !== 'undefined' && Intl.Segmenter) {{
        try {{
          const seg = new Intl.Segmenter(lang, {{ granularity: 'sentence' }});
          const rows = Array.from(seg.segment(t)).map((r) => String(r.segment || '').trim()).filter(Boolean);
          if (rows.length) return rows;
        }} catch (_) {{}}
      }}
      const parts = t.match(/[^.!?。！？！？؛،\\n]+[.!?。！？！？؛،]?/g) || [t];
      return parts.map((p) => p.trim()).filter(Boolean);
    }}

    function splitWords(sentence) {{
      const s = String(sentence || '');
      if (!s) return [];
      const lang = getCurrentLanguageTag();
      if (typeof Intl !== 'undefined' && Intl.Segmenter) {{
        try {{
          const seg = new Intl.Segmenter(lang, {{ granularity: 'word' }});
          const out = [];
          for (const r of seg.segment(s)) {{
            const token = String(r.segment || '');
            if (!token) continue;
            const isWordLike = r.isWordLike === undefined ? !!token.trim() : !!r.isWordLike;
            if (!isWordLike) continue;
            out.push({{ text: token, index: Number(r.index || 0) }});
          }}
          if (out.length) return out;
        }} catch (_) {{}}
      }}
      const out = [];
      const re = /\\S+/g;
      let m;
      while ((m = re.exec(s)) !== null) {{
        out.push({{ text: m[0], index: m.index }});
      }}
      return out;
    }}

    function escapeHtml(text) {{
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function setSubtitleSentence(text, sentenceIndex = 0, charIndexInSentence = -1) {{
      const parts = splitSentences(text);
      const idxSafe = Math.max(0, Math.min(parts.length - 1, sentenceIndex));
      const sentence = parts[idxSafe] || String(text || '').trim();
      const el = document.getElementById('slideCaptionOverlay');
      if (!el) return;

      const words = splitWords(sentence);
      if (!words.length || charIndexInSentence < 0) {{
        el.textContent = sentence;
        return;
      }}

      let activeIdx = 0;
      for (let i = 0; i < words.length; i += 1) {{
        const start = Number(words[i].index || 0);
        const nextStart = i + 1 < words.length ? Number(words[i + 1].index || sentence.length) : sentence.length + 1;
        if (charIndexInSentence >= start && charIndexInSentence < nextStart) {{
          activeIdx = i;
          break;
        }}
      }}

      let html = '';
      let cursor = 0;
      for (let i = 0; i < words.length; i += 1) {{
        const w = words[i];
        const start = Number(w.index || 0);
        const token = String(w.text || '');
        if (start > cursor) {{
          html += escapeHtml(sentence.slice(cursor, start));
        }}
        const cls = i === activeIdx ? 'subtitle-word active' : 'subtitle-word';
        html += `<span class="${{cls}}">${{escapeHtml(token)}}</span>`;
        cursor = start + token.length;
      }}
      if (cursor < sentence.length) {{
        html += escapeHtml(sentence.slice(cursor));
      }}
      el.innerHTML = html;
    }}

    function clearSubtitleTicker() {{
      if (subtitleTicker) {{
        clearInterval(subtitleTicker);
        subtitleTicker = null;
      }}
    }}

    function getVoices() {{
      if (!window.speechSynthesis) return [];
      return window.speechSynthesis.getVoices() || [];
    }}

    function chooseDefaultVoiceName(voices) {{
      const rows = Array.isArray(voices) ? voices : [];
      if (!rows.length) return '';
      // Hard default requested by product requirement.
      const exact = rows.find((v) => String(v?.name || '').trim() === 'Google UK English Female');
      if (exact) return String(exact.name);

      // Accept near matches in case browser labels differ slightly.
      const scored = rows
        .map((v) => {{
          const name = String(v?.name || '').toLowerCase();
          const lang = String(v?.lang || '').toLowerCase();
          let score = 0;
          if (name.includes('google')) score += 40;
          if (name.includes('uk') || name.includes('british')) score += 35;
          if (name.includes('female') || name.includes('woman')) score += 28;
          if (lang.startsWith('en-gb')) score += 40;
          if (lang.startsWith('en')) score += 8;
          return {{ v, score }};
        }})
        .sort((a, b) => b.score - a.score);
      if (scored.length && scored[0].score >= 45) return String(scored[0].v.name || '');
      return rows[0]?.name ? String(rows[0].name) : '';
    }}

    function scoreVoice(v) {{
      let score = 0;
      const name = String(v?.name || '').toLowerCase();
      const lang = String(v?.lang || '').toLowerCase();
      if (lang.startsWith('en')) score += 20;
      if (name.includes('neural')) score += 50;
      if (name.includes('natural')) score += 40;
      if (name.includes('enhanced')) score += 30;
      if (name.includes('premium')) score += 25;
      if (name.includes('siri')) score += 20;
      if (name.includes('google')) score += 18;
      if (name.includes('microsoft')) score += 14;
      if (v?.default) score += 10;
      return score;
    }}

    function getPreferredVoice() {{
      const voices = getVoices();
      if (!voices.length) return null;
      if (selectedVoiceName) {{
        const chosen = voices.find((v) => v.name === selectedVoiceName);
        if (chosen) return chosen;
      }}
      const preferred = chooseDefaultVoiceName(voices);
      if (preferred) {{
        const chosen = voices.find((v) => v.name === preferred);
        if (chosen) return chosen;
      }}
      return voices.slice().sort((a, b) => scoreVoice(b) - scoreVoice(a))[0] || voices[0];
    }}

    function renderVoiceSelect() {{
      const sel = document.getElementById('voiceSelect');
      if (!sel) return;
      const voices = getVoices();
      const previous = selectedVoiceName || chooseDefaultVoiceName(voices);
      sel.innerHTML = '';
      voices.forEach((v) => {{
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = `${{v.name}} (${{v.lang || 'n/a'}})`;
        sel.appendChild(opt);
      }});
      const canUsePrev = voices.some((v) => v.name === previous);
      sel.value = canUsePrev ? previous : chooseDefaultVoiceName(voices);
      selectedVoiceName = sel.value;
      sel.onchange = async () => {{
        selectedVoiceName = sel.value || chooseDefaultVoiceName(voices);
        await applyVoiceLanguage();
        renderScript();
        render();
      }};
    }}

    function detectVoiceLanguage() {{
      const v = getPreferredVoice();
      const lang = String((v && v.lang) || 'en-US').trim();
      return lang || 'en-US';
    }}

    function setStatusProgress(kind, text, percent, visible) {{
      const isVideo = kind === 'video';
      const wrap = document.getElementById(isVideo ? 'videoStatusWrap' : 'translationStatusWrap');
      const label = document.getElementById(isVideo ? 'videoStatus' : 'translationStatus');
      const bar = document.getElementById(isVideo ? 'videoProgressBar' : 'translationProgressBar');
      if (label && typeof text === 'string' && text.length) label.textContent = text;
      if (bar && Number.isFinite(percent)) {{
        const p = Math.max(0, Math.min(100, Math.round(percent)));
        bar.style.width = `${{p}}%`;
      }}
      if (wrap && typeof visible === 'boolean') wrap.style.display = visible ? 'block' : 'none';
    }}

    function startVideoProgressTicker() {{
      if (videoProgressTicker) clearInterval(videoProgressTicker);
      let p = 2;
      setStatusProgress('video', 'Preparing video request... 2%', p, true);
      const t0 = Date.now();
      videoProgressTicker = setInterval(() => {{
        const elapsed = (Date.now() - t0) / 1000;
        if (p < 92) p += elapsed < 8 ? 5 : elapsed < 20 ? 2 : 1;
        p = Math.min(92, p);
        let stage = 'Preparing chapters';
        if (p >= 28) stage = 'Generating narration audio';
        if (p >= 58) stage = 'Rendering scenes';
        if (p >= 82) stage = 'Merging and mixing audio';
        setStatusProgress('video', `${{stage}}... ${{p}}%`, p, true);
      }}, 1200);
    }}

    function stopVideoProgressTicker() {{
      if (videoProgressTicker) {{
        clearInterval(videoProgressTicker);
        videoProgressTicker = null;
      }}
    }}

    async function translateNarrations(targetLanguage) {{
      const lang = String(targetLanguage || '').trim();
      if (!lang || lang.toLowerCase().startsWith('en')) {{
        activeLanguage = 'source';
        setStatusProgress('translation', 'Subtitles are already in selected language.', 100, false);
        return sourceNarrations;
      }}
      if (narrationByLang[lang]) {{
        activeLanguage = lang;
        setStatusProgress('translation', 'Using cached subtitle translation.', 100, false);
        return narrationByLang[lang];
      }}

      const runId = ++translationRunId;
      const total = Math.max(1, sourceNarrations.length);
      const translated = sourceNarrations.slice();
      const chunkSize = Math.max(1, Math.min(4, Math.ceil(total / 8)));
      let completed = 0;
      setStatusProgress('translation', `Translating subtitles... 0% (0/${{total}})`, 0, true);
      try {{
        for (let start = 0; start < total; start += chunkSize) {{
          if (runId !== translationRunId) return sourceNarrations;
          const chunk = sourceNarrations.slice(start, start + chunkSize);
          const resp = await fetch('/api/story/translate', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ language: lang, narrations: chunk }}),
          }});
          if (!resp.ok) throw new Error('translation_failed');
          const payload = await resp.json();
          const rows = Array.isArray(payload?.narrations) ? payload.narrations : [];
          for (let i = 0; i < chunk.length; i += 1) {{
            translated[start + i] = String(rows[i] || chunk[i] || sourceNarrations[start + i] || '');
          }}
          completed = Math.min(total, start + chunk.length);
          const pct = Math.round((completed / total) * 100);
          setStatusProgress('translation', `Translating subtitles... ${{pct}}% (${{completed}}/${{total}})`, pct, true);
        }}
        narrationByLang[lang] = translated;
        if (runId === translationRunId) {{
          activeLanguage = lang;
        }}
        setStatusProgress('translation', `Subtitle translation complete (100%).`, 100, true);
        return translated;
      }} catch (_) {{
        activeLanguage = 'source';
        setStatusProgress('translation', 'Translation failed. Using source subtitles.', 100, true);
        return sourceNarrations;
      }}
    }}

    async function applyVoiceLanguage() {{
      if (playing) {{
        playing = false;
      }}
      stopSpeech();
      stopBgm();
      const lang = detectVoiceLanguage();
      isTranslating = true;
      setPlaybackControlsDisabled(true);
      translationPromise = translateNarrations(lang)
        .catch(() => sourceNarrations)
        .finally(() => {{
          isTranslating = false;
          setTimeout(() => setStatusProgress('translation', '', 0, false), 900);
          setPlaybackControlsDisabled(false);
        }});
      await translationPromise;
    }}

    async function waitForTranslation() {{
      try {{
        await translationPromise;
      }} catch (_) {{
        // Fallback to source subtitles on translation failure.
      }}
    }}

    function setPlaybackControlsDisabled(disabled) {{
      const ids = ['prevBtn', 'playBtn', 'readBtn', 'pauseBtn', 'nextBtn', 'chapterSelect', 'voiceSelect', 'videoBtn', 'regenBtn'];
      ids.forEach((id) => {{
        const el = document.getElementById(id);
        if (el) el.disabled = !!disabled;
      }});
    }}

    async function regenerateCurrentImage() {{
      if (!slides.length) return;
      const ok = window.confirm('Replace current chapter image with a newly generated one?');
      if (!ok) return;
      const btn = document.getElementById('regenBtn');
      const status = document.getElementById('regenStatus');
      if (btn) {{
        btn.disabled = true;
        btn.textContent = 'Regenerating...';
      }}
      if (status) status.style.display = 'inline';
      try {{
        const resp = await fetch(`/api/story/${{encodeURIComponent(taskId)}}/slides/${{idx}}/regenerate-image`, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
        }});
        if (!resp.ok) {{
          let msg = 'Image regeneration failed.';
          try {{
            const e = await resp.json();
            if (e?.error) msg = `${{msg}} ${{e.error}}`;
          }} catch (_) {{}}
          throw new Error(msg);
        }}
        const data = await resp.json();
        const nextUrl = String(data?.image_url || '');
        if (nextUrl) {{
          slides[idx] = {{ ...(slides[idx] || {{}}), image_url: nextUrl }};
          render(true);
        }}
      }} catch (e) {{
        window.alert(String(e?.message || e || 'Image regeneration failed.'));
      }} finally {{
        if (btn) {{
          btn.disabled = false;
          btn.textContent = 'Regenerate Image';
        }}
        if (status) status.style.display = 'none';
      }}
    }}

    function inferStoryMood() {{
      const blob = sourceNarrations.join(' ').toLowerCase();
      if (/(fear|dark|haunted|mystery|shadow|secret|nightmare)/.test(blob)) return 'mystery';
      if (/(battle|quest|adventure|dragon|hero|journey|epic)/.test(blob)) return 'adventure';
      if (/(happy|joy|love|friend|magic|dream|peace)/.test(blob)) return 'uplifting';
      return 'calm';
    }}

    function playBgmNote(ctx, bus, freq, startAt, dur, gain, type) {{
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = type || 'sine';
      osc.frequency.setValueAtTime(Math.max(40, Number(freq || 220)), startAt);
      g.gain.setValueAtTime(0.0001, startAt);
      g.gain.linearRampToValueAtTime(gain, startAt + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, startAt + Math.max(0.08, dur));
      osc.connect(g);
      g.connect(bus);
      osc.start(startAt);
      osc.stop(startAt + Math.max(0.1, dur + 0.04));
    }}

    async function startBgm() {{
      if (bgmTrackAudio || bgmCtx) return;
      try {{
        const bgmResp = await fetch(`/api/story/${{encodeURIComponent(taskId)}}/bgm`);
        if (bgmResp.ok) {{
          const bgmData = await bgmResp.json();
          const trackUrl = String(bgmData?.track_url || '');
          if (trackUrl) {{
            const a = new Audio(trackUrl);
            a.loop = true;
            a.volume = 0.28;
            bgmTrackAudio = a;
            await a.play();
            return;
          }}
        }}
      }} catch (_) {{}}
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      try {{
        bgmCtx = new Ctx();
        bgmMaster = bgmCtx.createGain();
        bgmMaster.gain.value = 0.0001;
        bgmMaster.connect(bgmCtx.destination);
        await bgmCtx.resume();
        const mood = inferStoryMood();
        const melodyByMood = {{
          mystery: [220, 246.94, 261.63, 246.94, 220, 196, 220, 246.94],
          adventure: [261.63, 329.63, 392.0, 329.63, 293.66, 329.63, 349.23, 392.0],
          uplifting: [293.66, 329.63, 392.0, 440.0, 392.0, 349.23, 329.63, 293.66],
          calm: [220, 246.94, 261.63, 293.66, 261.63, 246.94, 220, 196],
        }};
        const bassByMood = {{
          mystery: [110, 110, 123.47, 123.47],
          adventure: [130.81, 146.83, 164.81, 146.83],
          uplifting: [146.83, 164.81, 174.61, 164.81],
          calm: [110, 123.47, 130.81, 123.47],
        }};
        const melody = melodyByMood[mood] || melodyByMood.calm;
        const bass = bassByMood[mood] || bassByMood.calm;
        const beatDur = 0.44;
        bgmNextTime = bgmCtx.currentTime + 0.06;
        bgmStep = 0;
        if (bgmScheduler) clearInterval(bgmScheduler);
        bgmScheduler = setInterval(() => {{
          if (!bgmCtx || !bgmMaster) return;
          while (bgmNextTime < bgmCtx.currentTime + 0.35) {{
            const i = bgmStep % melody.length;
            const b = bgmStep % bass.length;
            const mFreq = melody[i];
            const bFreq = bass[b];
            playBgmNote(bgmCtx, bgmMaster, mFreq, bgmNextTime, beatDur * 0.82, 0.11, 'triangle');
            playBgmNote(bgmCtx, bgmMaster, bFreq, bgmNextTime, beatDur * 0.95, 0.06, 'sine');
            if ((bgmStep % 2) === 0) {{
              playBgmNote(bgmCtx, bgmMaster, mFreq * 1.5, bgmNextTime, beatDur * 0.62, 0.035, 'sine');
            }}
            bgmNextTime += beatDur;
            bgmStep += 1;
          }}
        }}, 110);
        const now = bgmCtx.currentTime;
        bgmMaster.gain.cancelScheduledValues(now);
        bgmMaster.gain.setValueAtTime(0.0001, now);
        bgmMaster.gain.linearRampToValueAtTime(0.22, now + 1.2);
      }} catch (_) {{
        stopBgm();
      }}
    }}

    function stopBgm() {{
      if (bgmTrackAudio) {{
        try {{
          bgmTrackAudio.pause();
          bgmTrackAudio.src = '';
        }} catch (_) {{}}
        bgmTrackAudio = null;
      }}
      if (!bgmCtx) return;
      try {{
        if (bgmScheduler) {{
          clearInterval(bgmScheduler);
          bgmScheduler = null;
        }}
        const now = bgmCtx.currentTime;
        if (bgmMaster) {{
          bgmMaster.gain.cancelScheduledValues(now);
          bgmMaster.gain.setValueAtTime(Math.max(0.0001, bgmMaster.gain.value || 0.0001), now);
          bgmMaster.gain.linearRampToValueAtTime(0.0001, now + 0.6);
        }}
        const nodes = bgmNodes.slice();
        setTimeout(() => {{
          nodes.forEach((n) => {{
            try {{ n.osc.stop(); }} catch (_) {{}}
          }});
          try {{ bgmCtx.close(); }} catch (_) {{}}
          bgmCtx = null;
          bgmMaster = null;
          bgmNodes = [];
        }}, 700);
      }} catch (_) {{
        bgmCtx = null;
        bgmMaster = null;
        bgmNodes = [];
      }}
    }}

    function renderChapterSelect() {{
      const sel = document.getElementById('chapterSelect');
      if (!sel) return;
      sel.innerHTML = '';
      slides.forEach((s, i) => {{
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `Chapter ${{i + 1}}`;
        sel.appendChild(opt);
      }});
      sel.value = String(idx);
      sel.onchange = () => {{
        const nextIdx = Number.parseInt(sel.value, 10);
        if (Number.isNaN(nextIdx)) return;
        playing = false;
        stopSpeech();
        idx = Math.max(0, Math.min(slides.length - 1, nextIdx));
        render();
      }};
    }}

    function renderScript() {{
      const root = document.getElementById('scriptList');
      if (!root) return;
      root.innerHTML = '';
      slides.forEach((s, i) => {{
        const li = document.createElement('li');
        li.id = 'script-item-' + i;
        const arr = narrationByLang[activeLanguage] || sourceNarrations;
        li.innerHTML = `<strong>Chapter ${{i + 1}}</strong><div>${{arr[i] || ''}}</div>`;
        root.appendChild(li);
      }});
    }}

    function copyText(value) {{
      navigator.clipboard.writeText(String(value || ''));
    }}

    function renderYoutubeMeta() {{
      const t = document.getElementById('ytTitle');
      const d = document.getElementById('ytDescription');
      const h = document.getElementById('ytHashtags');
      const k = document.getElementById('ytKeywords');
      if (t) t.value = String(youtubeMeta.title || '');
      if (d) d.value = String(youtubeMeta.description || '');
      if (h) h.value = String(youtubeMeta.hashtags || '');
      if (k) k.value = String(youtubeMeta.keywords || '');
    }}

    function syncActiveScript() {{
      slides.forEach((_, i) => {{
        const el = document.getElementById('script-item-' + i);
        if (el) el.classList.toggle('active', i === idx);
      }});
      const active = document.getElementById('script-item-' + idx);
      if (active) active.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
    }}

    function render(withTransition = true) {{
      const s = slides[idx] || {{}};
      const arr = narrationByLang[activeLanguage] || sourceNarrations;
      const narration = arr[idx] || '';
      const imgEl = document.getElementById('slideImage');
      const nextSrc = s.image_url || '';
      if (imgEl) {{
        const prevSrc = imgEl.getAttribute('data-src') || '';
        if (withTransition && prevSrc && prevSrc !== nextSrc) {{
          imgEl.style.opacity = '0.12';
          imgEl.style.transition = `opacity ${{Math.round(CHAPTER_TRANSITION_MS / 2)}}ms ease`;
          setTimeout(() => {{
            imgEl.src = nextSrc;
            imgEl.style.opacity = '1';
            imgEl.setAttribute('data-src', nextSrc);
          }}, Math.round(CHAPTER_TRANSITION_MS / 2));
        }} else {{
          imgEl.src = nextSrc;
          imgEl.style.opacity = '1';
          imgEl.setAttribute('data-src', nextSrc);
        }}
      }}
      setSubtitleSentence(narration, 0);
      document.getElementById('slideNarration').textContent = narration;
      const chapterSel = document.getElementById('chapterSelect');
      if (chapterSel) chapterSel.value = String(idx);
      syncActiveScript();
    }}

    function speak(text, onEnd, shouldContinue) {{
      if (!window.speechSynthesis) {{
        if (onEnd) onEnd();
        return;
      }}
      const full = String(text || '').trim();
      if (!full) {{
        if (onEnd) onEnd();
        return;
      }}

      const runId = ++narrationRunId;
      const parts = splitSentences(full).filter(Boolean);
      if (!parts.length) {{
        if (onEnd) onEnd();
        return;
      }}

      const ranges = [];
      let seek = 0;
      for (let i = 0; i < parts.length; i += 1) {{
        const sentence = String(parts[i] || '');
        const idxInFull = full.indexOf(sentence, seek);
        const startAt = idxInFull >= 0 ? idxInFull : seek;
        const endAt = startAt + sentence.length;
        ranges.push({{ start: startAt, end: endAt }});
        seek = Math.max(endAt, seek);
      }}

      const updateByChar = (charIndex) => {{
        if (runId !== narrationRunId) return;
        const ch = Math.max(0, Number(charIndex || 0));
        let sentenceIdx = 0;
        for (let i = 0; i < ranges.length; i += 1) {{
          if (ch >= ranges[i].start && ch <= Math.max(ranges[i].end, ranges[i].start)) {{
            sentenceIdx = i;
            break;
          }}
          if (ch > ranges[i].end) sentenceIdx = i;
        }}
        const sentence = parts[sentenceIdx] || '';
        const localChar = Math.max(0, ch - Number(ranges[sentenceIdx]?.start || 0));
        setSubtitleSentence(sentence, 0, localChar);
      }};

      const u = new SpeechSynthesisUtterance(full);
      const v = getPreferredVoice();
      if (v) {{
        u.voice = v;
        u.lang = v.lang || 'en-US';
      }} else {{
        u.lang = 'en-US';
      }}
      u.rate = 0.94;
      u.pitch = 1.0;
      u.volume = 1.0;

      let seenWordBoundaryEvents = 0;
      let lastBoundaryChar = -1;

      u.onstart = () => {{
        if (runId !== narrationRunId) return;
        clearSubtitleTicker();
        setSubtitleSentence(parts[0], 0, 0);

        // Fallback when boundary events are unavailable/sparse.
        const words = splitWords(full);
        if (words.length > 1) {{
          const estWps = Math.max(1.6, 2.6 * (u.rate || 1.0));
          const intervalMs = Math.max(120, Math.round((1000 / estWps) * 0.8));
          let fallbackIdx = 0;
          subtitleTicker = setInterval(() => {{
            if (runId !== narrationRunId) {{
              clearSubtitleTicker();
              return;
            }}
            if (fallbackIdx >= words.length) {{
              clearSubtitleTicker();
              return;
            }}
            const ch = Number(words[fallbackIdx].index || 0);
            updateByChar(ch);
            fallbackIdx += 1;
          }}, intervalMs);
        }}
      }};

      u.onboundary = (ev) => {{
        if (runId !== narrationRunId) return;
        if (!ev || typeof ev.charIndex !== 'number') return;
        const nextChar = Math.max(0, ev.charIndex);
        if (nextChar < lastBoundaryChar) return;
        const boundaryKind = String(ev.name || '').toLowerCase();
        const isWordBoundary = boundaryKind === 'word' || boundaryKind === '';
        if (isWordBoundary) seenWordBoundaryEvents += 1;
        lastBoundaryChar = nextChar;
        if (seenWordBoundaryEvents >= 2) clearSubtitleTicker();
        updateByChar(nextChar);
      }};

      u.onend = () => {{
        if (runId !== narrationRunId) return;
        clearSubtitleTicker();
        if (typeof shouldContinue === 'function' && !shouldContinue()) return;
        if (onEnd) onEnd();
      }};

      u.onerror = () => {{
        if (runId !== narrationRunId) return;
        clearSubtitleTicker();
        if (typeof shouldContinue === 'function' && !shouldContinue()) return;
        if (onEnd) onEnd();
      }};

      window.speechSynthesis.speak(u);
    }}

    function stopSpeech() {{
      narrationRunId += 1;
      clearSubtitleTicker();
      if (activeAudioEl) {{
        try {{
          activeAudioEl.pause();
          activeAudioEl.src = '';
        }} catch (_) {{}}
        activeAudioEl = null;
      }}
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }}

    async function readCurrent() {{
      await waitForTranslation();
      if (isTranslating) return;
      playing = false;
      stopSpeech();
      stopBgm();
      const s = slides[idx] || {{}};
      speak(getNarration(s), null, () => !playing);
    }}

    function playFromCurrent() {{
      if (!playing) return;
      const s = slides[idx] || {{}};
      speak(getNarration(s), () => {{
        if (!playing) return;
        if (idx < slides.length - 1) {{
          idx += 1;
          render();
          playFromCurrent();
        }} else {{
          playing = false;
          stopBgm();
          if (storyPlaybackEndResolver) {{
            const resolveEnd = storyPlaybackEndResolver;
            storyPlaybackEndResolver = null;
            resolveEnd();
          }}
        }}
      }}, () => playing);
    }}

    async function generateStoryVideo() {{
      if (!slides.length) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia || typeof MediaRecorder === 'undefined') {{
        window.alert('This browser does not support tab recording. Please use latest Chrome/Edge.');
        return;
      }}
      const btn = document.getElementById('videoBtn');
      const dl = document.getElementById('videoDownload');
      if (btn) {{
        btn.disabled = true;
        btn.textContent = 'Recording...';
      }}
      if (dl) dl.style.display = 'none';

      let stream = null;
      let composeStream = null;
      let recorder = null;
      let progressTimer = null;
      let drawTimer = null;
      const chunks = [];
      try {{
        const lang = detectVoiceLanguage();
        setStatusProgress('video', 'Preparing translated subtitles... 5%', 5, true);
        await translateNarrations(lang);
        setStatusProgress('video', 'Select This Tab and enable tab audio before recording.', 10, true);

        stream = await navigator.mediaDevices.getDisplayMedia({{
          video: {{ frameRate: 30 }},
          audio: true,
        }});
        if (!stream) throw new Error('record_stream_unavailable');
        const audioTracks = stream.getAudioTracks();
        if (!audioTracks.length) {{
          throw new Error('No tab audio detected. Please share this tab and enable tab audio.');
        }}

        const outW = 1920;
        const outH = 1080;
        const canvas = document.createElement('canvas');
        canvas.width = outW;
        canvas.height = outH;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('canvas_context_unavailable');

        const wrapText = (text, maxWidth) => {{
          const words = String(text || '').replace(/\\s+/g, ' ').trim().split(' ');
          const lines = [];
          let cur = '';
          for (const w of words) {{
            const probe = cur ? `${{cur}} ${{w}}` : w;
            if (ctx.measureText(probe).width <= maxWidth) {{
              cur = probe;
            }} else {{
              if (cur) lines.push(cur);
              cur = w;
            }}
          }}
          if (cur) lines.push(cur);
          return lines.slice(0, 3);
        }};

        const drawFrame = () => {{
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, outW, outH);
          const imgEl = document.getElementById('slideImage');
          if (imgEl && imgEl.naturalWidth > 0 && imgEl.naturalHeight > 0) {{
            const iw = imgEl.naturalWidth;
            const ih = imgEl.naturalHeight;
            const scale = Math.max(outW / iw, outH / ih);
            const dw = iw * scale;
            const dh = ih * scale;
            const dx = (outW - dw) / 2;
            const dy = (outH - dh) / 2;
            try {{ ctx.drawImage(imgEl, dx, dy, dw, dh); }} catch (_) {{}}
          }}

          const subtitleEl = document.getElementById('slideCaptionOverlay');
          const subtitle = String((subtitleEl && (subtitleEl.innerText || subtitleEl.textContent)) || '').trim();
          const boxH = 230;
          ctx.fillStyle = 'rgba(0,0,0,0.58)';
          ctx.fillRect(0, outH - boxH, outW, boxH);
          ctx.font = 'bold 56px Arial, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = '#ffffff';
          ctx.shadowColor = 'rgba(0,0,0,0.55)';
          ctx.shadowBlur = 6;
          const lines = wrapText(subtitle, outW - 120);
          const lineH = 66;
          const startY = outH - boxH / 2 - ((lines.length - 1) * lineH) / 2;
          lines.forEach((line, i) => {{
            ctx.fillText(line, outW / 2, startY + i * lineH);
          }});
          ctx.shadowBlur = 0;
        }};

        drawFrame();
        drawTimer = setInterval(drawFrame, Math.round(1000 / 30));
        composeStream = canvas.captureStream(30);
        const recTracks = [
          ...(composeStream.getVideoTracks() || []),
          ...audioTracks,
        ];
        const recStream = new MediaStream(recTracks);

        const mimeCandidates = [
          'video/webm;codecs=vp9,opus',
          'video/webm;codecs=vp8,opus',
          'video/webm',
        ];
        const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) || '';
        recorder = mimeType ? new MediaRecorder(recStream, {{ mimeType }}) : new MediaRecorder(recStream);
        recorder.ondataavailable = (ev) => {{
          if (ev && ev.data && ev.data.size > 0) chunks.push(ev.data);
        }};

        const stopped = new Promise((resolve) => {{
          recorder.onstop = () => resolve();
        }});

        const cancelAll = () => {{
          playing = false;
          stopSpeech();
          stopBgm();
          if (storyPlaybackEndResolver) {{
            const resolveEnd = storyPlaybackEndResolver;
            storyPlaybackEndResolver = null;
            resolveEnd();
          }}
        }};
        stream.getTracks().forEach((t) => {{
          t.onended = cancelAll;
        }});

        setStatusProgress('video', 'Recording image+subtitle video... 15%', 15, true);
        recorder.start(1000);
        const storyEnded = new Promise((resolve) => {{
          storyPlaybackEndResolver = resolve;
        }});

        await playStory();
        progressTimer = setInterval(() => {{
          const chapterFrac = slides.length > 0 ? (idx + 1) / slides.length : 0.0;
          const p = Math.max(18, Math.min(96, Math.round(18 + (chapterFrac * 76))));
          setStatusProgress('video', `Recording story... ${{p}}%`, p, true);
        }}, 700);
        await storyEnded;
        if (progressTimer) {{
          clearInterval(progressTimer);
          progressTimer = null;
        }}
        setStatusProgress('video', 'Finalizing recording... 98%', 98, true);

        if (recorder && recorder.state !== 'inactive') recorder.stop();
        if (drawTimer) {{
          clearInterval(drawTimer);
          drawTimer = null;
        }}
        if (composeStream) composeStream.getTracks().forEach((t) => t.stop());
        if (stream) stream.getTracks().forEach((t) => t.stop());
        await stopped;
        if (!chunks.length) throw new Error('recording_empty');

        const blob = new Blob(chunks, {{ type: recorder.mimeType || 'video/webm' }});
        const url = URL.createObjectURL(blob);
        const ext = (recorder.mimeType || '').includes('webm') ? 'webm' : 'webm';
        setStatusProgress('video', 'Video recording complete (100%).', 100, true);
        if (dl) {{
          dl.href = url;
          dl.download = `story_${{taskId}}_ui_voice.${{ext}}`;
          dl.style.display = 'inline';
          dl.textContent = 'Download Recorded Story Video (UI voice)';
        }}
      }} catch (e) {{
        setStatusProgress('video', 'Video generation failed.', 100, true);
        window.alert(String((e && e.message) || e || 'Video generation failed.'));
      }} finally {{
        if (progressTimer) clearInterval(progressTimer);
        if (drawTimer) clearInterval(drawTimer);
        progressTimer = null;
        drawTimer = null;
        storyPlaybackEndResolver = null;
        if (recorder && recorder.state !== 'inactive') {{
          try {{ recorder.stop(); }} catch (_) {{}}
        }}
        if (composeStream) {{
          try {{ composeStream.getTracks().forEach((t) => t.stop()); }} catch (_) {{}}
        }}
        if (stream) {{
          try {{ stream.getTracks().forEach((t) => t.stop()); }} catch (_) {{}}
        }}
        if (btn) {{
          btn.disabled = false;
          btn.textContent = 'Generate Video';
        }}
        setTimeout(() => setStatusProgress('video', '', 0, false), 1600);
      }}
    }}

    async function playStory() {{
      await waitForTranslation();
      if (isTranslating) return;
      if (!slides.length) return;
      stopSpeech();
      idx = 0;
      playing = true;
      await startBgm();
      render();
      playFromCurrent();
    }}

    function pauseStory() {{
      playing = false;
      stopSpeech();
      stopBgm();
    }}

    function prev() {{ playing = false; stopSpeech(); stopBgm(); idx = (idx - 1 + slides.length) % slides.length; render(); }}
    function next() {{ playing = false; stopSpeech(); stopBgm(); idx = (idx + 1) % slides.length; render(); }}

    renderScript();
    renderYoutubeMeta();
    renderChapterSelect();
    renderVoiceSelect();
    if (window.speechSynthesis) {{
      window.speechSynthesis.onvoiceschanged = () => renderVoiceSelect();
    }}
    applyVoiceLanguage().finally(() => {{
      renderScript();
      render(false);
    }});
  </script>
</body>
</html>"""

    def _story_placeholder_image(self, caption: str) -> str:
        text = (caption or "Illustration").strip()[:80]
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='1920' height='1080'>"
            "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0%' stop-color='#1e3a8a'/>"
            "<stop offset='100%' stop-color='#0f172a'/>"
            "</linearGradient></defs>"
            "<rect width='100%' height='100%' fill='url(#g)'/>"
            "<text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' "
            "fill='#e2e8f0' font-family='Arial, sans-serif' font-size='44'>"
            f"{text}"
            "</text></svg>"
        )
        return f"data:image/svg+xml;utf8,{quote(svg)}"

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
                intent = self._intent_for_task(task)
                task_category = self._task_category(task)

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
                        event_type="llm_prompt",
                        data={
                            "provider": "reasoning",
                            "model": (self.llm.get_model_for_task("reasoning").name if self.llm else ""),
                            "output": self._trim_text(f"LLM prompt (image refine): {task.description}", 2500),
                        },
                    )
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
                        data={
                            "original": task.description,
                            "refined": refined_prompt,
                            "source": refine_source,
                            "output": self._trim_text(f"Stability prompt: {refined_prompt}", 2500),
                        },
                    )
                    result = self.toolbus.image_generate(
                        refined_prompt,
                        aspect_ratio="16:9",
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
                            context_snapshot={
                                "model_used": {
                                    "provider": "stability",
                                    "name": "stable-image-core",
                                }
                            },
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
                            context_snapshot={
                                "model_used": {
                                    "provider": "stability",
                                    "name": "stable-image-core",
                                }
                            },
                        )
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="plan_step_update",
                            data={"index": 1, "status": "error", "step": image_plan_steps[1], "error": output},
                        )
                        self._log(session_id=sess.id, task_id=task.id, event_type="task_failed", data={"error": output})
                    return

                # Fast path: story generation with optional illustrations + slideshow preview.
                if task_category == "story":
                    story_steps = [
                        "Generate story structure and narration",
                        "Generate illustration images (optional)",
                        "Build slideshow with captions and voice narration",
                        "Finalize story preview",
                    ]
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_initialized",
                        data={"steps": story_steps},
                    )
                    model_info = self.llm.get_model_for_task("general")
                    self._update_task(task, context_snapshot={"model_info": model_info})
                    opts = self._story_options(task)
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="llm_prompt",
                        data={
                            "provider": model_info.provider,
                            "model": model_info.name,
                            "output": self._trim_text(
                                f"LLM prompt (story generation, target {opts.get('target_minutes', 5)} min): {task.description}",
                                2500,
                            ),
                        },
                    )
                    story_title, slides = self._generate_story_slides(task, model_info, opts)
                    story_character_bible = self._derive_story_character_bible(task.description, slides, model_info)
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_step_update",
                        data={"index": 0, "status": "completed", "step": story_steps[0]},
                    )
                    if story_character_bible:
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="story_character_bible",
                            data={"character_bible": story_character_bible},
                        )

                    illustration_summary = {
                        "requested": 0,
                        "generated": 0,
                        "failed": 0,
                        "errors": [],
                    }
                    if opts.get("generate_illustrations"):
                        stability = self.llm.config.get("stability_settings") or {}
                        style_key = str(opts.get("illustration_style") or "ghibli").strip().lower()
                        style_text, style_preset = self._story_style_prompt_and_preset(style_key)
                        count_mode = str(opts.get("illustration_count_mode") or "auto").strip().lower()
                        api_key = (
                            security.get_api_key("stability_api_key")
                            or os.getenv("LASE_STABILITY_API_KEY")
                            or stability.get("api_key")
                        )
                        if count_mode == "manual":
                            max_imgs = min(len(slides), int(opts.get("illustration_count", 10)))
                        else:
                            max_imgs = self._auto_story_illustration_count(slides)
                        # Guarantee at least one illustration attempt per chapter/slide.
                        max_imgs = max(len(slides), max_imgs)
                        illustration_summary["requested"] = max_imgs
                        if max_imgs > 0 and not api_key:
                            err = "Stability API key is not configured; using fallback placeholder illustrations."
                            illustration_summary["failed"] = max_imgs
                            illustration_summary["errors"].append(err)
                            self._log(
                                session_id=sess.id,
                                task_id=task.id,
                                event_type="warning",
                                data={"stage": "story_illustration_generation", "error": err},
                            )
                        for i in range(max_imgs):
                            if not api_key:
                                break
                            raw_prompt = slides[i].get("image_prompt") or slides[i].get("narration") or task.description
                            style_augmented = self._build_story_illustration_prompt(
                                raw_prompt,
                                style_key,
                                character_bible=story_character_bible,
                                chapter_idx=i + 1,
                            )
                            refined_prompt, _ = self._refine_image_prompt(style_augmented)
                            self._log(
                                session_id=sess.id,
                                task_id=task.id,
                                event_type="tool_execution",
                                data={
                                    "tool": "image.generate",
                                    "slide_index": i,
                                    "prompt": refined_prompt,
                                    "style": style_key,
                                    "output": self._trim_text(f"Stability prompt (slide {i + 1}): {refined_prompt}", 2500),
                                },
                            )
                            img_res = self.toolbus.image_generate(
                                refined_prompt,
                                negative_prompt=self._story_negative_prompt(),
                                aspect_ratio="16:9",
                                style_preset=style_preset or stability.get("default_style_preset", "") or None,
                                output_format=stability.get("default_output_format", "png"),
                                api_key=api_key,
                                base_url=stability.get("base_url", "https://api.stability.ai"),
                                timeout_s=int(stability.get("timeout", 120)),
                            )
                            if getattr(img_res, "ok", False):
                                image_path = (img_res.meta or {}).get("image_path", "")
                                image_url = self._to_image_preview_url(image_path)
                                if image_url:
                                    slides[i]["image_url"] = image_url
                                    illustration_summary["generated"] += 1
                                    self._log(
                                        session_id=sess.id,
                                        task_id=task.id,
                                        event_type="tool_result",
                                        data={
                                            "tool": "image.generate",
                                            "slide_index": i,
                                            "ok": True,
                                            "image_url": image_url,
                                        },
                                    )
                                    continue

                            # Retry once with a shorter prompt and without style preset.
                            fallback_prompt = (
                                slides[i].get("caption")
                                or slides[i].get("title")
                                or slides[i].get("narration")
                                or task.description
                            )
                            fallback_prompt = (
                                f"{fallback_prompt}, {style_text}, "
                                + (f"character bible: {story_character_bible}, " if story_character_bible else "")
                                + "coherent composition, clean anatomy, no text or watermark"
                            )
                            retry_res = self.toolbus.image_generate(
                                fallback_prompt,
                                negative_prompt=self._story_negative_prompt(),
                                aspect_ratio="16:9",
                                style_preset=style_preset,
                                output_format=stability.get("default_output_format", "png"),
                                api_key=api_key,
                                base_url=stability.get("base_url", "https://api.stability.ai"),
                                timeout_s=int(stability.get("timeout", 120)),
                            )
                            if getattr(retry_res, "ok", False):
                                image_path = (retry_res.meta or {}).get("image_path", "")
                                image_url = self._to_image_preview_url(image_path)
                                if image_url:
                                    slides[i]["image_url"] = image_url
                                    illustration_summary["generated"] += 1
                                    self._log(
                                        session_id=sess.id,
                                        task_id=task.id,
                                        event_type="tool_result",
                                        data={
                                            "tool": "image.generate",
                                            "slide_index": i,
                                            "ok": True,
                                            "retry": True,
                                            "image_url": image_url,
                                        },
                                    )
                                    continue

                            illustration_summary["failed"] += 1
                            err_out = getattr(retry_res, "output", "") or getattr(img_res, "output", "") or "unknown_error"
                            if len(illustration_summary["errors"]) < 3:
                                illustration_summary["errors"].append(err_out)
                            self._log(
                                session_id=sess.id,
                                task_id=task.id,
                                event_type="tool_result",
                                data={
                                    "tool": "image.generate",
                                    "slide_index": i,
                                    "ok": False,
                                    "error": err_out,
                                },
                            )
                        step_status = "completed"
                        if illustration_summary["requested"] > 0 and illustration_summary["generated"] == 0:
                            step_status = "error"
                        step_data = {
                            "index": 1,
                            "status": step_status,
                            "step": story_steps[1],
                            "generated": illustration_summary["generated"],
                            "requested": illustration_summary["requested"],
                        }
                        if illustration_summary["errors"]:
                            step_data["error"] = "; ".join(illustration_summary["errors"])
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="plan_step_update",
                            data=step_data,
                        )

                    # Ensure every slide has an image so preview always shows visuals.
                    for i in range(len(slides)):
                        if not str(slides[i].get("image_url") or "").strip():
                            slides[i]["image_url"] = self._story_placeholder_image(
                                slides[i].get("caption") or slides[i].get("title") or "Illustration"
                            )

                    story_dir = os.path.join(workspace, "story")
                    os.makedirs(story_dir, exist_ok=True)
                    html_path = os.path.join(story_dir, f"{task.id}.html")
                    html = self._build_story_html(task.id, story_title, slides)
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    preview_url = f"/api/story/{task.id}"
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_step_update",
                        data={"index": 2, "status": "completed", "step": story_steps[2]},
                    )
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="plan_step_update",
                        data={"index": 3, "status": "completed", "step": story_steps[3]},
                    )
                    self._update_task(
                        task,
                        status="completed",
                        progress=1.0,
                        status_detail="story_generated",
                        last_output=(
                            f"Story generated successfully.\nPreview URL: {preview_url}\n"
                            f"Target story length: {opts.get('target_minutes', 5)} minutes\n"
                            f"Illustrations: {'enabled' if opts.get('generate_illustrations') else 'disabled'}\n"
                            f"Illustration style: {opts.get('illustration_style', 'ghibli')}\n"
                            f"Illustration images generated: {illustration_summary['generated']}/{illustration_summary['requested']}"
                        ),
                        context_snapshot={
                            "story_artifact_path": html_path,
                            "story_title": story_title,
                            "story_slides": slides,
                            "story_options": opts,
                            "story_character_bible": story_character_bible,
                            "illustration_summary": illustration_summary,
                        },
                    )
                    self._log(
                        session_id=sess.id,
                        task_id=task.id,
                        event_type="tool_result",
                        data={"output": f"Story slideshow created. Preview URL: {preview_url}", "preview_url": preview_url},
                    )
                    self._log(session_id=sess.id, task_id=task.id, event_type="task_completed", data={})
                    return

                model_info = self.llm.get_model_for_task(intent.task_type)

                # --- Generative Planning Phase ---
                self._log(session_id=sess.id, task_id=task.id, event_type="thinking", data={"phase": "planning"})
                
                # Get workspace context
                ws_structure = self._get_workspace_structure(workspace)
                
                plan = self._generate_plan(task.description, model_info, ws_structure)
                self._log(
                    session_id=sess.id,
                    task_id=task.id,
                    event_type="llm_prompt",
                    data={
                        "provider": model_info.provider,
                        "model": model_info.name,
                        "output": self._trim_text(f"LLM prompt (planning): {task.description}", 2500),
                    },
                )
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
{self._category_instruction(task_category)}

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
            intent = self._intent_for_task(task)
            model_info = self.llm.get_model_for_task(intent.task_type)
            tools = self.toolbus.list_tools() # Tools might have changed
            website_task = self._task_category(task) == "website" or self._is_website_task(task.description)
            website_ready = False

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
                    latest_user = self._latest_user_message(messages)
                    if latest_user:
                        self._log(
                            session_id=sess.id,
                            task_id=task.id,
                            event_type="llm_prompt",
                            data={
                                "provider": model_info.provider,
                                "model": model_info.name,
                                "step": i + 1,
                                "output": self._trim_text(f"LLM prompt (step {i + 1}): {latest_user}", 2500),
                            },
                        )
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
                        if website_task and not website_ready:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Website task is not complete yet. Before DONE, you must ensure:\n"
                                        "1) The website is running and returns a Preview URL.\n"
                                        "2) The project contains executable ./start.sh for launching the site.\n"
                                        "Use tools to satisfy both requirements, then output DONE."
                                    ),
                                }
                            )
                            continue
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
                                aspect_ratio=tool_args.get("aspect_ratio", stability.get("default_aspect_ratio", "16:9")),
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
                                data={
                                    "original": raw_prompt,
                                    "refined": refined_prompt,
                                    "source": refine_source,
                                    "output": self._trim_text(f"Stability prompt: {refined_prompt}", 2500),
                                },
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
                                     if website_task:
                                         website_ready = True
                        else:
                             output_str = str(result)

                        if website_task and "Preview URL:" in output_str and "http://localhost:" in output_str:
                            website_ready = True

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
