import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    task_type: str  # coding|general|reasoning|vision|image_generation
    confidence: float


CODING_WORDS = {
    "code",
    "script",
    "program",
    "function",
    "develop",
    "implement",
    "debug",
    "refactor",
    "algorithm",
    "programming",
    "software",
    "application",
    "api",
    "database",
    "web",
    "frontend",
    "backend",
}

IMAGE_GEN_WORDS = {
    "image",
    "picture",
    "draw",
    "generate visual",
    "create art",
    "illustration",
    "graphic",
    "design",
    "artwork",
    "photo",
}

REASONING_WORDS = {
    "reason",
    "analyze",
    "understand",
    "explain",
    "concept",
    "research",
    "study",
    "investigate",
    "evaluate",
    "assess",
}

VISION_WORDS = {
    "see",
    "look at",
    "interpret image",
    "describe image",
    "visual",
    "screenshot",
    "photo analysis",
}

CATEGORY_HINTS = {
    "story": {"story", "novel", "narrative", "plot", "character", "fairy tale", "bedtime"},
    "image": {"image", "illustration", "draw", "art", "picture", "photo", "poster"},
    "website": {"website", "web app", "landing page", "frontend", "react", "vite", "html", "css"},
    "android_app": {"android app", "apk", "expo", "kotlin", "jetpack compose", "mobile app"},
    "python_app": {"python app", "python script", "flask app", "fastapi", "django", "cli tool"},
    "research": {"research", "analyze", "analysis", "compare", "investigate", "report", "study"},
}


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def classify_intent(text: str) -> Intent:
    """
    Very small, deterministic classifier based on the design doc keyword lists.
    This is intentionally simple: it’s cheap, local-first, and easy to audit.
    """
    t = _norm(text)
    if not t:
        return Intent(task_type="general", confidence=0.2)

    def contains_any(phrases: set[str]) -> bool:
        return any(p in t for p in phrases)

    if contains_any(IMAGE_GEN_WORDS):
        return Intent(task_type="image_generation", confidence=0.7)
    if contains_any(VISION_WORDS):
        return Intent(task_type="vision", confidence=0.6)
    if contains_any(CODING_WORDS):
        return Intent(task_type="coding", confidence=0.7)
    if contains_any(REASONING_WORDS):
        return Intent(task_type="reasoning", confidence=0.6)

    return Intent(task_type="general", confidence=0.4)


def infer_category(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    # Prioritize more specific categories first.
    for cat in ["story", "android_app", "python_app", "website", "image", "research"]:
        if any(p in t for p in CATEGORY_HINTS[cat]):
            return cat
    return None
