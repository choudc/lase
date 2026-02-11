import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRun:
    name: str
    ok: bool
    returncode: int | None
    output: str
    meta: dict


class CodeQualityAuditor:
    """
    Design-aligned auditor that shells out to common tools when available.
    Tools are optional: missing tools are reported, not treated as a hard failure.
    """

    def _run(self, argv: list[str], cwd: str, timeout_s: int = 120) -> ToolRun:
        name = argv[0]
        exe = shutil.which(name)
        if not exe:
            return ToolRun(name=name, ok=False, returncode=None, output=f"{name} not installed", meta={"installed": False})

        try:
            cp = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = (cp.stdout or "") + (cp.stderr or "")
            return ToolRun(name=name, ok=cp.returncode == 0, returncode=cp.returncode, output=out.strip(), meta={"installed": True})
        except Exception as e:
            return ToolRun(name=name, ok=False, returncode=None, output=str(e), meta={"installed": True})

    def run(self, target_path: str) -> dict:
        target_path = os.path.abspath(target_path)
        runs: list[ToolRun] = []

        # Only run on folders; keep it deterministic.
        if not os.path.isdir(target_path):
            return {"status": "error", "error": "target_path_not_a_directory", "target_path": target_path, "runs": []}

        # Simple, common checks.
        runs.append(self._run(["flake8", "."], cwd=target_path))
        runs.append(self._run(["mypy", "."], cwd=target_path))
        runs.append(self._run(["bandit", "-r", "."], cwd=target_path))
        runs.append(self._run(["pylint", "."], cwd=target_path))
        runs.append(self._run(["safety", "check", "--full-report"], cwd=target_path))

        status = "ok" if all(r.ok or not r.meta.get("installed") for r in runs) else "issues_found"
        return {
            "status": status,
            "target_path": target_path,
            "runs": [
                {
                    "name": r.name,
                    "ok": r.ok,
                    "returncode": r.returncode,
                    "output": r.output[:20000],
                    "meta": r.meta,
                }
                for r in runs
            ],
        }

