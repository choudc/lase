import os
import subprocess
import time
import threading
import signal
import queue
import base64
from dataclasses import dataclass
from typing import Dict, Optional
import shutil
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from urllib.parse import quote_plus

@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    meta: dict

class ProcessManager:
    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        self._logs: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def start_process(self, cmd: str, cwd: str, id: str) -> bool:
        with self._lock:
            if id in self._processes and self._processes[id].poll() is None:
                return False  # Already running
            
            # Use a queue to capture output specifically for this process
            self._logs[id] = queue.Queue()

            try:
                # Start process with pipes for stdout/stderr
                # set_new_process_group=True to easily kill tree if needed
                proc = subprocess.Popen(
                    cmd, 
                    shell=True, 
                    cwd=cwd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1, # Line buffered
                    preexec_fn=os.setsid 
                )
                self._processes[id] = proc
                
                # Start a thread to read output
                def reader():
                    for line in iter(proc.stdout.readline, ''):
                        if line:
                            self._logs[id].put(line)
                            # In a real impl, we'd also emit to socketio here via a callback
                    proc.stdout.close()
                
                t = threading.Thread(target=reader, daemon=True)
                t.start()
                
                return True
            except Exception as e:
                print(f"Failed to start process {id}: {e}")
                return False

    def stop_process(self, id: str) -> bool:
        with self._lock:
            proc = self._processes.get(id)
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except:
                        pass
                del self._processes[id]
                if id in self._logs:
                    del self._logs[id]
                return True
            return False

    def get_logs(self, id: str) -> list[str]:
        # Flush queue
        lines = []
        q = self._logs.get(id)
        if q:
            while not q.empty():
                try:
                    lines.append(q.get_nowait())
                except queue.Empty:
                    break
        return lines

    def is_running(self, id: str) -> bool:
        proc = self._processes.get(id)
        return proc is not None and proc.poll() is None

    def get_active_pids(self) -> list[str]:
        return list(self._processes.keys())

class ToolBus:
    """
    Small, auditable tool surface area.
    """

    def __init__(self, workspaces_dir: str, generated_images_dir: str):
        self.workspaces_dir = workspaces_dir
        self.generated_images_dir = generated_images_dir
        self.process_manager = ProcessManager()

    def _write_web_scripts(self, project_dir: str) -> tuple[str, str]:
        start_script = os.path.join(project_dir, "start.sh")
        deploy_script = os.path.join(project_dir, "deploy.sh")
        with open(start_script, "w", encoding="utf-8") as f:
            f.write(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "PORT=\"${1:-3000}\"\n"
                "npm install\n"
                "npm run dev -- --port \"$PORT\" --host\n"
            )
        with open(deploy_script, "w", encoding="utf-8") as f:
            f.write(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "PORT=\"${1:-4173}\"\n"
                "npm install\n"
                "npm run build\n"
                "npm run preview -- --port \"$PORT\" --host\n"
            )
        os.chmod(start_script, 0o755)
        os.chmod(deploy_script, 0o755)
        return start_script, deploy_script

    def shell_run(self, command: str, cwd: str | None = None, timeout_s: int = 30) -> ToolResult:
        t0 = time.time()
        try:
            cp = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = (cp.stdout or "") + (cp.stderr or "")
            return ToolResult(ok=cp.returncode == 0, output=out.strip(), meta={"returncode": cp.returncode, "elapsed_s": time.time() - t0})
        except Exception as e:
            return ToolResult(ok=False, output=str(e), meta={"elapsed_s": time.time() - t0})

    def fs_read(self, path: str) -> ToolResult:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return ToolResult(ok=True, output=f.read(), meta={})
        except Exception as e:
            return ToolResult(ok=False, output=str(e), meta={})

    def fs_write(self, path: str, content: str) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(ok=True, output="written", meta={})
        except Exception as e:
            return ToolResult(ok=False, output=str(e), meta={})

    def fs_ls(self, path: str) -> ToolResult:
        try:
            if not os.path.exists(path):
                return ToolResult(ok=False, output=f"Path not found: {path}", meta={})
            
            items = []
            for entry in os.scandir(path):
                kind = "DIR" if entry.is_dir() else "FILE"
                items.append(f"{kind:4} {entry.name}")
            return ToolResult(ok=True, output="\n".join(sorted(items)), meta={})
        except Exception as e:
            return ToolResult(ok=False, output=str(e), meta={})

    def fs_mkdir(self, path: str) -> ToolResult:
        try:
            os.makedirs(path, exist_ok=True)
            return ToolResult(ok=True, output=f"Created directory: {path}", meta={})
        except Exception as e:
            return ToolResult(ok=False, output=str(e), meta={})

    def fs_delete(self, path: str) -> ToolResult:
        try:
            if not os.path.exists(path):
                 return ToolResult(ok=False, output=f"Path not found: {path}", meta={})
            
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return ToolResult(ok=True, output=f"Deleted: {path}", meta={})
        except Exception as e:
            return ToolResult(ok=False, output=str(e), meta={})

    # --- Web Tools ---
    
    def web_search(self, query: str) -> ToolResult:
        def _format_results(rows: list[dict], source: str) -> ToolResult:
            if not rows:
                return ToolResult(ok=True, output="No results found.", meta={"source": source})
            output = []
            for r in rows[:5]:
                title = r.get("title") or "Untitled"
                href = r.get("href") or ""
                body = r.get("body") or ""
                output.append(f"- [{title}]({href})\n  {body}")
            return ToolResult(ok=True, output="\n\n".join(output), meta={"source": source})

        def _fallback_html_search(q: str) -> list[dict]:
            # Lightweight fallback when DDGS API is rate-limited.
            url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = []
            for item in soup.select(".result")[:5]:
                a = item.select_one(".result__a")
                snippet = item.select_one(".result__snippet")
                if not a:
                    continue
                href = a.get("href") or ""
                rows.append(
                    {
                        "title": a.get_text(" ", strip=True),
                        "href": href,
                        "body": snippet.get_text(" ", strip=True) if snippet else "",
                    }
                )
            return rows

        try:
            results = DDGS().text(query, max_results=5)
            return _format_results(list(results or []), "ddgs")
        except Exception as e:
            msg = str(e)
            if "ratelimit" in msg.lower():
                try:
                    fallback_rows = _fallback_html_search(query)
                    if fallback_rows:
                        return _format_results(fallback_rows, "duckduckgo_html_fallback")
                except Exception as e2:
                    return ToolResult(ok=False, output=f"Search failed (rate-limited): {msg}; fallback failed: {e2}", meta={"recoverable": True})
                return ToolResult(ok=False, output=f"Search failed (rate-limited): {msg}", meta={"recoverable": True})
            return ToolResult(ok=False, output=f"Search failed: {e}", meta={})

    def web_read(self, url: str) -> ToolResult:
        try:
            # Fake user agent to avoid blocking
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.content, "html.parser")
            
            # Remove script/style
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text(separator="\n")
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit length
            return ToolResult(ok=True, output=text[:8000], meta={"url": url})
        except Exception as e:
             return ToolResult(ok=False, output=f"Failed to read URL: {e}", meta={})

    # --- Project Management Tools ---

    def project_init(self, project_type: str, name: str, parent_path: str = ".") -> ToolResult:
        """
        Scaffolds a new project.
        project_type: 'web' | 'android' | 'python'
        """
        try:
            full_path = os.path.join(self.workspaces_dir, parent_path) if not os.path.isabs(parent_path) else parent_path
            os.makedirs(full_path, exist_ok=True)
            
            target_dir = os.path.join(full_path, name)
            if os.path.exists(target_dir):
                 return ToolResult(ok=False, output=f"Directory {name} already exists in {full_path}", meta={})

            if project_type == "web":
                # Vite React
                cmd = f"npm create vite@latest {name} -- --template react"
                # We need to run this non-interactively. Vite create might prompt.
                # 'npm create vite@latest my-app -- --template react' is usually safe if dir doesn't exist.
                # However, to be ultra safe and autonomous, we use the template flag.
                r = self.shell_run(cmd, cwd=full_path, timeout_s=300)
                if not r.ok: return r
                
                # Install dependencies immediately to be helpful
                self.shell_run("npm install", cwd=target_dir, timeout_s=300)
                start_script, deploy_script = self._write_web_scripts(target_dir)
                return ToolResult(
                    ok=True,
                    output=(
                        f"Initialized Web (React) project in {target_dir}. Dependencies installed.\n"
                        f"Start script: {start_script}\nDeploy script: {deploy_script}"
                    ),
                    meta={"path": target_dir, "start_script": start_script, "deploy_script": deploy_script},
                )

            elif project_type == "android":
                # Expo
                # npx create-expo-app my-app -t default
                cmd = f"npx create-expo-app {name} -t default --no-install" 
                # --no-install to save time? No, we probably want deps. But standard create-expo-app installs them.
                # Let's just run it. It might take a while.
                r = self.shell_run(cmd, cwd=full_path, timeout_s=600)
                if not r.ok: return r
                return ToolResult(ok=True, output=f"Initialized Android (Expo) project in {target_dir}", meta={"path": target_dir})

            elif project_type == "python":
                os.makedirs(target_dir, exist_ok=True)
                # Create venv
                self.shell_run("python3 -m venv venv", cwd=target_dir)
                # Create main.py
                with open(os.path.join(target_dir, "main.py"), "w") as f:
                    f.write("print('Hello from Python Project')\n")
                return ToolResult(ok=True, output=f"Initialized Python project in {target_dir} with venv.", meta={"path": target_dir})

            else:
                return ToolResult(ok=False, output=f"Unknown project type: {project_type}", meta={})
        except Exception as e:
             return ToolResult(ok=False, output=str(e), meta={})

    def project_run(self, project_type: str, path: str, port: int = 3000) -> ToolResult:
        """
        Runs the project in the background.
        Returns a process ID (pid) that can be used to stop it.
        """
        full_path = os.path.join(self.workspaces_dir, path) if not os.path.isabs(path) else path
        if not os.path.exists(full_path):
             return ToolResult(ok=False, output=f"Path not found: {full_path}", meta={})
        
        cmd = ""
        if project_type == "web":
            # Vite
            cmd = f"npm run dev -- --port {port} --host"
        elif project_type == "android":
            # Expo web
            cmd = f"npx expo start --web --port {port}"
        elif project_type == "python":
            # Python main.py
            cmd = f"../venv/bin/python main.py" if os.path.exists(os.path.join(full_path, "../venv")) else "python3 main.py"
        else:
            return ToolResult(ok=False, output=f"Unknown/unsupported run type: {project_type}", meta={})

        proc_id = f"{project_type}_{port}" # Simple ID scheme for now
        
        if self.process_manager.start_process(cmd, full_path, proc_id):
            output = f"Started {project_type} on port {port}. PID: {proc_id}"
            meta = {"pid": proc_id, "port": port}
            if project_type == "web":
                preview_url = f"http://localhost:{port}"
                try:
                    start_script, deploy_script = self._write_web_scripts(full_path)
                    meta["start_script"] = start_script
                    meta["deploy_script"] = deploy_script
                    output = (
                        f"{output}\nPreview URL: {preview_url}\n"
                        f"Start script: {start_script}\nDeploy script: {deploy_script}"
                    )
                except Exception as e:
                    output = f"{output}\nPreview URL: {preview_url}\nWarning: Failed to create helper scripts: {e}"
                meta["preview_url"] = preview_url
            return ToolResult(ok=True, output=output, meta=meta)
        else:
            return ToolResult(ok=False, output=f"Failed to start or already running: {proc_id}", meta={})

    def project_stop(self, pid: str) -> ToolResult:
        if self.process_manager.stop_process(pid):
             return ToolResult(ok=True, output=f"Stopped process {pid}", meta={})
        else:
             return ToolResult(ok=False, output=f"Failed to stop or not found: {pid}", meta={})

    def project_build(self, project_type: str, path: str, platform: str = "android") -> ToolResult:
        """
        Builds the project. For Android, this means generating the APK.
        """
        full_path = os.path.join(self.workspaces_dir, path) if not os.path.isabs(path) else path
        if not os.path.exists(full_path):
             return ToolResult(ok=False, output=f"Path not found: {full_path}", meta={})

        if project_type == "web":
            cmd = "npm run build"
            r = self.shell_run(cmd, cwd=full_path, timeout_s=300)
            return r

        elif project_type == "android":
            # 1. Expo Prebuild (if needed creates android/ folder)
            # We assume it's an Expo project.
            
            # Check if android/ exists
            android_dir = os.path.join(full_path, "android")
            if not os.path.exists(android_dir):
                # Run prebuild
                # input: "y" might be needed if package name needs generation? 
                # --no-interactive should handle it?
                prebuild_cmd = "npx expo prebuild --platform android --no-interactive"
                r = self.shell_run(prebuild_cmd, cwd=full_path, timeout_s=300)
                if not r.ok:
                    return ToolResult(ok=False, output=f"Expo prebuild failed: {r.output}", meta={})
            
            # 2. Gradlew assembleDebug
            if not os.path.exists(os.path.join(android_dir, "gradlew")):
                 return ToolResult(ok=False, output="gradlew not found even after prebuild.", meta={})

            # Make gradlew executable
            self.shell_run("chmod +x gradlew", cwd=android_dir)
            
            # Run assembleDebug
            # This can take a LONG time. 10-15 mins first run.
            # We set a long timeout.
            build_cmd = "./gradlew assembleDebug"
            r = self.shell_run(build_cmd, cwd=android_dir, timeout_s=1200)
            
            if r.ok:
                # Find the APK
                apk_path = os.path.join(android_dir, "app/build/outputs/apk/debug/app-debug.apk")
                if os.path.exists(apk_path):
                     return ToolResult(ok=True, output=f"Build Successful! APK at: {apk_path}", meta={"apk_path": apk_path})
                else:
                     return ToolResult(ok=True, output="Build reported success but APK not found at expected path.", meta={})
            else:
                return ToolResult(ok=False, output=f"Gradle build failed:\n{r.output}", meta={})

        elif project_type == "python":
            # Syntax check?
            # or pip install?
            if os.path.exists(os.path.join(full_path, "requirements.txt")):
                 self.shell_run("pip install -r requirements.txt", cwd=full_path)
            
            # Recursively compile to check syntax
            r = self.shell_run("python3 -m compileall .", cwd=full_path)
            return r

        else:
            return ToolResult(ok=False, output=f"Unknown project type: {project_type}", meta={})

    def browser_screenshot(self, url: str) -> ToolResult:
        """
        Captures a screenshot of the given URL.
        Returns the path to the screenshot.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult(ok=False, output="Playwright not installed. Please run 'pip install playwright && playwright install chromium'", meta={})

        try:
            filename = f"screenshot_{int(time.time())}.png"
            path = os.path.join(self.generated_images_dir, filename)
            
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                # Wait a bit for render
                time.sleep(2)
                page.screenshot(path=path)
                browser.close()
                
            return ToolResult(ok=True, output=f"Screenshot saved to {path}", meta={"image_path": path})
        except Exception as e:
            return ToolResult(ok=False, output=f"Screenshot failed: {str(e)}", meta={})

    def image_generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        aspect_ratio: str = "1:1",
        seed: int = 0,
        style_preset: str | None = None,
        output_format: str = "png",
        api_key: str | None = None,
        base_url: str = "https://api.stability.ai",
        timeout_s: int = 120,
    ) -> ToolResult:
        """
        Generate an image using Stability AI Stable Image Core API.
        """
        if not prompt or not str(prompt).strip():
            return ToolResult(ok=False, output="Prompt is required for image generation.", meta={})
        if not api_key:
            return ToolResult(ok=False, output="Stability API key is not configured.", meta={})

        try:
            endpoint = f"{base_url.rstrip('/')}/v2beta/stable-image/generate/core"
            # Stability requires multipart/form-data. requests will set the proper
            # boundary/content-type when fields are sent via `files`.
            multipart_fields: list[tuple[str, tuple[None, str]]] = [
                ("prompt", (None, str(prompt))),
                ("aspect_ratio", (None, str(aspect_ratio or "1:1"))),
                ("output_format", (None, str(output_format or "png"))),
            ]
            if negative_prompt:
                multipart_fields.append(("negative_prompt", (None, str(negative_prompt))))
            if seed and int(seed) > 0:
                multipart_fields.append(("seed", (None, str(int(seed)))))
            if style_preset:
                multipart_fields.append(("style_preset", (None, str(style_preset))))

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*",
            }

            resp = requests.post(endpoint, headers=headers, files=multipart_fields, timeout=int(timeout_s))
            content_type = (resp.headers.get("content-type") or "").lower()

            if not resp.ok:
                try:
                    err_payload = resp.json()
                except Exception:
                    err_payload = resp.text
                return ToolResult(
                    ok=False,
                    output=f"Stability API error ({resp.status_code}): {err_payload}",
                    meta={"endpoint": endpoint, "status_code": resp.status_code},
                )

            if "application/json" in content_type:
                data = resp.json()
                img_b64 = data.get("image")
                if not img_b64:
                    return ToolResult(ok=False, output="Stability response missing image payload.", meta={"endpoint": endpoint})
                img_bytes = base64.b64decode(img_b64)
            else:
                img_bytes = resp.content

            ext = "png"
            if "jpeg" in content_type or output_format == "jpeg":
                ext = "jpg"
            elif "webp" in content_type or output_format == "webp":
                ext = "webp"
            os.makedirs(self.generated_images_dir, exist_ok=True)
            image_path = os.path.join(self.generated_images_dir, f"generated_{int(time.time())}.{ext}")
            with open(image_path, "wb") as f:
                f.write(img_bytes)

            return ToolResult(
                ok=True,
                output=f"Image generated successfully: {image_path}",
                meta={
                    "image_path": image_path,
                    "endpoint": endpoint,
                    "finish_reason": resp.headers.get("finish-reason"),
                    "seed": resp.headers.get("seed"),
                },
            )
        except Exception as e:
            return ToolResult(ok=False, output=f"Image generation failed: {e}", meta={})

    def list_tools(self) -> list[dict]:
        # Mirrors frontend expectations: name, description, safety fields.
        return [
            {"name": "shell.run", "description": "Run a shell command (local).", "safety": {"network": False, "needs_confirm": True}},
            {"name": "fs.read", "description": "Read a file.", "safety": {"network": False, "needs_confirm": False}},
            {"name": "fs.write", "description": "Write a file.", "safety": {"network": False, "needs_confirm": True}},
            {"name": "fs.ls", "description": "List directory contents.", "safety": {"network": False, "needs_confirm": False}},
            {"name": "fs.mkdir", "description": "Create a directory.", "safety": {"network": False, "needs_confirm": False}},
            {"name": "image.generate", "description": "Generate an image via Stability AI Stable Image Core API.", "safety": {"network": True, "needs_confirm": True}},
            {"name": "project.init", "description": "Initialize a new project (web, android, python).", "safety": {"network": True, "needs_confirm": True}},
            {"name": "project.run", "description": "Run a project in background (web, android, python).", "safety": {"network": True, "needs_confirm": True}},
            {"name": "project.stop", "description": "Stop a background project.", "safety": {"network": False, "needs_confirm": True}},
            {"name": "project.build", "description": "Build a project (Native Android APK, Web Build).", "safety": {"network": True, "needs_confirm": True}},
            {"name": "browser.screenshot", "description": "Capture a screenshot of a URL.", "safety": {"network": True, "needs_confirm": False}},
            {"name": "vision.analyze", "description": "Analyze an image using the configured Vision model.", "safety": {"network": True, "needs_confirm": False}},
            {"name": "web.search", "description": "Search the web for information.", "safety": {"network": True, "needs_confirm": False}},
            {"name": "web.read", "description": "Read the text content of a webpage.", "safety": {"network": True, "needs_confirm": False}},
            {"name": "fs.delete", "description": "Delete a file or directory. RISKY.", "safety": {"network": False, "needs_confirm": True}},
        ]
