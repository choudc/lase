import os
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    name: str


import json
import logging
import requests
import time
from flask import has_app_context

import logging
import requests
import time
from . import security
from ..config import load_config_safely
from ..db import db
from ..models.session import ApiUsage

logger = logging.getLogger(__name__)

class LLMAdapter:
    """
    Minimal, configuration-driven model router.
    This backend is local-first and works without any external model provider running.
    """

    def __init__(self, models_yaml_path: str):
        self.models_yaml_path = models_yaml_path
        self.config = self._load_config()

    def reload_config(self) -> dict:
        """
        Reload model/provider configuration from disk.
        Returns the effective config now in use.
        """
        self.config = self._load_config()
        return self.config

    def _is_chat_provider(self, provider: str) -> bool:
        return provider in {"ollama", "openai", "gemini", "anthropic", "deepseek"}

    def _load_config(self) -> dict:
        if not self.models_yaml_path:
            return {"default_models": {}}
        return load_config_safely(self.models_yaml_path)

    def _is_effective_key(self, value: str | None) -> bool:
        if not value:
            return False
        v = str(value).strip()
        if not v:
            return False
        if "YOUR_" in v:
            return False
        if v.startswith("****"):
            return False
        return True

    def _resolve_api_key(self, settings_section: str, key_name: str) -> str | None:
        key_from_keyring = security.get_api_key(key_name)
        if self._is_effective_key(key_from_keyring):
            return key_from_keyring

        settings = self.config.get(settings_section) or {}
        key_from_config = settings.get("api_key")
        if self._is_effective_key(key_from_config):
            return str(key_from_config).strip()
        return None

    def _estimate_openai_cost_usd(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Approximate costs in USD per 1K tokens.
        pricing_per_1k = {
            "gpt-4o": (0.005, 0.015),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0005, 0.0015),
        }
        in_cost, out_cost = pricing_per_1k.get(model_name, pricing_per_1k["gpt-4o"])
        return round((prompt_tokens / 1000.0) * in_cost + (completion_tokens / 1000.0) * out_cost, 8)

    def _record_api_usage(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost_usd: float) -> None:
        if not has_app_context():
            return
        try:
            row = ApiUsage(
                provider=provider,
                model=model,
                prompt_tokens=max(0, int(prompt_tokens or 0)),
                completion_tokens=max(0, int(completion_tokens or 0)),
                total_tokens=max(0, int(total_tokens or 0)),
                cost_usd=max(0.0, float(cost_usd or 0.0)),
            )
            db.session.add(row)
            db.session.commit()
        except Exception:
            db.session.rollback()

    def get_model_for_task(self, task_type: str) -> ModelInfo:
        default_models = self.config.get("default_models", {}) or {}
        # Fallback order: specific task -> general -> hardcoded local stub
        model = default_models.get(task_type) or default_models.get("general") or {"provider": "local", "name": "stub"}
        provider = str(model.get("provider", "local"))
        name = str(model.get("name", "stub"))

        # Non-chat providers (e.g. local_sdxl) cannot run the orchestration chat loop.
        # Fall back to a chat-capable general model while tools perform the specialist work.
        if not self._is_chat_provider(provider):
            general = default_models.get("general") or {}
            g_provider = str(general.get("provider", "local"))
            g_name = str(general.get("name", "stub"))
            if self._is_chat_provider(g_provider):
                return ModelInfo(provider=g_provider, name=g_name)
        return ModelInfo(provider=provider, name=name)

    def call_model(self, model_info: ModelInfo, messages: list[dict], tools: list[dict] = None) -> str:
        """
        Unified way to call LLMs.
        Returns the content string from the assistant.
        """
        if model_info.provider == "ollama":
            return self._call_ollama(model_info, messages, tools)
        elif model_info.provider == "openai":
            return self._call_openai(model_info, messages, tools)
        elif model_info.provider == "gemini":
            return self._call_gemini(model_info, messages, tools)
        elif model_info.provider == "anthropic":
            return self._call_anthropic(model_info, messages, tools)
        elif model_info.provider == "deepseek":
            return self._call_deepseek(model_info, messages, tools)
        else:
            return f"Error: Unsupported chat provider '{model_info.provider}' for model '{model_info.name}'."

    def _call_ollama(self, model_info: ModelInfo, messages: list[dict], tools: list[dict] = None) -> str:
        settings = self.config.get("ollama_settings") or {}
        base_url = (settings.get("base_url") or "http://localhost:11434").rstrip("/")
        timeout = int(settings.get("timeout") or 300)

        url = f"{base_url}/api/chat"
        
        # Convert standard multimodal messages (list of dicts) to Ollama format
        ollama_messages = []
        for m in messages:
            content = m["content"]
            images = []
            text_content = ""
            
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif part.get("type") == "image_url":
                        # Extract base64 from data url
                        url_str = part["image_url"]["url"]
                        if "base64," in url_str:
                            images.append(url_str.split("base64,")[1])
            else:
                text_content = content
            
            msg = {"role": m["role"], "content": text_content}
            if images:
                msg["images"] = images
            ollama_messages.append(msg)

        logger.info(f"Calling Ollama model: {model_info.name} at {url}")
        print(f"DEBUG: Calling Ollama model: '{model_info.name}' at {url}", flush=True)

        payload = {
            "model": model_info.name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 404:
                return f"Error calling Ollama: 404 Not Found. Model '{model_info.name}' not found."
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"Error calling Ollama: {e}"

    def _call_openai(self, model_info: ModelInfo, messages: list[dict], tools: list[dict] = None) -> str:
        settings = self.config.get("openai_settings") or {}
        base_url = (settings.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        api_key = self._resolve_api_key("openai_settings", "openai_api_key")
        
        timeout = int(settings.get("timeout") or 60)
        max_retries = int(settings.get("max_retries") or 3)
        retry_backoff_s = float(settings.get("retry_backoff_s") or 2.0)
        
        if not api_key:
             return "Error: OpenAI API Key not configured."

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_info.name,
            "messages": messages,
            "temperature": 0.0
        }
        
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                if resp.status_code == 429 and attempt <= max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = retry_backoff_s * (2 ** (attempt - 1))
                    try:
                        if retry_after:
                            wait_s = max(wait_s, float(retry_after))
                    except Exception:
                        pass
                    logger.warning(f"OpenAI 429 rate limited; retrying in {wait_s:.1f}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait_s)
                    continue

                if 500 <= resp.status_code <= 599 and attempt <= max_retries:
                    wait_s = retry_backoff_s * (2 ** (attempt - 1))
                    logger.warning(f"OpenAI {resp.status_code}; retrying in {wait_s:.1f}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait_s)
                    continue

                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                cost_usd = self._estimate_openai_cost_usd(model_info.name, prompt_tokens, completion_tokens)
                self._record_api_usage("openai", model_info.name, prompt_tokens, completion_tokens, total_tokens, cost_usd)
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                if attempt <= max_retries:
                    wait_s = retry_backoff_s * (2 ** (attempt - 1))
                    time.sleep(wait_s)
                    continue
                return f"Error calling OpenAI: {e}"

    def _call_gemini(self, model_info: ModelInfo, messages: list[dict], tools: list[dict] = None) -> str:
        settings = self.config.get("gemini_settings", {})
        api_key = self._resolve_api_key("gemini_settings", "gemini_api_key")
        if not api_key:
            return "Error: Gemini API key not configured."
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_info.name}:generateContent?key={api_key}"
        
        # Convert to Gemini Content format
        gemini_contents = []
        for m in messages:
            role = "user" if m["role"] in ["user", "system"] else "model"
            parts = []
            
            content = m["content"]
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                         url_str = part["image_url"]["url"]
                         if "base64," in url_str:
                             mime_type = url_str.split(";")[0].split(":")[1]
                             data_b64 = url_str.split("base64,")[1]
                             parts.append({"inline_data": {"mime_type": mime_type, "data": data_b64}})
            else:
                if m["role"] == "system":
                     parts.append({"text": "System: " + str(content)})
                     role = "user" # System prompt hack
                else:
                     parts.append({"text": str(content)})
            
            gemini_contents.append({"role": role, "parts": parts})

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=settings.get("timeout", 60))
            if resp.status_code != 200:
                 return f"Error from Gemini: {resp.text}"
            
            data = resp.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"Error calling Gemini: {str(e)}"

    def _call_anthropic(self, model_info: ModelInfo, messages: list[dict], tools: list[dict] = None) -> str:
        settings = self.config.get("anthropic_settings", {})
        api_key = self._resolve_api_key("anthropic_settings", "anthropic_api_key")
        if not api_key:
            return "Error: Anthropic API key not configured."

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        system_prompt = ""
        filtered_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += m["content"] + "\n"
            else:
                filtered_messages.append({"role": m["role"], "content": m["content"]})
        
        payload = {
            "model": model_info.name,
            "messages": filtered_messages,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=settings.get("timeout", 60))
            if resp.status_code != 200:
                return f"Error from Anthropic: {resp.text}"
            
            return resp.json()['content'][0]['text']
        except Exception as e:
             return f"Error calling Anthropic: {str(e)}"

    def _call_deepseek(self, model_info: ModelInfo, messages: list[dict], tools: list[dict] = None) -> str:
        settings = self.config.get("deepseek_settings", {})
        api_key = self._resolve_api_key("deepseek_settings", "deepseek_api_key")
        base_url = settings.get("base_url", "https://api.deepseek.com")
        
        if not api_key:
            return "Error: DeepSeek API key not configured."
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_info.name,
            "messages": messages,
            "temperature": 0.7
        }
        
        try:
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=settings.get("timeout", 60))
            if resp.status_code != 200:
                 return f"Error from DeepSeek: {resp.text}"
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling DeepSeek: {str(e)}"


    def list_available_models(self) -> list[dict]:
        """
        Returns a list of available models from configured providers.
        For Ollama, it fetches from the local instance.
        For APIs, it returns a curated list of popular models.
        """
        models = []
        
        # 1. Ollama Models
        ollama_settings = self.config.get("ollama_settings") or {}
        base_url = (ollama_settings.get("base_url") or "http://localhost:11434").rstrip("/")
        timeout = int(ollama_settings.get("timeout") or 5) # Short timeout for listing

        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("models", []):
                    models.append({
                        "provider": "ollama",
                        "name": model["name"],
                        "description": f"Local Ollama: {model['name']}",
                        "group": "Local (Ollama)"
                    })
        except Exception:
            # excessive logging might be annoying if ollama is down
            pass

        # 2. OpenAI Models (Static list for now, could be dynamic but expensive/slow)
        if self._resolve_api_key("openai_settings", "openai_api_key"):
            models.extend([
                {"provider": "openai", "name": "gpt-4o", "description": "OpenAI GPT-4o", "group": "OpenAI"},
                {"provider": "openai", "name": "gpt-4-turbo", "description": "OpenAI GPT-4 Turbo", "group": "OpenAI"},
                {"provider": "openai", "name": "gpt-3.5-turbo", "description": "OpenAI GPT-3.5 Turbo", "group": "OpenAI"},
            ])
            
        # 3. Gemini Models
        if self._resolve_api_key("gemini_settings", "gemini_api_key"):
            models.extend([
                {"provider": "gemini", "name": "gemini-1.5-pro-latest", "description": "Google Gemini 1.5 Pro", "group": "Google Gemini"},
                {"provider": "gemini", "name": "gemini-1.5-flash-latest", "description": "Google Gemini 1.5 Flash", "group": "Google Gemini"},
            ])
            
        # 4. Anthropic Models
        if self._resolve_api_key("anthropic_settings", "anthropic_api_key"):
             models.extend([
                {"provider": "anthropic", "name": "claude-3-5-sonnet-20240620", "description": "Anthropic Claude 3.5 Sonnet", "group": "Anthropic"},
                {"provider": "anthropic", "name": "claude-3-opus-20240229", "description": "Anthropic Claude 3 Opus", "group": "Anthropic"},
                {"provider": "anthropic", "name": "claude-3-haiku-20240307", "description": "Anthropic Claude 3 Haiku", "group": "Anthropic"},
            ])
             
        # 5. DeepSeek
        if self._resolve_api_key("deepseek_settings", "deepseek_api_key"):
            models.extend([
                {"provider": "deepseek", "name": "deepseek-chat", "description": "DeepSeek Chat", "group": "DeepSeek"},
                {"provider": "deepseek", "name": "deepseek-coder", "description": "DeepSeek Coder", "group": "DeepSeek"},
            ])

        return models
