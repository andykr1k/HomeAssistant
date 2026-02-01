from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


class LLMBackend:
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        raise NotImplementedError


class TransformersBackend(LLMBackend):
    def __init__(self, model_id: str, device: str, dtype: str, trust_remote_code: bool) -> None:
        self._logger = logging.getLogger(__name__)
        torch_dtype = self._parse_dtype(dtype)
        device_map = "auto" if device == "auto" else None
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if device_map is None:
            self._model.to(device)
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._apply_chat_template(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p", 0.9)
        max_tokens = kwargs.get("max_tokens", 256)
        do_sample = temperature > 0.0

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        text = self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        return text.strip()

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        system = messages[0]["content"]
        user = messages[1]["content"]
        return f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n"

    @staticmethod
    def _parse_dtype(value: str):
        value = (value or "auto").lower()
        if value == "auto":
            return "auto"
        if value == "float16":
            return torch.float16
        if value == "bfloat16":
            return torch.bfloat16
        if value == "float32":
            return torch.float32
        return "auto"


class LLMRouter:
    def __init__(self, tools, debug: bool = False) -> None:
        self._tools = tools
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._backend = self._init_backend()

    def enabled(self) -> bool:
        return self._backend is not None

    def handle(self, text: str) -> str:
        if not self._backend:
            return "LLM not configured."
        try:
            tool_calls = self._tool_call(text)
            tool_results = []
            for call in tool_calls:
                result = self._tools.call_tool(call.name, call.arguments)
                tool_results.append(
                    {
                        "name": call.name,
                        "ok": result.ok,
                        "message": result.message,
                        "data": result.data or {},
                        "silent": result.silent,
                    }
                )

            return self._answer(text, tool_results)
        except Exception as exc:
            if self._debug:
                self._logger.debug("LLM routing failed: %s", exc)
            return (
                "The local LLM is unavailable. Check that the model is installed "
                "and the Transformers backend can load it."
            )

    def _execute_tool_calls(
        self, text: str, tool_calls: List[ToolCall], allow_llm_answer: bool = True
    ) -> str:
        tool_results = []
        for call in tool_calls:
            result = self._tools.call_tool(call.name, call.arguments)
            tool_results.append(
                {
                    "name": call.name,
                    "ok": result.ok,
                    "message": result.message,
                    "data": result.data or {},
                    "silent": result.silent,
                }
            )
        if tool_results and all(item.get("silent") for item in tool_results):
            return ""
        if not allow_llm_answer:
            return ""
        return self._answer(text, tool_results)

    def _tool_call(self, text: str) -> List[ToolCall]:
        tool_specs = self._tools.tool_specs()
        now = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        system_prompt = (
            "You are a tool router for a local Jarvis assistant.\n"
            "Return ONLY valid JSON. No markdown, no commentary.\n"
            "If no tool is needed, return {\"tool_calls\": []}.\n"
        )
        user_prompt = (
            f"Current date/time: {now}\n"
            f"User message: {text}\n\n"
            "Available tools (name, description, arguments schema):\n"
            f"{json.dumps(tool_specs, indent=2)}\n\n"
            "Return JSON in the form:\n"
            "{\"tool_calls\": [{\"name\": \"tool_name\", \"arguments\": {}}]}"
        )
        raw = self._backend.generate(
            system_prompt,
            user_prompt,
            temperature=float(os.getenv("LLM_TOOL_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_TOOL_MAX_TOKENS", "256")),
        )
        if self._debug:
            self._logger.debug("Tool caller raw: %s", raw)
        data = self._parse_json(raw)
        calls = []
        for item in data.get("tool_calls", []):
            name = item.get("name")
            args = item.get("arguments") or {}
            if name:
                calls.append(ToolCall(name=name, arguments=args))
        return calls

    def _answer(self, text: str, tool_results: List[Dict[str, Any]]) -> str:
        now = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        if tool_results and all(item.get("silent") for item in tool_results):
            return ""
        system_prompt = (
            "You are Jarvis, a concise, confident home assistant.\n"
            "Respond naturally to the user. Use tool results when provided.\n"
            "If tools failed, explain briefly and suggest next steps.\n"
        )
        user_prompt = (
            f"Current date/time: {now}\n"
            f"User message: {text}\n"
            f"Tool results (JSON): {json.dumps(tool_results, indent=2)}\n\n"
            "Answer the user in one or two sentences."
        )
        raw = self._backend.generate(
            system_prompt,
            user_prompt,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "256")),
        )
        if self._debug:
            self._logger.debug("Answer raw: %s", raw)
        return raw.strip()

    def _init_backend(self) -> Optional[LLMBackend]:
        model_id = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-3b-Instruct")
        device = os.getenv("LLM_DEVICE", "auto")
        dtype = os.getenv("LLM_DTYPE", "auto")
        trust_remote_code = os.getenv("LLM_TRUST_REMOTE_CODE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        return TransformersBackend(
            model_id=model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"tool_calls": []}

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {"tool_calls": []}

        return {"tool_calls": []}
