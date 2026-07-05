from __future__ import annotations

import json
from functools import lru_cache
from urllib import error, request

from fastapi import HTTPException

from backend.config import AGENT_OLLAMA_MODEL, OLLAMA_BASE_URL


class AgentLLM:
    def answer(self, prompt: str) -> str:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = json.dumps(
            {
                "model": AGENT_OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        http_request = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama is unavailable at {OLLAMA_BASE_URL}. Start Ollama and pull {AGENT_OLLAMA_MODEL}.",
            ) from exc

        answer = response_payload.get("response", "")
        if not answer:
            raise HTTPException(status_code=502, detail="Ollama returned an empty response")

        return answer.strip()


@lru_cache(maxsize=1)
def get_agent_llm() -> AgentLLM:
    return AgentLLM()
