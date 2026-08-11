from __future__ import annotations

import json
from functools import lru_cache
from urllib import error, request

from fastapi import HTTPException

from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class RagLLM:
    def answer(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

        payload = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        http_request = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=120) as response:
                response_payload = json.loads(
                    response.read().decode("utf-8")
                )

        except error.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Ollama returned HTTP {exc.code} "
                    f"while using model {OLLAMA_MODEL}."
                ),
            ) from exc

        except error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ollama is unavailable at {OLLAMA_BASE_URL}. "
                    f"Start Ollama and make sure {OLLAMA_MODEL} is available."
                ),
            ) from exc

        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned invalid JSON.",
            ) from exc

        answer = response_payload.get("response")

        if not isinstance(answer, str):
            raise HTTPException(
                status_code=502,
                detail="Ollama response does not contain a valid answer.",
            )

        answer = answer.strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an empty response.",
            )

        return answer


@lru_cache(maxsize=1)
def get_rag_llm() -> RagLLM:
    return RagLLM()