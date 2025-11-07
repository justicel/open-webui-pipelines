from typing import Any, List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os
import requests
from urllib.parse import urljoin


class Pipeline:
    MEDIA_EXTENSIONS = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".svg",
        ".webp",
        ".avif",
        ".heic",
        ".heif",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".mpg",
        ".mpeg",
        ".m4v",
    )
    MEDIA_MIME_PREFIXES = ("image/", "video/")

    class Valves(BaseModel):
        OPENAI_API_KEY: str = ""
        OPENAI_BASE_URL: str = "https://api.openai.com/v1"
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "Default Open Pipeline"
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv(
                    "OPENAI_API_KEY", "your-openai-api-key-here"
                ),
                "OPENAI_BASE_URL": os.getenv(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                "DEFAULT_MEDIA_MODEL": os.getenv(
                    "DEFAULT_MEDIA_MODEL", "qwen3-vl-235b-thinking-fp8"
                ),
                "DEFAULT_MODEL": os.getenv(
                    "DEFAULT_MODEL", "ring-1t-fp8"
                ),
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def _looks_like_media_reference(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False

        lowered = value.lower()
        if lowered.startswith("data:image/") or lowered.startswith("data:video/"):
            return True

        trimmed = lowered.split("?", 1)[0].split("#", 1)[0]
        return any(trimmed.endswith(ext) for ext in self.MEDIA_EXTENSIONS)

    def _contains_media_payload(self, payload: Any, _visited=None) -> bool:
        if _visited is None:
            _visited = set()

        if isinstance(payload, dict):
            obj_id = id(payload)
            if obj_id in _visited:
                return False
            _visited.add(obj_id)

            type_val = payload.get("type")
            if isinstance(type_val, str) and any(
                keyword in type_val.lower() for keyword in ("image", "video")
            ):
                return True

            for key in ("media_type", "mime_type", "content_type"):
                value = payload.get(key)
                if isinstance(value, str) and value.lower().startswith(
                    self.MEDIA_MIME_PREFIXES
                ):
                    return True

            asset_pointer = payload.get("asset_pointer")
            if asset_pointer and self._contains_media_payload(asset_pointer, _visited):
                return True

            for value in payload.values():
                if self._contains_media_payload(value, _visited):
                    return True

            return False

        if isinstance(payload, list):
            obj_id = id(payload)
            if obj_id in _visited:
                return False
            _visited.add(obj_id)

            return any(self._contains_media_payload(item, _visited) for item in payload)

        if isinstance(payload, str):
            return self._looks_like_media_reference(payload)

        return False

    def _is_media_analysis_request(
        self, user_message: str, messages: List[dict], body: dict
    ) -> bool:
        if self._contains_media_payload(messages):
            return True

        for key in (
            "messages",
            "files",
            "attachments",
            "inputs",
            "input",
            "media",
            "assets",
        ):
            candidate = body.get(key)
            if candidate and self._contains_media_payload(candidate):
                return True

        return self._looks_like_media_reference(user_message)

    def _select_model(self, user_message: str, messages: List[dict], body: dict) -> str:
        if self._is_media_analysis_request(user_message, messages, body):
            return self.valves.DEFAULT_MEDIA_MODEL
        return self.valves.DEFAULT_MODEL

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        OPENAI_API_KEY = self.valves.OPENAI_API_KEY
        MODEL = self._select_model(user_message, messages, body)

        headers = {}
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "model": MODEL}
        base_url = self.valves.OPENAI_BASE_URL

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url=urljoin(base_url.rstrip("/") + "/", "chat/completions"),
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
