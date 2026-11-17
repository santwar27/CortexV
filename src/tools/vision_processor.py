import logging
import os

import requests

log = logging.getLogger(__name__)

# Multimodal generation is the LIKELIEST Ollama hang on the M1 16GB (big
# model, model swap under memory pressure) — and this used to be the one
# request in the repo with NO timeout, so a stuck Llava pinned the turn's
# thread forever (M4-A5). Connect fails fast; read is generous because a
# cold multimodal load + generation legitimately takes a while.
_CONNECT_TIMEOUT_S = float(os.getenv("CORTEX_VISION_CONNECT_TIMEOUT", "5"))
_READ_TIMEOUT_S = float(os.getenv("CORTEX_VISION_READ_TIMEOUT", "120"))


class VisionProcessor:
    """
    Tool to process Base64 encoded images using the local Llava multimodal model.
    """
    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name
        # Same host the rest of the stack talks to — OLLAMA_HOST may point
        # at another box (e.g. the Mac Studio) without a code change.
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = f"http://{host}"
        self.api_url = f"{host}/api/generate"

    def analyze_image(self, prompt: str, image_b64: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt if prompt else "Describe this image in detail.",
            "images": [image_b64],
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url, json=payload,
                timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
            )
            response.raise_for_status()
            return response.json().get("response", "[SYSTEM ERROR] Empty response from vision model.")
        except requests.exceptions.Timeout:
            log.warning(f"vision model timed out after {_READ_TIMEOUT_S:.0f}s read ceiling")
            return ("[SYSTEM ERROR] Vision model didn't answer in time. "
                    "The image was probably too heavy for the local model right now.")
        except Exception as e:
            return f"[SYSTEM ERROR] Vision processing failed: {str(e)}"
