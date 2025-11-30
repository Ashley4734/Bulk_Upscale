# Agent Instructions

- Cache AI model weights under `~/.cache/realesrgan` and avoid hard-coding absolute paths elsewhere.
- When updating AI upscaling logic, ensure model download URLs stay in sync with the supported `--ai-model` options.
- Prefer standard library downloads (e.g., `urllib.request`) to keep dependencies minimal for setup instructions.
- For Python 3.11–3.13 environments, use the torch 2.9.1 / torchvision 0.24.1 pairing specified in `requirements-ai.txt` to avoid missing wheel errors on newer interpreters.
