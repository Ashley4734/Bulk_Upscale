# Agent Instructions

- Cache AI model weights under `~/.cache/realesrgan` and avoid hard-coding absolute paths elsewhere.
- When updating AI upscaling logic, ensure model download URLs stay in sync with the supported `--ai-model` options.
- Prefer standard library downloads (e.g., `urllib.request`) to keep dependencies minimal for setup instructions.
