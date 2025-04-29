# HISTORY_OFFLOAD_API.md

## Chat History: RL Swarm Offload to OpenAI API

### Context
- Date: April 29, 2025
- OS: macOS
- Workspace: RL Swarm (open source, peer-to-peer RL system)
- Key files/folders: `hivemind_exp/runner/grpo_runner.py`, `hivemind_exp/trainer/hivemind_grpo_trainer.py`, `OFFLOAD_API.md`

### User Goals
- Reduce local GPU/CPU usage by offloading model inference to an external API (OpenAI, e.g., GPT-3.5/4).
- Do not remove or break the original local inference code; comment it out and provide a switch.
- Document the process and implementation steps.

### Steps Taken
1. User asked for best options to lower local usage; chose option-2 (offload to API).
2. Implementation plan:
   - Add a config flag (e.g., `use_openai_api`) to toggle API offload.
   - Create an `OpenAIModelWrapper` in `grpo_runner.py`.
   - In `get_model`, return the wrapper if offload is enabled; otherwise, use the local model (original code commented for clarity).
   - In `hivemind_grpo_trainer.py`, detect the wrapper and use it for inference, with error handling; otherwise, use the local model (original code preserved).
   - Write all steps and code samples in `OFFLOAD_API.md`.
3. All changes are implemented and documented. The user can now toggle between local and API-based inference.

### How to Continue
- Update your config (YAML or CLI) to include `use_openai_api: true`, `openai_api_key`, and (optionally) `openai_model_name`.
- Ensure `openai` Python package is installed (`pip install openai`).
- If you want to offload to a different API (e.g., Deepseek), create a similar wrapper and update the logic accordingly.
- For further extension, consider:
  - Supporting batch inference via API (if the provider allows).
  - Adding more robust error/retry logic for API calls.
  - Allowing per-stage or per-round switching between local and API inference.
- Refer to `OFFLOAD_API.md` for detailed step-by-step instructions and code locations.

---

This file tracks the offload-to-API implementation history and context. Update it as you make further changes or add new API providers.
