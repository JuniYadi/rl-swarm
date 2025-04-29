# Offloading RL Swarm Model Inference to OpenAI API

This guide describes how to offload model inference in RL Swarm to the OpenAI API, reducing local GPU/CPU requirements. The original local inference code is preserved (commented out) for easy switching.

## 1. Install OpenAI Python SDK

```bash
pip install openai
```

## 2. Add a Config Flag

Add a flag (e.g., `use_openai_api`) to your config file (YAML or CLI) to control whether to use the OpenAI API for inference.

## 3. Modify Model Loading

In `hivemind_exp/runner/grpo_runner.py`, update `get_model` to return an OpenAI wrapper if `use_openai_api` is set:

```python
# ...existing code...
import openai

class OpenAIModelWrapper:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = api_key

    def __call__(self, prompt, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return response["choices"][0]["message"]["content"]

# In get_model:
if getattr(args, "use_openai_api", False):
    return OpenAIModelWrapper(api_key=args.openai_api_key, model_name=getattr(args, "openai_model_name", "gpt-3.5-turbo"))
# ...existing code...
```

## 4. Modify Inference Calls

In `hivemind_exp/trainer/hivemind_grpo_trainer.py`, wherever the model is called for inference (e.g., `model(inputs)`), add a branch:

```python
# ...existing code...
if hasattr(self.model, "__class__") and self.model.__class__.__name__ == "OpenAIModelWrapper":
    # Offload to OpenAI
    output = self.model(prompt)
else:
    # output = self.model(inputs)  # Local inference (commented out)
    ...existing code...
```

## 5. Pass API Key

Make sure your config or CLI passes the OpenAI API key and model name to the runner.

## 6. Error Handling

Add try/except blocks around API calls to handle rate limits and errors gracefully.

---

This approach lets you switch between local and API-based inference by toggling a config flag. Training (backpropagation) is not supported via OpenAI API, so this is for inference-only RL stages.
