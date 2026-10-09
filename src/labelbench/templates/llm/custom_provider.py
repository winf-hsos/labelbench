"""Custom LLM provider for labelbench, created by `labelbench new classifier`.

Connect any model or API here. labelbench calls complete() once per item:

- system  the rendered system prompt (instructions and label list), or None
- user    the rendered item prompt
- schema  JSON schema the answer must follow: {"reasoning": ..., "label": ...}
- model   the model name from the config
- further keyword arguments: everything under `options` in the config

Return the model's answer as JSON text, or a dict with "text" and optionally
"usage" (token counts) and "model" (the model that answered).
"""


def complete(system, user, schema, model, **options):
    raise NotImplementedError("Call your model here and return its JSON answer as text.")
