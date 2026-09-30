# Models Folder

Put local Hugging Face model/config/tokenizer folders here.

Example:

```text
models/
  Llama-3.2-1B/
    config.json
    tokenizer.json
    ...
```

For from-scratch pretraining, the decoder-only paper scripts load the model
config and initialize random weights when their `--init-from-config` path is
used. See the repository-level README for the setting-specific launcher and
trainer names.

No model weights are committed to this release folder.
