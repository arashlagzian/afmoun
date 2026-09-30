# Data Folder

Put cached token files here. The training scripts expect a `metadata.json` file
and contiguous token memmaps. Depending on tokenizer vocabulary size, caches may
use either `uint16` or `uint32`.

```text
data/<cache_name>/
  metadata.json
  train_tokens_uint16.bin or train_tokens_uint32.bin
  eval_tokens_uint16.bin or eval_tokens_uint32.bin
```

The `metadata.json` should contain at least:

```json
{
  "dtype": "uint16",
  "block_size": 1024,
  "train_blocks": 195313,
  "eval_blocks": 977,
  "train_path": "train_tokens_uint16.bin",
  "eval_path": "eval_tokens_uint16.bin"
}
```

The paths may be absolute or relative to the metadata file's directory.

No dataset is committed to this release folder.
