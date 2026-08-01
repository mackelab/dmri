# Pretrained Model API

The package root exposes `dmri.load_pretrained`, `dmri.list_pretrained_models`, and `dmri.PretrainedModel`.

```python
import dmri

available = dmri.list_pretrained_models()
bundle = dmri.load_pretrained("b3s_2_4_6_128")

model = bundle.model
simulator = bundle.simulator
config = bundle.config
step = bundle.step
```

By default, these functions use the public `manugloeck/dmri-pretrained` repository. Model files are downloaded and cached by `huggingface_hub`. `load_pretrained` accepts `revision`, `cache_dir`, and `local_files_only` for reproducible, custom-cache, and offline loading. `list_pretrained_models` accepts a repository and optional revision.

Pretrained models are acquisition-specific. Verify that a checkpoint is compatible with the acquisition scheme before using its output. DMRI is research-only and has not been clinically validated.

## API

::: dmri.hub
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - PretrainedModel
        - load_pretrained
        - list_pretrained_models
