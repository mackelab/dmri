# Prediction Python API

Use [`dmri predict`](../cli/predict.md) for the complete folder-based prediction
workflow. For Python code that manages model inputs and inference directly, the
package root exposes `dmri.load_pretrained`, `dmri.list_pretrained_models`, and
`dmri.PretrainedModel`.

```python
import dmri

available = dmri.list_pretrained_models()
bundle = dmri.load_pretrained("msb3s_2_4_6_128")

model = bundle.model
simulator = bundle.simulator
config = bundle.config
step = bundle.step
```

By default, these functions use the public `manugloeck/dmri-pretrained`
repository. Model files are downloaded and cached by `huggingface_hub`.
`load_pretrained` accepts `revision`, `cache_dir`, `local_files_only`, and a
network-forward `precision`. CPU fp16 requests fall back to fp32.

`load_pretrained` applies EMA parameters when available and otherwise applies
raw parameters. The returned model is in evaluation mode. The `simulator` field
is the reconstructed `MultiCompartment` class.

Pretrained models are acquisition-specific. Verify compatibility with the input
acquisition before using their output. DMRI has not been clinically validated.

## API

::: dmri.hub
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - PretrainedModel
        - load_pretrained
        - list_pretrained_models
