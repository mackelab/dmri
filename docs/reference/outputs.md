# Checkpoints And Outputs

## Training run

A single training job normally writes:

```text
results/<run>/<timestamp>/
|-- .hydra/
|-- artifact.yaml
|-- checkpoints/
|   |-- <step>/
|   `-- best/
|       `-- <step>/
`-- train_script.log
```

Numeric directories directly below `checkpoints/` are regular checkpoints.
`latest` means the highest retained numeric step. `best` is selected by the
validation objective when best-checkpoint tracking is enabled.

Checkpoint payloads contain raw parameters and training state. Runs with EMA
also store `params_ema`; raw parameters remain available as `params`.

## Portable model bundle

`bundle_checkpoint` and `upload_checkpoint_to_hub` create this layout:

```text
<model_name>/
|-- config.yaml
|-- artifact.yaml
`-- checkpoints/
    |-- best/
    `-- <latest_step>/
```

Best and latest are included by default. Temporary Orbax checkpoint directories
are excluded. A Hugging Face repository can contain multiple model subfolders.

## Prediction output

`dmri predict FOLDER` writes below `FOLDER/dmri_output/` unless
`--output-subdir` changes the name:

```text
dmri_output/
|-- .hydra/
|-- ball3stick_inference_results/
|-- ball3stick_model_selection_results/
`-- view_results.html
```

`view_results.html` contains selected scalar maps. `--no-viewer` skips it. The
NIfTI files are the complete output.

Common inference filenames use these prefixes:

- `mean_`: posterior mean.
- `std_`: posterior standard deviation when enabled by the exporter.
- `merged_`: retained sample arrays or merged sample results where configured.
- `frac_`: fraction of samples satisfying a condition.

For Ball3Stick exports, `f0` is the isotropic fraction and `f1`, `f2`, and `f3`
are stick fractions after fraction-based ordering. `fsum` is the total
anisotropic fraction. Direction maps and diffusivity/SNR summaries are written
by the same exporter. Outside-mask voxels are zero.

The precise file set depends on the selected exporter, model family, whether
standard deviations are enabled, and whether model selection ran.

## Evaluation output

`dmri eval` chooses the export root from `run.output_dir` when set. Otherwise it
uses the selected checkpoint directory. Export configurations create named
subdirectories such as:

```text
ball3stick_inference_results/
ball3stick_model_selection_results/
```

Configured metrics can write:

- one or more NIfTI maps;
- `<metric>_summary.json` files with configured aggregations;
- additional JSON diagnostics for calibration, classification, or SBC.

Metrics that do not apply to the available data may be skipped and write no
file. Ground-truth metrics require synthetic truth or another explicitly
provided reference.

## Loading a checkpoint

`load_checkpoint` rebuilds a model and returns the checkpoint payload separately:

```python
from flax import nnx
from dmri.train.utils import load_checkpoint

checkpoint, model, simulator_class = load_checkpoint(RUN_DIR, which="latest")
params = checkpoint.get("params_ema", checkpoint["params"])
nnx.update(model, params)
model.eval()
```

Use `dmri.load_pretrained` when you want a Hub model with parameters already
applied.
