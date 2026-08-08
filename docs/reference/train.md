# Training Python API

The supported training interface is the `dmri train` command. The functions
below are reusable when constructing simulators, models, dataloaders, or
checkpoint bundles in Python. The functions and state classes in
`dmri.train.train_script` implement the CLI loop and are not a supported
programmatic trainer or callback interface.

See the [training guide](../guides/training.md) for configuration, output, resume,
and evaluation examples.

## Builders

::: dmri.train.build_simulator.build_simulator
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.train.build_model.build_model
    options:
      show_root_heading: true
      show_root_full_path: false

`build_simulator(cfg)` returns the configured `MultiCompartment` class and one
training-data generator per acquisition entry. `build_model(cfg, sim_type)`
builds an initialized model for that simulator class.

## Simulation data

::: dmri.train.dataset.SimulationDataset
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - reset
        - close
        - set_data
        - get_stats

::: dmri.train.dataset.instantiate_dataloader
    options:
      show_root_heading: true
      show_root_full_path: false

## Checkpoint loading

::: dmri.train.utils.load_cfg
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.train.utils.load_checkpoint
    options:
      show_root_heading: true
      show_root_full_path: false

`load_checkpoint` accepts a local run or a Hugging Face model subfolder. It
returns `(checkpoint, model, simulator_class)`. The model is rebuilt in eval
mode, but the returned checkpoint parameters are not applied automatically.

## Checkpoint bundles

::: dmri.train.utils.bundle_checkpoint
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.train.utils.upload_checkpoint_to_hub
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.train.utils.download_checkpoint_from_hub
    options:
      show_root_heading: true
      show_root_full_path: false
