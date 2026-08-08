# Neural Nets API

Core network building blocks for embedding, model selection, and inference. Start with `dmri_reconstruction_model` for the assembled model; use the remaining modules when customizing an architecture.

For a small in-process example, see
[standalone training](../examples/04_train_standalone.md). Use `dmri train` for
normal checkpointed training runs.

## Reconstruction and selection

::: dmri.nn.dmri_reconstruction_model
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.nn.autoregressive
    options:
      show_root_heading: true
      show_root_full_path: false

## Embeddings and tokenization

::: dmri.nn.embedding_net
    options:
      show_root_heading: true
      show_root_full_path: false

::: dmri.nn.tokenizer
    options:
      show_root_heading: true
      show_root_full_path: false

## Transformer backbones

::: dmri.nn.simformer
    options:
      show_root_heading: true
      show_root_full_path: false
