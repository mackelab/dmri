# Neural Nets API

Core network building blocks for embedding, model selection, and inference. Start with `dmri_reconstruction_model` for the assembled model; use the remaining modules when customizing an architecture.

For an end-to-end workflow, see [training from scratch](../examples/04_train_standalone.md).

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
