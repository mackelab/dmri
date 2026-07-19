from __future__ import annotations

from functools import partial
from typing import Any, Iterable, List, Literal, Mapping

import torch
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu, tensor, uint8

from sbi.neural_nets.estimators import NFlowsFlow
from sbi.neural_nets.net_builders.flow import (
    ContextSplineMap,
    build_nsf as sbi_build_nsf,
    get_base_dist,
)
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import standardizing_transform, z_score_parser
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.user_input_checks import check_data_device


Condition = Tensor | Mapping[str, Any]


def _iter_condition_tensors(condition: Condition) -> Iterable[Tensor]:
    if torch.is_tensor(condition):
        yield condition
        return
    if isinstance(condition, Mapping):
        for value in condition.values():
            yield from _iter_condition_tensors(value)
        return
    raise TypeError(
        "Conditions must be tensors or nested mappings of tensors, "
        f"got {type(condition)}."
    )


def _condition_batch_size(condition: Condition) -> int:
    for tensor_leaf in _iter_condition_tensors(condition):
        if tensor_leaf.ndim == 0:
            raise ValueError("Condition tensor leaves must have a batch dimension.")
        return int(tensor_leaf.shape[0])
    raise ValueError("Condition mapping does not contain any tensor leaves.")


def _condition_device(condition: Condition) -> torch.device:
    for tensor_leaf in _iter_condition_tensors(condition):
        return tensor_leaf.device
    raise ValueError("Condition mapping does not contain any tensor leaves.")


def _slice_condition(condition: Condition, stop: int) -> Condition:
    if torch.is_tensor(condition):
        return condition[:stop]
    return {key: _slice_condition(value, stop) for key, value in condition.items()}


def _repeat_condition(condition: Condition, repeats: int) -> Condition:
    if torch.is_tensor(condition):
        return condition.repeat(repeats, *([1] * (condition.ndim - 1)))
    return {key: _repeat_condition(value, repeats) for key, value in condition.items()}


def _check_condition_data_device(batch_x: Tensor, batch_y: Condition) -> None:
    for tensor_leaf in _iter_condition_tensors(batch_y):
        check_data_device(batch_x, tensor_leaf)


def _infer_condition_features(batch_y: Condition, embedding_net: nn.Module) -> int:
    example_condition = _slice_condition(batch_y, 1)
    example_device = _condition_device(example_condition)

    was_training = embedding_net.training
    embedding_net = embedding_net.to(example_device)
    embedding_net.eval()
    with torch.no_grad():
        embedded = embedding_net(example_condition)
    if was_training:
        embedding_net.train()

    if not torch.is_tensor(embedded):
        raise TypeError("embedding_net must return a tensor.")
    if embedded.ndim == 0:
        return 1
    return int(embedded.reshape(embedded.shape[0], -1).shape[-1])


class DictConditionNFlowsFlow(NFlowsFlow):
    """NFlows wrapper that accepts dict or nested-mapping conditions."""

    def log_prob(self, input: Tensor, condition: Condition) -> Tensor:
        input_sample_dim = input.shape[0]
        input_batch_dim = input.shape[1]
        condition_batch_dim = _condition_batch_size(condition)

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        flattened_input = input.reshape((input_batch_dim * input_sample_dim, -1))
        repeated_condition = _repeat_condition(condition, input_sample_dim)
        log_probs = self.net._log_prob(flattened_input, repeated_condition)
        return log_probs.reshape((input_sample_dim, input_batch_dim))

    def loss(self, input: Tensor, condition: Condition) -> Tensor:
        return -self.log_prob(input.unsqueeze(0), condition)[0]

    def sample(self, sample_shape: torch.Size, condition: Condition) -> Tensor:
        condition_batch_dim = _condition_batch_size(condition)
        num_samples = torch.Size(sample_shape).numel()

        samples = self.net._sample(num_samples, condition)
        samples = samples.transpose(0, 1)
        return samples.reshape((*sample_shape, condition_batch_dim, *self.input_shape))

    def sample_and_log_prob(
        self,
        sample_shape: torch.Size,
        condition: Condition,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        condition_batch_dim = _condition_batch_size(condition)
        num_samples = torch.Size(sample_shape).numel()

        samples = self.net._sample(num_samples, condition)
        flattened_samples = samples.reshape(condition_batch_dim * num_samples, -1)
        repeated_condition = _repeat_condition(condition, num_samples)
        log_probs = self.net._log_prob(flattened_samples, repeated_condition)
        samples = samples.transpose(0, 1).reshape(
            (*sample_shape, condition_batch_dim, *self.input_shape)
        )
        log_probs = log_probs.reshape(condition_batch_dim, num_samples).transpose(0, 1)
        log_probs = log_probs.reshape((*sample_shape, condition_batch_dim))
        return samples, log_probs


def build_nsf(
    batch_x: Tensor,
    batch_y: Condition,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> NFlowsFlow:
    """Build an NSF that also supports dict-structured conditions."""
    if torch.is_tensor(batch_y):
        return sbi_build_nsf(
            batch_x=batch_x,
            batch_y=batch_y,
            z_score_x=z_score_x,
            z_score_y=z_score_y,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_bins=num_bins,
            embedding_net=embedding_net,
            tail_bound=tail_bound,
            hidden_layers_spline_context=hidden_layers_spline_context,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            **kwargs,
        ).to(batch_x.device)

    _check_condition_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)

    z_score_y_bool, _ = z_score_parser(z_score_y)
    if z_score_y_bool:
        raise NotImplementedError(
            "Dict-conditioned NSF does not support automatic z-scoring for y. "
            "Apply any preprocessing inside the embedding_net instead."
        )

    y_numel = _infer_condition_features(batch_y, embedding_net)

    def mask_in_layer(index: int) -> Tensor:
        return create_alternating_binary_mask(
            features=x_numel,
            even=(index % 2 == 0),
        )

    if x_numel == 1:
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    transform_list: List[transforms.Transform] = []
    for index in range(num_transforms):
        block: List[transforms.Transform] = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(index) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        if x_numel > 1:
            block.append(transforms.LULinear(x_numel, identity_init=True))
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [standardizing_transform(batch_x, structured_x)] + transform_list

    distribution = get_base_dist(x_numel, **kwargs)
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, embedding_net)
    return DictConditionNFlowsFlow(
        neural_net,
        input_shape=batch_x[0].shape,
        condition_shape=torch.Size([y_numel]),
    ).to(batch_x.device)
