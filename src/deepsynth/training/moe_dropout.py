"""Utilities to regularise Mixture-of-Experts layers via dropout."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import torch

LOGGER = logging.getLogger(__name__)


class ExpertGradientDropout:
    """Randomly drops expert gradients during fine-tuning.

    The class groups parameters by expert index using the naming convention
    ``experts.{id}`` that is commonly used in MoE decoder implementations.  At
    each optimisation step, gradients belonging to a randomly selected subset
    of experts are set to zero which mimics expert-level dropout.  The
    remaining gradients are re-scaled to keep the expected update magnitude
    unchanged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dropout_rate: float,
        min_keep: int = 1,
    ) -> None:
        self.dropout_rate = float(max(0.0, dropout_rate))
        self.min_keep = max(1, int(min_keep))
        self._expert_groups = self._group_parameters(model)

        if self.dropout_rate <= 0.0 or not self._expert_groups:
            if self.dropout_rate > 0.0:
                LOGGER.warning(
                    "Expert dropout requested but no experts were detected. "
                    "Parameters must contain the pattern 'experts.<id>'."
                )
            return

        LOGGER.info(
            "Expert dropout enabled: %.0f%% drop rate across %d experts",
            self.dropout_rate * 100.0,
            len(self._expert_groups),
        )

    @staticmethod
    def _group_parameters(model: torch.nn.Module) -> Dict[int, List[torch.nn.Parameter]]:
        expert_pattern = re.compile(r"experts\.(\d+)")
        groups: Dict[int, List[torch.nn.Parameter]] = defaultdict(list)

        for name, parameter in model.named_parameters():
            match = expert_pattern.search(name)
            if match:
                groups[int(match.group(1))].append(parameter)

        return dict(groups)

    def apply(self) -> None:
        """Apply dropout to expert gradients if configured."""

        if self.dropout_rate <= 0.0 or not self._expert_groups:
            return

        keep_prob = 1.0 - self.dropout_rate
        expert_ids: Sequence[int] = sorted(self._expert_groups)

        if not expert_ids:
            return

        mask = torch.bernoulli(
            torch.full((len(expert_ids),), keep_prob, dtype=torch.float32)
        )

        # Guarantee that at least ``min_keep`` experts remain active.
        if int(mask.sum().item()) < self.min_keep:
            active_indices = torch.randperm(len(expert_ids))[: self.min_keep]
            mask[active_indices] = 1.0

        scale = 1.0 / keep_prob if keep_prob > 0 else 0.0

        for idx, expert_id in enumerate(expert_ids):
            params = self._expert_groups[expert_id]
            if mask[idx].item() <= 0.0:
                for param in params:
                    if param.grad is not None:
                        param.grad.zero_()
                continue

            if scale != 1.0:
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(scale)


class GateGradientDropout:
    """Applies dropout to router/gating parameters during optimisation."""

    def __init__(
        self,
        model: torch.nn.Module,
        dropout_rate: float,
        keywords: Iterable[str] = ("gate", "router"),
    ) -> None:
        self.dropout_rate = float(max(0.0, dropout_rate))
        self.keep_prob = 1.0 - self.dropout_rate
        lowered_keywords = tuple(keyword.lower() for keyword in keywords)

        self._parameters: List[torch.nn.Parameter] = [
            param
            for name, param in model.named_parameters()
            if any(keyword in name.lower() for keyword in lowered_keywords)
        ]

        if self.dropout_rate <= 0.0 or not self._parameters:
            if self.dropout_rate > 0.0:
                LOGGER.warning(
                    "Gate dropout requested but no parameters matched keywords: %s",
                    ", ".join(lowered_keywords),
                )
            return

        LOGGER.info(
            "Gate dropout enabled: %.0f%% drop rate across %d parameter tensors",
            self.dropout_rate * 100.0,
            len(self._parameters),
        )

    def apply(self) -> None:
        """Randomly zero router gradients while preserving expectation."""

        if self.dropout_rate <= 0.0 or not self._parameters:
            return

        if self.keep_prob <= 0.0:
            for param in self._parameters:
                if param.grad is not None:
                    param.grad.zero_()
            return

        scale = 1.0 / self.keep_prob

        for param in self._parameters:
            grad = param.grad
            if grad is None:
                continue

            mask = torch.empty_like(grad, dtype=torch.float32).bernoulli_(
                self.keep_prob
            )
            grad.mul_(mask.to(grad.dtype))
            if scale != 1.0:
                grad.mul_(scale)


__all__ = [
    "ExpertGradientDropout",
    "GateGradientDropout",
]
