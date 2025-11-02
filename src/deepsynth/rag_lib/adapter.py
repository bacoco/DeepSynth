"""Helpers for loading and caching LoRA adapters for the decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

try:  # pragma: no cover - torch may be unavailable in lightweight environments
    import torch
except ImportError:  # pragma: no cover - fallback when torch is absent
    torch = None  # type: ignore[assignment]
    if TYPE_CHECKING:  # pragma: no cover
        import torch as torch_typing  # type: ignore

AdapterLoader = Callable[[Any, str], Any]


@dataclass(slots=True, kw_only=True)
class LoRAAdapterManager:
    """Load and attach LoRA adapters to a decoder model on demand."""

    base_model: Any
    adapter_path: str
    loader: AdapterLoader | None = None
    device: Optional["torch.device"] = None
    _cached_model: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = self._default_loader
        if self.device is None and hasattr(self.base_model, "device"):
            self.device = getattr(self.base_model, "device")  # type: ignore[assignment]
        
    def ensure_loaded(self) -> Any:
        """Load the adapter if needed and return the adapter-enhanced model."""

        if self._cached_model is not None:
            return self._cached_model
        model = self.loader(self.base_model, self.adapter_path)
        if self.device is not None and hasattr(model, "to"):
            model = model.to(self.device)
        self._cached_model = model
        return model

    def apply_to_decoder(self, decoder: Any) -> None:
        """Ensure the decoder uses the adapter-loaded model before generation."""

        if not hasattr(decoder, "model"):
            raise AttributeError("Decoder object must expose a 'model' attribute")
        if getattr(decoder, "model", None) is self._cached_model and self._cached_model is not None:
            return
        decoder.model = self.ensure_loaded()

    # ------------------------------------------------------------------
    def _default_loader(self, base_model: Any, adapter_path: str) -> Any:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - dependency provided in runtime env
            raise RuntimeError("peft is required to load LoRA adapters") from exc

        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        return model


__all__ = ["LoRAAdapterManager"]
