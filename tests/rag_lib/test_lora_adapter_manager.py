from __future__ import annotations
from dataclasses import dataclass

from deepsynth.rag_lib.adapter import LoRAAdapterManager


class DummyModel:
    def __init__(self) -> None:
        self.loaded_adapters = 0
        self.device = "cpu"

    def to(self, device):  # pragma: no cover - simple pass-through
        self.device = device
        return self

    def eval(self):  # pragma: no cover - no-op for interface compatibility
        return self


@dataclass
class DummyDecoder:
    model: DummyModel


def test_lora_adapter_manager_loads_once() -> None:
    base_model = DummyModel()

    def loader(model: DummyModel, adapter_path: str) -> DummyModel:
        model.loaded_adapters += 1
        assert adapter_path == "adapter"
        return model

    manager = LoRAAdapterManager(base_model=base_model, adapter_path="adapter", loader=loader, device=None)
    decoder = DummyDecoder(model=base_model)

    manager.apply_to_decoder(decoder)
    assert decoder.model.loaded_adapters == 1

    # Subsequent application should reuse cached model without reloading.
    manager.apply_to_decoder(decoder)
    assert decoder.model.loaded_adapters == 1
    assert decoder.model is base_model
