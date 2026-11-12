import torch

from sippo.actor_update import SippoPolicyUpdater


def test_collective_device_follows_backend(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *a, **k: "gloo")
    assert SippoPolicyUpdater._collective_device() == "cpu"
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *a, **k: "cpu:gloo,npu:hccl")
    assert SippoPolicyUpdater._collective_device() == "cpu"  # gloo serves CPU tensors in the mixed mapping
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *a, **k: "hccl")
    monkeypatch.setattr("sippo.actor_update.get_device_name", lambda: "npu")
    monkeypatch.setattr("sippo.actor_update.get_device_id", lambda: 3)
    assert SippoPolicyUpdater._collective_device() == torch.device("npu", 3)  # pure HCCL needs a device tensor