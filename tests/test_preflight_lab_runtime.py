from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_lab_runtime.py"
PROFILES = ROOT / "scripts" / "lab_runtime_profiles.json"


def load_module():
    spec = importlib.util.spec_from_file_location("preflight_lab_runtime", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiles_cover_known_runtime_boundaries() -> None:
    document = json.loads(PROFILES.read_text(encoding="utf-8"))
    profiles = document["profiles"]
    assert document["schema"] == "jade-lab-runtime-profiles-v1"
    assert profiles["shoppingbench-official"]["probes"][0]["type"] == "lucene"
    assert profiles["shoppingbench-official"]["prepend_ld_library_path"]
    assert "zju-57" in profiles["qwen-vllm-blackwell"]["allowed_hosts"]
    zju49 = profiles["qwen-vllm-zju49-rtx3090-eval"]
    assert zju49["allowed_hosts"] == ["zju-49"]
    assert zju49["gpu_name_pattern"] == "NVIDIA GeForce RTX 3090"
    assert "vllm" in zju49["python_imports"]
    assert profiles["a800-replacement-shared-environment"]["status"] == "blocked"


def test_expand_text_fails_closed_on_unknown_or_unsafe_values() -> None:
    module = load_module()
    assert module.expand_text("${ROOT}/x", {"ROOT": "/tmp"}) == "/tmp/x"
    with pytest.raises(module.PreflightFailure, match="Unresolved variable"):
        module.expand_text("${MISSING}/x", {})
    with pytest.raises(module.PreflightFailure, match="Unsafe rendered value"):
        module.expand_text("bad\rpath", {})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0=GPU-abc", (0, "GPU-abc")),
        ("7=GPU-123", (7, "GPU-123")),
    ],
)
def test_parse_gpu_spec(value: str, expected: tuple[int, str]) -> None:
    module = load_module()
    assert module.parse_gpu_spec(value) == expected


@pytest.mark.parametrize("value", ["0", "x=GPU-abc", "-1=GPU-abc", "0=abc"])
def test_parse_gpu_spec_rejects_unsafe_values(value: str) -> None:
    module = load_module()
    with pytest.raises(Exception):
        module.parse_gpu_spec(value)


def test_required_paths_distinguish_file_directory_and_executable(tmp_path: Path) -> None:
    module = load_module()
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()
    executable = tmp_path / "run"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    assert module.check_required_path({"path": str(file_path), "kind": "file"})["status"] == "passed"
    assert module.check_required_path({"path": str(directory), "kind": "directory"})["status"] == "passed"
    with patch.object(module.os, "access", return_value=True):
        assert module.check_required_path({"path": str(executable), "kind": "executable"})["status"] == "passed"
    with pytest.raises(module.PreflightFailure):
        module.check_required_path({"path": str(file_path), "kind": "directory"})


def test_blocked_profile_fails_before_any_runtime_probe() -> None:
    module = load_module()
    with pytest.raises(module.PreflightFailure, match="broken shared environment"):
        module.run_slow(
            {"status": "blocked", "blocked_reason": "broken shared environment"},
            {},
        )


def test_list_profiles_does_not_require_gpu() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-profiles"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "shoppingbench-official" in json.loads(result.stdout)


def test_gpu_inventory_fails_closed_when_compute_query_fails() -> None:
    module = load_module()
    gpu_result = subprocess.CompletedProcess(
        [],
        0,
        stdout="0, GPU-abc, NVIDIA A800-SXM4-80GB, 81920, 0\n",
        stderr="",
    )
    process_result = subprocess.CompletedProcess([], 1, stdout="", stderr="NVML unavailable")
    with patch.object(module, "run", side_effect=[gpu_result, process_result]):
        with pytest.raises(module.PreflightFailure, match="compute-process query failed"):
            module.gpu_inventory()


def test_gpu_inventory_scopes_compute_queries_to_claimed_gpus() -> None:
    module = load_module()
    gpu_result = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-busy, NVIDIA A100-PCIE-40GB, 40960, 100\n"
            "2, GPU-free, NVIDIA A100-SXM4-40GB, 40960, 0\n"
        ),
        stderr="",
    )
    target_process_result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with patch.object(module, "run", side_effect=[gpu_result, target_process_result]) as mocked:
        gpus, processes = module.gpu_inventory([2])

    assert [gpu["index"] for gpu in gpus] == [0, 2]
    assert processes == []
    assert mocked.call_args_list[1].args[0][1] == "--id=2"


def test_gpu_inventory_accepts_exact_nvml_idle_placeholder() -> None:
    module = load_module()
    gpu_result = subprocess.CompletedProcess(
        [],
        0,
        stdout="0, GPU-free, NVIDIA RTX A6000, 49140, 1\n",
        stderr="",
    )
    process_result = subprocess.CompletedProcess(
        [],
        0,
        stdout="GPU-free, [N/A], [N/A], [N/A]\n",
        stderr="",
    )
    with patch.object(module, "run", side_effect=[gpu_result, process_result]):
        _, processes = module.gpu_inventory([0])

    assert processes == []


def test_gpu_inventory_rejects_partial_nvml_placeholder() -> None:
    module = load_module()
    gpu_result = subprocess.CompletedProcess(
        [],
        0,
        stdout="0, GPU-free, NVIDIA RTX A6000, 49140, 1\n",
        stderr="",
    )
    process_result = subprocess.CompletedProcess(
        [],
        0,
        stdout="GPU-free, [N/A], python, [N/A]\n",
        stderr="",
    )
    with patch.object(module, "run", side_effect=[gpu_result, process_result]):
        with pytest.raises(module.PreflightFailure, match="Unparseable nvidia-smi"):
            module.gpu_inventory([0])


def test_optional_gpu_memory_accepts_nvml_na() -> None:
    module = load_module()
    assert module.parse_optional_int("123") == 123
    assert module.parse_optional_int("[N/A]") is None
