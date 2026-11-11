from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp/evaluate_dagent_step468_planner_swap_2wiki500_zju57_gpu14_r1.sh"
ZJU55_LAUNCHER = ROOT / "exp/evaluate_dagent_step468_planner_swap_2wiki500_zju55_gpu12_r1.sh"
ZJU55_R2_LAUNCHER = ROOT / "exp/evaluate_dagent_step468_planner_swap_2wiki500_zju55_gpu12_r2.sh"


def test_planner_swap_launcher_is_symmetric_and_2wiki_only():
    script = LAUNCHER.read_text(encoding="utf-8")

    assert 'run_swap "${PURE_MODEL}" "${PSC_MODEL}"' in script
    assert 'run_swap "${PSC_MODEL}" "${PURE_MODEL}"' in script
    assert "FORMAL_DATASETS=2wikimultihopqa" in script
    assert "--dataset 2wikimultihopqa=500" in script
    assert "SPECIALIST_ROUTE_SCOPE=planner" in script
    assert "PLANNER_ROUTE_MODE=all" in script
    assert "PLANNER_FALLBACK_TO_MAIN=0" in script


def test_planner_swap_launcher_freezes_contract_and_protects_resources():
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "export MAX_NODES_PER_TURN=0" in script
    assert 'export PLANNER_PROMPT_MAX_NODES="${PLANNER_PROMPT_MAX_NODES:-5}"' in script
    assert "GPU-2495a41f-0d3f-aee8-4573-c977fb303b54" in script
    assert "GPU-1e9a8566-54a3-eac9-bed5-9aa4b6e59039" in script
    assert "Refusing to overwrite existing evidence" in script
    assert "specialist_generated" in script
    assert 'planner_route") != "specialist"' in script


def test_zju55_wrapper_freezes_two_empty_3090_devices():
    script = ZJU55_LAUNCHER.read_text(encoding="utf-8")

    assert "EXPECTED_HOST=zju-55" in script
    assert "RUN_LOCATION_TAG=zju55-gpu1-2" in script
    assert "MAIN_GPU=1" in script
    assert "GPU-541d9661-d0a4-9b0b-e088-05737bc62ac2" in script
    assert "PLANNER_GPU=2" in script
    assert "GPU-3da27f3a-3c2a-7d6f-503d-3fc0834b1ef6" in script
    assert "MAIN_MEM_UTIL=0.45" in script
    assert "PLANNER_MEM_UTIL=0.45" in script


def test_zju55_r2_uses_fresh_paths_and_ports_after_r1_metadata_failure():
    script = ZJU55_R2_LAUNCHER.read_text(encoding="utf-8")

    assert "RUN_REVISION=r2" in script
    assert "PSC_PLANNER_MAIN_PORT=12910" in script
    assert "PSC_PLANNER_API_PORT=12911" in script
    assert "PURE_PLANNER_MAIN_PORT=13010" in script
    assert "PURE_PLANNER_API_PORT=13011" in script
