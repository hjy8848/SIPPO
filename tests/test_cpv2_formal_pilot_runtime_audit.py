from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_audit_tracks_current_cp_protocol_version():
    text = (
        ROOT
        / "exp"
        / "evaluate_q25i_cpv2_sparsepsc_step50_100_formalpilot175_zju47_gpu6_r1.sh"
    ).read_text(encoding="utf-8")
    assert "from qa_manager.ROSA.contract_protocol import CONTRACT_PROTOCOL_VERSION" in text
    assert "cp_rows += version == CONTRACT_PROTOCOL_VERSION" in text
    assert '"expected_cp_runtime_version": CONTRACT_PROTOCOL_VERSION' in text
    assert 'protocol.get("version") == "cp-dagent-v1"' not in text
