from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp" / "evaluate_shoppingbench_visible_atomic_shop_a800_repl_r1.sh"


def test_visible_atomic_launcher_freezes_shop_only_single_variable_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--task shop" in text
    assert "--limit 16" in text
    assert "--planner-output-override-manifest" in text
    assert "fb3de22f1b598d374d3bf9aedc78929da3a578a7935d57f5eef345a24dc5deb7" in text
    assert "0ab8f5a8bcc3eb7d3ddf1437e966c6d302519199bb39fa5c305df1b636b2ea55" in text
    assert "compare_shoppingbench_visible_atomic_shop.py" in text
    assert "voucher" not in text.lower()
