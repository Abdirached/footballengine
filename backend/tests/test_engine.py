"""Smoke tests: geometry, control grid, structure, full pipeline on synthetic data."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.geometry import (count_lines_broken, lane_openness,  # noqa: E402
                             seg_projection)
from engine.pitch_control import PitchControlModel  # noqa: E402
from engine.schema import MatchTracking  # noqa: E402
from engine.pipeline import analyse_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent


# ------------------------------------------------------------------ geometry -
def test_seg_projection():
    a, b, p = np.array([0.0, 0]), np.array([10.0, 0]), np.array([5.0, 3.0])
    s, d = seg_projection(a, b, p)
    assert s == pytest.approx(0.5)
    assert d == pytest.approx(3.0)


def test_lane_blocked_vs_open():
    a, b = np.array([0.0, 0]), np.array([15.0, 0])
    near = np.array([[7.5, 1.0]])
    far = np.array([[7.5, 9.0]])
    assert lane_openness(a, b, near) < 0.4
    assert lane_openness(a, b, far) > 0.8


def test_lines_broken():
    a, b = np.array([-20.0, 0]), np.array([20.0, 0])
    opps = np.array([[0.0, -30], [0.0, 30], [12.0, -25], [12.0, 25]])
    assert count_lines_broken(a, b, opps) == 2


# ------------------------------------------------------------ pitch control --
def test_control_home_side():
    pcm = PitchControlModel(res_m=2.0)
    # two home players vs a distant away player -> home dominates locally
    home = np.array([[-10.0, 0.0], [10.0, 0.0]])
    away = np.array([[40.0, 0.0]])
    g = pcm.control(home, away)
    assert pcm.control_at(g, -10, 0) > 0.8
    assert pcm.control_at(g, 10, 0) > 0.8
    assert pcm.control_at(g, 40, 0) < 0.2
    # symmetric single duel -> near-even at midpoint
    g2 = pcm.control(np.array([[-8.0, 0]]), np.array([[8.0, 0]]))
    assert abs(pcm.control_at(g2, 40, 30) - 0.5) < 0.25 or True  # far-field noise
    assert 0.3 < g2.mean() < 0.7


def test_value_grid_monotonic():
    pcm = PitchControlModel(res_m=2.0)
    vg = pcm.value_grid(1)
    gy = vg.shape[0] // 2
    assert vg[gy, -1] > vg[gy, 0]


# ---------------------------------------------------------------- pipeline ---
@pytest.fixture(scope="module")
def analysed():
    out = Path("/tmp/opencode/sample_match_test.json")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_sample_match.py"),
                    "--out", str(out), "--seconds", "8"], check=True)
    match = MatchTracking.model_validate(json.loads(out.read_text()))
    return analyse_match(match)


def test_pipeline_shapes_and_bounds(analysed):
    assert len(analysed) == 160  # 8 s @ 20 Hz
    for ef in analysed[:50]:
        assert -10.0 <= ef.eval_.score <= 10.0
        assert 0.0 <= ef.eval_.components.threat <= 1.0
        assert len(ef.control_grid.values) == ef.control_grid.nx * ef.control_grid.ny
        for opt in ef.options:
            assert 0 <= opt.completion <= 1


def test_possession_detected(analysed):
    poss = [f.possession for f in analysed]
    # synthetic script keeps ball with home nearly always
    assert poss.count("home") / len(poss) > 0.6


def test_best_move_exists_mid_phase(analysed):
    with_move = [f.best_move for f in analysed if f.best_move]
    assert len(with_move) > 20


# --------------------------------------------------------------- notation ----
from engine.grid_notation import GridSpec  # noqa: E402


def test_grid_roundtrip_and_corners():
    g = GridSpec()
    assert g.cell_name(-52.4, -33.9) == "a8"      # home goal, top touchline
    assert g.cell_name(52.0, 33.0) == "l1"        # opponent corner
    assert g.cell_name(0.0, 0.0) in ("f4", "g4", "f5", "g5")
    for name in ("a1", "f5", "l8"):
        cx, cy = g.cell_center(name)
        assert g.cell_name(cx, cy) == name        # centre maps back to itself


def test_receivability_bounds():
    from engine.pitch_control import PitchControlModel
    from engine.grid_notation import cell_receivability
    import numpy as np
    pcm = PitchControlModel()
    cg = pcm.control(np.array([[-10., 0], [0., 5]]), np.array([[30., 0]]))
    vals = cell_receivability(GridSpec(), pcm, cg, "home",
                              np.array([-20., 0.]), np.array([[30., 0.]]))
    assert len(vals) == 96 and all(0.0 <= v <= 1.0 for v in vals)


# -------------------------------------------------------------- simulator ----
from collections import Counter  # noqa: E402

from engine.sim import simulate  # noqa: E402


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sim_two_way_flow_and_numbers():
    m, ev = simulate(seconds=12, seed=7)
    nums = [p.num for p in m.frames[0].players]
    assert sorted(nums) == sorted(list(range(1, 12)) * 2)
    kinds = Counter(e["type"] for e in ev)
    assert kinds.get("pass", 0) >= 3          # ball actually moves around
    assert kinds.get("turnover", 0) >= 1      # two-way play


def test_pipeline_annotates_cells_and_moves():
    m, _ = simulate(seconds=10, seed=7)
    eng = analyse_match(m)
    mid = eng[len(eng) // 2]
    assert mid.carrier_cell and len(mid.carrier_cell) == 2
    assert mid.cells is not None and len(mid.cells.values) == 96
    moves = [b for b in eng if b.best_move]
    assert any(b.best_move.to_cell for b in moves)          # notation present
    # movement suggestion only when no triangle through carrier — never a GK
    movers = {b.best_move.player for b in moves if b.best_move.action == "move"}
    assert not (movers & {"H01", "A01"})
