def test_config_loads_phase0_smoke_yaml() -> None:
    from pathlib import Path

    from chartai.core.config import load_experiment_config

    path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "phase0_smoke.yaml"
    cfg = load_experiment_config(path)
    assert cfg.experiment_id == "phase0_smoke"
    assert cfg.state.use_incomplete_higher_tf_bars is False
    assert cfg.state.fusion is None
    assert cfg.market.sigma_timeframe is None
