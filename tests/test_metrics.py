import numpy as np

from src.cinema_demand.evaluate import interval_metrics, point_metrics


def test_point_metrics_known_values():
    y, p, cap = np.array([10., 20.]), np.array([5., 30.]), np.array([50., 50.])
    m = point_metrics(y, p, cap, cu=2.0, co=1.0)
    assert m["MAE"] == 7.5 and m["Bias"] == 2.5
    assert m["WAPE_pct"] == 100 * 15 / 30
    assert m["Asymmetric_Cost"] == (2.0 * 5 + 1.0 * 10) / 2     # under-forecast costs more


def test_interval_metrics_coverage():
    m = interval_metrics([1, 5, 9], [0, 0, 0], [2, 4, 10], [10, 10, 10])
    assert round(m["coverage_pct"], 2) == 66.67 and m["above_hi_pct"] > 0
