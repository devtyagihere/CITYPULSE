"""Unit tests for router.py (A*, time-dependent A*, haversine, reroute gate)."""

from router import _astar, _haversine_m, _horizon_idx, _td_astar, should_reroute


def test_haversine_known_distance():
    # Times Square to Empire State Building (~1.05 km)
    d = _haversine_m(40.7580, -73.9855, 40.7484, -73.9857)
    assert 1000.0 < d < 1150.0


def test_should_reroute_threshold():
    assert should_reroute(1000.0, 899.0, min_improvement=0.10) is True
    assert should_reroute(1000.0, 900.0, min_improvement=0.10) is False
    assert should_reroute(1000.0, 950.0, min_improvement=0.10) is False
    assert should_reroute(0.0, 500.0) is True


def test_horizon_idx_buckets():
    assert _horizon_idx(0.0) == 0
    assert _horizon_idx(899.9) == 0
    assert _horizon_idx(900.0) == 1
    assert _horizon_idx(1800.0) == 2
    assert _horizon_idx(2700.0) == 3


def test_astar_finds_lowest_cost_path():
    coords = {
        1: (40.750, -73.990),
        2: (40.751, -73.990),
        3: (40.752, -73.990),
    }
    adj = {
        1: [(2, 10.0, 10.0, 100.0), (3, 50.0, 50.0, 200.0)],
        2: [(3, 15.0, 15.0, 100.0)],
    }
    path, cost_s, time_s, dist_m = _astar(adj, coords, 1, 3)
    assert path == [1, 2, 3]
    assert cost_s == 25.0
    assert time_s == 25.0
    assert dist_m == 200.0


def test_td_astar_switches_horizon_on_elapsed_time():
    coords = {
        1: (40.750, -73.990),
        2: (40.751, -73.990),
        3: (40.752, -73.990),
    }
    # First hop takes 1000s (>= 900s -> horizon index 1 at node 2)
    adj = {
        1: [(2, (1000.0, 1000.0, 1000.0, 1000.0), (1000.0, 1000.0, 1000.0, 1000.0), 100.0)],
        2: [(3, (50.0, 20.0, 30.0, 40.0), (50.0, 20.0, 30.0, 40.0), 100.0)],
    }
    path, cost_s, time_s, _ = _td_astar(adj, coords, 1, 3)
    assert path == [1, 2, 3]
    assert cost_s == 1020.0
    assert time_s == 1020.0
