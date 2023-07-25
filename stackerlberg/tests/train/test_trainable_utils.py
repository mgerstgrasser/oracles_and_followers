from stackerlberg.train.utils import (
    trainable_distance_from_target,
    trainable_filter_key_on_condition,
    trainable_penalty_if_follower_not_best_responding,
    trainable_rewrite_key,
    trainable_sliding_window_over_key,
    trainable_stop_updating_key_on_condition,
)


def test_sliding_window():
    """Tests the sliding window trainable wrapper"""

    def constant_trainable(config):
        for value in config.get("values", [0, 10, 10, 10]):
            yield {"test_value": value}

    sliding_window_trainable = trainable_sliding_window_over_key(constant_trainable, "test_value", window_size=2)

    results = list(sliding_window_trainable({"values": [0, 10, 20, 30]}))

    assert results[0]["test_value"] == 0
    assert results[1]["test_value"] == 10
    assert results[2]["test_value"] == 20
    assert results[3]["test_value"] == 30
    assert results[0]["mean_window"] == 0
    assert results[1]["mean_window"] == 5
    assert results[2]["mean_window"] == 15
    assert results[3]["mean_window"] == 25

    sliding_window_trainable = trainable_sliding_window_over_key(constant_trainable, "test_value", window_size=3, new_key="new_key")
    results = list(sliding_window_trainable({"values": [0, 10, 20, 30]}))
    assert results[0]["new_key"] == 0
    assert results[1]["new_key"] == 5
    assert results[2]["new_key"] == 10
    assert results[3]["new_key"] == 20


def test_sliding_window_phase():
    """Tests follower best response checking wrapper."""

    def trainable(config):
        for result in config.get("results"):
            yield result

    wrapped_trainable = trainable_sliding_window_over_key(
        trainable, "leader_perf", window_size=2, new_key="mean_window", reset_on_new_phase_key="phase"
    )
    # wrapped_trainable = (trainable, "leader_perf", "follower_perf", phase=2, phase_key="phase", leader_new_key="leader_perf_penalized", weight=2)
    config = {
        "results": [
            {"leader_perf": 1, "follower_perf": 1, "phase": 1},
            {"leader_perf": 2, "follower_perf": 1, "phase": 1},
            {"leader_perf": 7, "follower_perf": 2, "phase": 2},
            {"leader_perf": 1, "follower_perf": 3, "phase": 3},
        ]
    }
    results = list(wrapped_trainable(config))
    assert results[0]["mean_window"] == 1
    assert results[1]["mean_window"] == 1.5
    assert results[2]["mean_window"] == 7
    assert results[3]["mean_window"] == 1


def test_stop_updating_key_on_condition():
    """Tests the stop-update trainable wrapper"""

    def constant_trainable(config):
        for value in config.get("values", [(0, 0), (10, 0), (20, 1), (20, 2)]):
            yield {"test_value": value[0], "condition_value": value[1]}

    sliding_window_trainable = trainable_stop_updating_key_on_condition(
        constant_trainable, "test_value", lambda result: result["condition_value"] > 0
    )

    results = list(sliding_window_trainable({"values": [(0, 0), (10, 0), (20, 1), (20, 2)]}))

    assert results[0]["test_value"] == 0
    assert results[1]["test_value"] == 10
    assert results[2]["test_value"] == 10
    assert results[3]["test_value"] == 10

    sliding_window_trainable = trainable_stop_updating_key_on_condition(
        constant_trainable, "test_value", lambda result: result["condition_value"] > 0, new_key="new_key"
    )

    results = list(sliding_window_trainable({"values": [(0, 0), (10, 0), (20, 1), (20, 2)]}))

    assert results[0]["new_key"] == 0
    assert results[1]["new_key"] == 10
    assert results[2]["new_key"] == 10
    assert results[3]["new_key"] == 10

    sliding_window_trainable = trainable_stop_updating_key_on_condition(
        constant_trainable,
        "nonexistent_value",
        lambda result: result["condition_value"] > 0,
        default_value=5.0,
        new_key="new_key",
    )

    results = list(sliding_window_trainable({"values": [(0, 0), (10, 0), (20, 1), (20, 2)]}))

    assert results[0]["new_key"] == 5.0
    assert results[1]["new_key"] == 5.0
    assert results[2]["new_key"] == 5.0
    assert results[3]["new_key"] == 5.0


def test_filter_key_on_condition():
    """Tests the filter trainable wrapper"""

    def constant_trainable(config):
        for value in config.get("values", [(0, 0), (10, 0), (20, 1), (20, 2)]):
            yield {"test_value": value[0], "condition_value": value[1]}

    filtered_trainable = trainable_filter_key_on_condition(constant_trainable, "test_value", lambda result: result["condition_value"] == 1)
    results = list(filtered_trainable({"values": [(0, 0), (10, 0), (20, 1), (20, 2)]}))

    assert "test_value" not in results[0]
    assert "test_value" not in results[1]
    assert results[2]["test_value"] == 20
    assert "test_value" not in results[3]

    filtered_trainable = trainable_filter_key_on_condition(
        constant_trainable, "test_value", lambda result: result["condition_value"] == 1, new_key="new_key"
    )
    results = list(filtered_trainable({"values": [(0, 0), (10, 0), (20, 1), (20, 2)]}))

    assert "new_key" not in results[0]
    assert "new_key" not in results[1]
    assert results[2]["new_key"] == 20
    assert "new_key" not in results[3]
    assert results[0]["test_value"] == 0
    assert results[1]["test_value"] == 10
    assert results[2]["test_value"] == 20
    assert results[3]["test_value"] == 20


def test_util_integration():
    """Tests several of the utils together"""

    def constant_trainable(config):
        for value in config.get("values", [(0, 0), (10, 0), (20, 1), (20, 2)]):
            yield {"test_value": value[0], "condition_value": value[1]}

    sliding_window_trainable = trainable_filter_key_on_condition(
        constant_trainable, "test_value", lambda result: result["condition_value"] == 1, new_key="filtered_key"
    )
    sliding_window_trainable = trainable_sliding_window_over_key(
        sliding_window_trainable, "filtered_key", window_size=6, new_key="sliding_window_key"
    )
    results = list(
        sliding_window_trainable(
            {
                "values": [
                    (0, 0),
                    (10, 0),
                    (0, 0),
                    (10, 0),
                    (20, 1),
                    (30, 1),
                    (30, 1),
                    (30, 1),
                    (20, 2),
                    (20, 2),
                    (20, 2),
                ]
            }
        )
    )
    assert "filtered_key" not in results[0]
    assert "new_key" not in results[0]
    assert results[0]["test_value"] == 0
    assert results[4]["test_value"] == 20
    assert results[4]["filtered_key"] == 20
    assert results[4]["sliding_window_key"] == 20
    assert results[5]["sliding_window_key"] == 25
    assert results[7]["test_value"] == 30
    assert results[7]["filtered_key"] == 30
    assert results[7]["sliding_window_key"] == 27.5
    assert "filtered_key" not in results[8]
    assert "new_key" not in results[8]
    assert results[8]["test_value"] == 20


def test_follower_best_response():
    """Tests follower best response checking wrapper."""

    def trainable(config):
        for result in config.get("results"):
            yield result

    wrapped_trainable = trainable_penalty_if_follower_not_best_responding(
        trainable,
        "leader_perf",
        "follower_perf",
        phase=2,
        phase_key="phase",
        leader_new_key="leader_perf_penalized",
        weight=2,
    )
    config = {
        "results": [
            {"leader_perf": 1, "follower_perf": 1, "phase": 1},
            {"leader_perf": 1, "follower_perf": 1, "phase": 2},
            {"leader_perf": 1, "follower_perf": 2, "phase": 2},
            {"leader_perf": 1, "follower_perf": 3, "phase": 2},
        ]
    }
    results = list(wrapped_trainable(config))
    assert results[0]["leader_perf_penalized"] == 1
    assert results[1]["leader_perf_penalized"] == 1
    assert results[2]["leader_perf_penalized"] == -1
    assert results[3]["leader_perf_penalized"] == -3


def test_follower_best_response_integration():
    """Tests follower best response checking wrapper."""

    def trainable(config):
        for result in config.get("results"):
            yield result

    trainable = trainable_sliding_window_over_key(
        trainable, "leader_perf", window_size=4, new_key="leader_perf_window", reset_on_new_phase_key="phase"
    )
    trainable = trainable_sliding_window_over_key(
        trainable, "follower_perf", window_size=4, new_key="follower_perf_window", reset_on_new_phase_key="phase"
    )
    trainable = trainable_penalty_if_follower_not_best_responding(
        trainable,
        "leader_perf_window",
        "follower_perf_window",
        phase=2,
        phase_key="phase",
        leader_new_key="leader_perf_penalized",
        weight=2,
    )
    config = {
        "results": [
            {"leader_perf": 1, "follower_perf": 1, "phase": 1},
            {"leader_perf": 1, "follower_perf": 2, "phase": 1},
            {"leader_perf": 1, "follower_perf": 3, "phase": 1},
            {"leader_perf": 1, "follower_perf": 3, "phase": 2},
            {"leader_perf": 1, "follower_perf": 3, "phase": 2},
            {"leader_perf": 1, "follower_perf": 6, "phase": 2},
        ]
    }
    results = list(trainable(config))
    assert results[0]["leader_perf_penalized"] == 1
    assert results[1]["leader_perf_penalized"] == 1
    assert results[2]["follower_perf_window"] == 2
    assert results[3]["follower_perf_window"] == 3
    assert results[3]["leader_perf_penalized"] == -1
    assert results[5]["follower_perf_window"] == 4
    assert results[5]["leader_perf_penalized"] == -3


def test_target():
    """Tests follower best response checking wrapper."""

    def trainable(config):
        for result in config.get("results"):
            yield result

    wrapped_trainable = trainable_distance_from_target(trainable, "leader_perf", 2.5, new_key="leader_perf_distance", negative=True)
    config = {
        "results": [
            {"leader_perf": 1, "follower_perf": 1, "phase": 1},
            {"leader_perf": 3, "follower_perf": 1, "phase": 2},
        ]
    }
    results = list(wrapped_trainable(config))
    assert results[0]["leader_perf_distance"] == -1.5
    assert results[1]["leader_perf_distance"] == -0.5


def test_rewrite():
    """Tests follower best response checking wrapper."""

    def trainable(config):
        for result in config.get("results"):
            yield result

    wrapped_trainable = trainable_rewrite_key(trainable, "leader_perf", new_key="target")
    config = {
        "results": [
            {"leader_perf": 1, "follower_perf": 1, "phase": 1},
            {"leader_perf": 3, "follower_perf": 1, "phase": 2},
        ]
    }
    results = list(wrapped_trainable(config))
    assert results[0]["target"] == 1
    assert results[1]["target"] == 3
