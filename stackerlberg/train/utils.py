from ast import Call
from collections import deque
from typing import Callable, Union


def trainable_sliding_window_over_key(
    trainable_to_wrap: Callable,
    key: Union[str, Callable],
    window_size=100,
    new_key="mean_window",
    reset_on_new_phase_key: str = None,
):
    """Wraps a trainable, and takes a sliding window over one reported value.
    This is to allow hyperparameter tuning for robustness of training rather than just highest final reported value.

    Args:
        trainable_to_wrap (function): The trainable to wrap.
        key (str or callable): The key to take the sliding window over (for top-level keys),
            or a function taking the result dict and returning the value to take the window over.
        window_size (int): The size of the sliding window.

    Returns:
        function: The wrapped trainable.
    """
    # For top-level keys we just grab the key
    if isinstance(key, str):
        get_key = lambda result: result.get(key, None)
    else:
        get_key = key

    def _trainable(config):
        # Set up sliding window
        window = deque(maxlen=window_size)
        if reset_on_new_phase_key is not None:
            phase = 0
        # Iterate over results of the wrapped trainable
        for result in trainable_to_wrap(config):
            # Reset the window when we start a new phase
            if reset_on_new_phase_key is not None:
                if result.get(reset_on_new_phase_key, phase) != phase:
                    window.clear()
                    phase = result[reset_on_new_phase_key]
            # Take the value to take the window over
            window_value = get_key(result)
            # Add it to the window
            if window_value is not None:
                window.append(window_value)
                # Take mean
                result[new_key] = sum(window) / len(window)
            # Yield the result
            yield result

    return _trainable


def trainable_stop_updating_key_on_condition(
    trainable_to_wrap: Callable, key: str, cond: Callable, default_value: float = 0.0, new_key: str = None
):
    """Wraps a trainable, and stops updating a key when a condition is met.
    This is to allow post-training logging (e.g. keep training follower to check equilibrium), without affecting hyperparameter tuning algorithms.

    Args:
        trainable_to_wrap (function): The trainable to wrap.
        key (str): The key to stop updating, only works for top-level keys
        cond (funciton): The condition to stop updating the key, takes the result dict and returns a boolean
        default_value (float): The value to set the key to if it's not yet been returned by the wrapped trainable
        new_key (str): The key to set the value to, if None, will overwrite the original key

    Returns:
        function: The wrapped trainable.
    """

    # If no new key is specified, overwrite the original key
    if new_key is None:
        new_key = key

    def _trainable(config):
        # Set up the default value
        last_val = default_value
        # Iterate over results of the wrapped trainable
        for result in trainable_to_wrap(config):
            if not cond(result):
                # Update last value
                last_val = result.get(key, last_val)
                result[new_key] = result.get(key, last_val)
            # If condition is met, stop updating the key, just return the last value
            else:
                result[new_key] = last_val
            # Yield the result
            yield result

    return _trainable


def trainable_filter_key_on_condition(trainable_to_wrap: Callable, key: str, cond: Callable, new_key: str = None):
    """Wraps a trainable, and filters a result key. If condition is true, key is returned as new_key. Otherwise new_key is removed from results.

    Args:
        trainable_to_wrap (function): The trainable to wrap.
        key (str): The key to stop updating, only works for top-level keys
        cond (funciton): The condition that decides if key should be returned. Takes the result dict and returns a boolean.
        new_key (str): The key to set the value to, if None, will overwrite the original key

    Returns:
        function: The wrapped trainable.
    """

    # If no new key is specified, overwrite the original key
    if new_key is None:
        new_key = key

    def _trainable(config):
        # Iterate over results of the wrapped trainable
        for result in trainable_to_wrap(config):
            # If condition is true, return key as new_key
            if cond(result) and key in result:
                result[new_key] = result[key]
            # Otherwise, don't return new_key (remove key if new_key=key)
            else:
                result.pop(new_key, None)
            # Yield the result
            yield result

    return _trainable


def trainable_penalty_if_follower_not_best_responding(
    trainable_to_wrap: Callable,
    leader_key: str,
    follower_key: str,
    phase: int,
    phase_key: str = "phase",
    leader_new_key: str = None,
    weight: float = 1.0,
):
    """Wraps a trainable, and stops updating a key when a condition is met.
    This is to allow post-training logging (e.g. keep training follower to check equilibrium), without affecting hyperparameter tuning algorithms.

    Args:
        trainable_to_wrap (function): The trainable to wrap.
        leader_key (str): Dict key that gives leader performance.
        follower_key (str): Dict key that gives follower performance.
        phase (int): Phase to start checking follower performance. I.e. phase = 2 means that follower performance is checked after phase 1.
        phase_key (str): Dict key that gives phase.
        leader_new_key (str): The key to set the leader performance to, if None, will overwrite the original key
        follower_new_key (str): The key to set the follower performance to, if None, will overwrite the original key

    Returns:
        function: The wrapped trainable.
    """

    # If no new key is specified, overwrite the original key
    if leader_new_key is None:
        leader_new_key = leader_key

    def _trainable(config):
        follower_perf = 0
        # Iterate over results of the wrapped trainable
        for result in trainable_to_wrap(config):
            if phase_key in result and result[phase_key] < phase:
                # We're in training, so keep track of the follower performance
                follower_perf = result.get(follower_key, follower_perf)
                result[leader_new_key] = result.get(leader_key)
            else:
                # We're in post-train, so check if follower performance is improving
                # If improving, deduct performance delta from leader perf.
                delta = max(0, result.get(follower_key, follower_perf) - follower_perf)
                if leader_key in result:
                    result[leader_new_key] = result.get(leader_key) - weight * delta
            # Yield the result
            yield result

    return _trainable


def trainable_distance_from_target(
    trainable_to_wrap: Callable,
    key: str,
    target: float,
    new_key: str = None,
    negative: bool = True,
):
    """Wraps a trainable, and calculates distance of key from a target
    Args:
        trainable_to_wrap (function): The trainable to wrap.
        key (str): Dict key that gives performance.
        target (float): Target value to calculate distance from.
        new_key (str): The key to set the distance to, if None, will overwrite the original key
        negative (bool): If True, distance is always negative, if False, distance is always positive.

    Returns:
        function: The wrapped trainable.
    """

    # If no new key is specified, overwrite the original key
    if new_key is None:
        new_key = key

    def _trainable(config):
        # Iterate over results of the wrapped trainable
        for result in trainable_to_wrap(config):
            if key in result:
                if negative:
                    result[new_key] = -abs(result[key] - target)
                else:
                    result[new_key] = abs(result[key] - target)
            # Yield the result
            yield result

    return _trainable


def trainable_rewrite_key(
    trainable_to_wrap: Callable,
    key: str,
    new_key: str,
):
    """Wraps a trainable, and copies key to new_key in results
    Args:
        trainable_to_wrap (function): The trainable to wrap.
        key (str): Old key
        new_key (str): New key

    Returns:
        function: The wrapped trainable.
    """

    def _trainable(config):
        # Iterate over results of the wrapped trainable
        for result in trainable_to_wrap(config):
            if key in result:
                result[new_key] = result[key]
            # Yield the result
            yield result

    return _trainable
