from collections.abc import Mapping


def update_recursively(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_recursively(d.get(k, {}), v)
        else:
            d[k] = v
    return d
