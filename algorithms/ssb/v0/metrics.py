"""Historical V0 instrumentation only.

V1 and later intentionally do not use this collector.
"""
from collections import defaultdict

_metrics = defaultdict(list)


def add(key, value):
    _metrics[key].append(value)


def flush():
    data = dict(_metrics)
    _metrics.clear()
    return data


def peek():
    return dict(_metrics)
