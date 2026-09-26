"""Walk-forward splits: purge >= horizon, optional embargo, never shuffled (constraint 7)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading.core.types import IST
from trading.training.splits import LeakageError, check_no_leakage, holdout_split, walk_forward

T0 = datetime(2026, 9, 1, 9, 15, tzinfo=IST)


def stamps(n: int, symbols: int = 1) -> pd.Series:
    times = [T0 + timedelta(minutes=i) for i in range(n)]
    return pd.Series([t for t in times for _ in range(symbols)])


def time_index(ts: pd.Series) -> np.ndarray:
    return (
        pd.Series(np.arange(ts.nunique()), index=np.sort(ts.unique())).loc[ts.to_numpy()].to_numpy()
    )


def test_a_purge_shorter_than_the_horizon_is_refused():
    with pytest.raises(LeakageError, match="shorter than the label horizon"):
        walk_forward(stamps(1000), purge=5, horizon=10)


def test_training_always_precedes_validation_by_the_purge():
    ts = stamps(1000)
    folds = walk_forward(ts, n_folds=4, purge=10, horizon=10, min_train=100)
    t = time_index(ts)
    assert len(folds) == 4
    for fold in folds:
        assert t[fold.train].max() < t[fold.validation].min() - 10
        assert not np.intersect1d(fold.train, fold.validation).size
        assert np.all(np.diff(fold.train) > 0) and np.all(np.diff(fold.validation) > 0)  # in order
    # validation blocks tile the recent data, oldest first, without overlap
    starts = [t[f.validation].min() for f in folds]
    assert starts == sorted(starts) and t[folds[-1].validation].max() == 999


def test_embargo_widens_the_gap():
    ts = stamps(1000)
    plain = walk_forward(ts, n_folds=3, purge=10, min_train=100)
    embargoed = walk_forward(ts, n_folds=3, purge=10, embargo=25, min_train=100)
    t = time_index(ts)
    gap = lambda f: t[f.validation].min() - t[f.train].max() - 1  # noqa: E731
    assert [gap(f) for f in plain] == [10, 10, 10]
    assert [gap(f) for f in embargoed] == [35, 35, 35]
    assert embargoed[0].describe()["embargo"] == 25


def test_several_symbols_are_split_by_time_not_by_row():
    ts = stamps(600, symbols=3)  # three rows per timestamp
    folds = walk_forward(ts, n_folds=3, purge=5, horizon=5, min_train=50)
    for fold in folds:
        train_times = set(ts.iloc[fold.train])
        val_times = set(ts.iloc[fold.validation])
        assert not train_times & val_times
        assert len(fold.train) % 3 == 0 and len(fold.validation) % 3 == 0  # whole timestamps
        assert max(train_times) < min(val_times)


def test_expanding_and_rolling_windows():
    ts = stamps(1000)
    expanding = walk_forward(ts, n_folds=3, purge=10, min_train=100)
    assert all(f.train[0] == 0 for f in expanding)
    assert len(expanding[2].train) > len(expanding[0].train)
    rolling = walk_forward(ts, n_folds=3, purge=10, min_train=100, expanding=False, train_size=200)
    sizes = [len(f.train) for f in rolling]
    assert sizes[1:] == [200, 200] and sizes[0] <= 200  # the first fold has less history
    with pytest.raises(ValueError, match="train_size"):
        walk_forward(ts, n_folds=3, purge=10, expanding=False)


def test_too_little_data():
    with pytest.raises(ValueError):
        walk_forward(stamps(105), n_folds=5, purge=10, min_train=100)


def test_leakage_checker_catches_overlap():
    ts = stamps(300)
    folds = walk_forward(ts, n_folds=2, purge=10, min_train=100)
    bad = folds[0].__class__(**{**folds[0].__dict__, "train": np.arange(0, folds[0].validation[0])})
    with pytest.raises(LeakageError):
        check_no_leakage([bad], ts, purge=10)


def test_holdout_split():
    ts = stamps(500, symbols=2)
    train, held = holdout_split(ts, holdout=100, purge=10, embargo=5)
    t = time_index(ts)
    assert t[held].min() == 400 and len(held) == 200
    assert t[train].max() == 400 - 10 - 5 - 1
    with pytest.raises(ValueError):
        holdout_split(ts, holdout=495, purge=10)
