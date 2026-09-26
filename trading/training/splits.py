"""Walk-forward splits with a purge gap and an embargo (constraint 7).

Folds are built on the sorted **unique timestamps** of the data, so a dataset that
stacks several symbols is split by time and never by row position. Nothing is ever
shuffled: every training row is earlier than every validation row of its fold.

Between the end of training and the start of validation there are two gaps:

``purge``
    at least the label horizon. A training label at bar *t* is built from prices up
    to *t + horizon*; without the purge the last training labels would be computed
    from validation prices. :func:`walk_forward` refuses a purge shorter than the
    horizon it is told about.
``embargo``
    an optional extra buffer (sklearn's ``gap``) against serial correlation: the
    last training bars and the first validation bars are near-copies of each other
    even when no label overlaps, which flatters validation scores.

Validation blocks tile the most recent part of the data, oldest first. With
``expanding=True`` each fold trains on everything before its gap; otherwise on a
rolling window of ``train_size`` timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    pass


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray  # row positions
    validation: np.ndarray  # row positions
    train_times: tuple[pd.Timestamp, pd.Timestamp]
    validation_times: tuple[pd.Timestamp, pd.Timestamp]
    purge: int
    embargo: int

    def describe(self) -> dict[str, object]:
        return {
            "fold": self.index,
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "train_start": self.train_times[0].isoformat(),
            "train_end": self.train_times[1].isoformat(),
            "validation_start": self.validation_times[0].isoformat(),
            "validation_end": self.validation_times[1].isoformat(),
            "purge": self.purge,
            "embargo": self.embargo,
        }


def walk_forward(
    timestamps: Sequence[pd.Timestamp] | pd.Series | np.ndarray,
    *,
    n_folds: int = 5,
    purge: int,
    embargo: int = 0,
    horizon: int | None = None,
    min_train: int = 200,
    validation_size: int | None = None,
    expanding: bool = True,
    train_size: int | None = None,
) -> list[Fold]:
    """Folds over the rows whose timestamps are given (one entry per row).

    ``purge``, ``embargo``, ``min_train``, ``validation_size`` and ``train_size`` are
    counted in unique timestamps (bars).
    """
    if horizon is not None and purge < horizon:
        raise LeakageError(
            f"purge of {purge} bars is shorter than the label horizon of {horizon}: "
            "training labels would be built from validation prices"
        )
    if n_folds < 1 or purge < 0 or embargo < 0:
        raise ValueError("n_folds must be >= 1 and purge/embargo >= 0")
    if not expanding and not train_size:
        raise ValueError("a rolling window needs train_size")

    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)))
    times = np.sort(ts.unique())
    position = pd.Series(np.arange(len(times)), index=times)
    row_time = position.loc[ts.to_numpy()].to_numpy()  # time position of every row
    n = len(times)
    gap = purge + embargo
    if validation_size is None:
        validation_size = (n - min_train - gap) // n_folds
    if validation_size < 1 or n - n_folds * validation_size - gap < min_train:
        raise ValueError(
            f"{n} bars cannot hold {n_folds} validation blocks of {validation_size} "
            f"plus a {gap}-bar gap and {min_train} training bars"
        )

    folds = []
    first = n - n_folds * validation_size
    for k in range(n_folds):
        v0 = first + k * validation_size
        v1 = n if k == n_folds - 1 else v0 + validation_size
        t1 = v0 - gap
        t0 = 0 if expanding else max(0, t1 - int(train_size or 0))
        if t1 - t0 < min_train:
            raise ValueError(f"fold {k} has only {t1 - t0} training bars (< {min_train})")
        train_rows = np.nonzero((row_time >= t0) & (row_time < t1))[0]
        val_rows = np.nonzero((row_time >= v0) & (row_time < v1))[0]
        folds.append(
            Fold(
                index=k,
                train=train_rows,
                validation=val_rows,
                train_times=(pd.Timestamp(times[t0]), pd.Timestamp(times[t1 - 1])),
                validation_times=(pd.Timestamp(times[v0]), pd.Timestamp(times[v1 - 1])),
                purge=purge,
                embargo=embargo,
            )
        )
    check_no_leakage(folds, ts, purge=purge)
    return folds


def check_no_leakage(folds: Sequence[Fold], timestamps: pd.Series, *, purge: int) -> None:
    """Every training row precedes its validation block by more than ``purge`` bars."""
    times = np.sort(timestamps.unique())
    position = pd.Series(np.arange(len(times)), index=times)
    row_time = position.loc[timestamps.to_numpy()].to_numpy()
    for fold in folds:
        if not len(fold.train) or not len(fold.validation):
            raise LeakageError(f"fold {fold.index} is empty")
        last_train = row_time[fold.train].max()
        first_val = row_time[fold.validation].min()
        if first_val - last_train <= purge:
            raise LeakageError(
                f"fold {fold.index}: only {first_val - last_train - 1} bars between training "
                f"and validation, purge is {purge}"
            )
        if np.intersect1d(fold.train, fold.validation).size:
            raise LeakageError(f"fold {fold.index}: a row is in both training and validation")


def holdout_split(
    timestamps: pd.Series, *, holdout: int, purge: int, embargo: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(training rows, holdout rows): the last ``holdout`` bars are held out, and the
    ``purge + embargo`` bars before them belong to neither."""
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)))
    times = np.sort(ts.unique())
    if holdout < 1 or holdout + purge + embargo >= len(times):
        raise ValueError(f"cannot hold out {holdout} of {len(times)} bars")
    position = pd.Series(np.arange(len(times)), index=times)
    row_time = position.loc[ts.to_numpy()].to_numpy()
    h0 = len(times) - holdout
    train = np.nonzero(row_time < h0 - purge - embargo)[0]
    held = np.nonzero(row_time >= h0)[0]
    return train, held
