"""Spike-train rate and peri-stimulus analyses."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import core

__all__ = ["PSTHResult", "psth"]


@dataclass(frozen=True)
class PSTHResult:
    """Result of a trial-aligned peri-stimulus time histogram estimate.

    Attributes
    ----------
    rate : AnalogSignalArray
        Average firing rate for every input series in Hz. Its time axis is
        relative to the start of each trial support interval.
    standard_error : AnalogSignalArray, optional
        Pointwise standard error of ``rate`` in Hz, or None when no error
        estimate was requested.
    n_trials : int
        Number of support intervals contributing to the estimate.
    sigma : float
        Standard deviation of the Gaussian kernel, in seconds.
    """

    rate: core.AnalogSignalArray
    standard_error: Optional[core.AnalogSignalArray]
    n_trials: int
    sigma: float


def _validate_positive_scalar(value, name):
    """Return a positive finite scalar as a float."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a positive scalar")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _relative_spikes_by_trial(spiketrain, trial_starts, trial_stops):
    """Collect each series' spikes in every support interval, relative to start."""
    spikes_by_trial = []
    for start, stop in zip(trial_starts, trial_stops):
        trial = []
        for series in spiketrain.data:
            series = np.asarray(series)
            mask = (series >= start) & (series < stop)
            trial.append(series[mask] - start)
        spikes_by_trial.append(trial)
    return spikes_by_trial


def _gaussian_trial_rates(spikes_by_trial, times, sigma):
    """Evaluate Gaussian rate estimates for every trial, series, and time."""
    n_trials = len(spikes_by_trial)
    n_series = len(spikes_by_trial[0])
    rates = np.zeros((n_trials, n_series, times.size), dtype=float)
    normalization = 1 / (sigma * np.sqrt(2 * np.pi))

    for trial_idx, trial in enumerate(spikes_by_trial):
        for series_idx, spikes in enumerate(trial):
            if spikes.size == 0:
                continue
            offsets = times[:, np.newaxis] - spikes[np.newaxis, :]
            rates[trial_idx, series_idx] = (
                np.exp(-0.5 * (offsets / sigma) ** 2).sum(axis=1) * normalization
            )
    return rates


def _make_rate_array(data, times, fs, duration, labels):
    """Construct a rate AnalogSignalArray with trial-relative support."""
    return core.AnalogSignalArray(
        data,
        timestamps=times,
        fs=fs,
        support=core.EpochArray([[0, duration]]),
        labels=labels,
    )


def psth(
    spiketrain,
    *,
    ds=0.001,
    sigma=0.05,
    error="none",
    n_boot=1000,
    random_state=None,
):
    """Estimate a trial-aligned Gaussian-kernel peri-stimulus time histogram.

    Each support interval of ``spiketrain`` is treated as a trial and is aligned
    to its own start time. All intervals must have equal duration. The returned
    time axis is therefore relative to trial onset, rather than absolute time.

    Parameters
    ----------
    spiketrain : SpikeTrainArray
        Spike trains whose support intervals represent equally long trials.
    ds : float, optional
        Spacing of rate evaluation points in seconds. Default is 1 ms.
    sigma : float, optional
        Standard deviation of the Gaussian kernel in seconds. Default is 50 ms.
    error : {'none', 'poisson', 'bootstrap'}, optional
        Error estimator. ``'poisson'`` gives the pointwise standard error under
        a Poisson process assumption. ``'bootstrap'`` resamples support
        intervals with replacement. Default is ``'none'``.
    n_boot : int, optional
        Number of trial-resampling replicates for ``error='bootstrap'``.
        Default is 1000.
    random_state : int or numpy.random.Generator, optional
        Random state used for bootstrap resampling.

    Returns
    -------
    PSTHResult
        Trial-averaged rate and, if requested, a pointwise standard-error
        AnalogSignalArray.

    Notes
    -----
    The estimate is evaluated directly from spike times; it does not first bin
    spikes. Kernels have natural boundaries, so a spike near a trial edge has
    some mass outside the displayed support. This avoids borrowing data from
    outside a trial.

    Examples
    --------
    >>> import nelpy as nel
    >>> spikes = nel.SpikeTrainArray(
    ...     [[0.1, 0.3, 2.2, 2.4]], support=nel.EpochArray([[0, 1], [2, 3]])
    ... )
    >>> from nelpy.analysis import psth
    >>> result = psth(spikes, sigma=0.02, error="poisson")
    >>> result.rate.data.shape
    (1, 1000)

    """
    if not isinstance(spiketrain, core.SpikeTrainArray):
        raise TypeError("spiketrain must be a SpikeTrainArray")

    ds = _validate_positive_scalar(ds, "ds")
    sigma = _validate_positive_scalar(sigma, "sigma")
    if error not in {"none", "poisson", "bootstrap"}:
        raise ValueError("error must be one of 'none', 'poisson', or 'bootstrap'")

    trial_support = spiketrain.support
    if trial_support.isempty:
        raise ValueError("spiketrain must have at least one support interval")
    trial_starts = trial_support.starts
    trial_stops = trial_support.stops
    trial_durations = trial_stops - trial_starts
    if not np.allclose(trial_durations, trial_durations[0]):
        raise ValueError("all support intervals must have equal duration for psth")

    n_bins = int(np.floor(trial_durations[0] / ds))
    if n_bins < 1:
        raise ValueError("ds must be shorter than the trial duration")
    duration = n_bins * ds
    times = (np.arange(n_bins) + 0.5) * ds
    spikes_by_trial = _relative_spikes_by_trial(spiketrain, trial_starts, trial_stops)
    trial_rates = _gaussian_trial_rates(spikes_by_trial, times, sigma)
    rate_data = trial_rates.mean(axis=0)
    labels = spiketrain.series_labels
    rate = _make_rate_array(rate_data, times, 1 / ds, duration, labels)

    standard_error = None
    if error == "poisson":
        poisson_error = np.sqrt(
            rate_data / (2 * spiketrain.n_intervals * sigma * np.sqrt(np.pi))
        )
        standard_error = _make_rate_array(
            poisson_error, times, 1 / ds, duration, labels
        )
    elif error == "bootstrap":
        if spiketrain.n_intervals < 2:
            raise ValueError("bootstrap error requires at least two support intervals")
        if not isinstance(n_boot, int) or n_boot < 2:
            raise ValueError("n_boot must be an integer of at least two")

        rng = np.random.default_rng(random_state)
        bootstrap_rates = np.empty((n_boot, *rate_data.shape), dtype=float)
        for boot_idx in range(n_boot):
            trial_idx = rng.integers(
                spiketrain.n_intervals, size=spiketrain.n_intervals
            )
            bootstrap_rates[boot_idx] = trial_rates[trial_idx].mean(axis=0)
        standard_error = _make_rate_array(
            bootstrap_rates.std(axis=0, ddof=1), times, 1 / ds, duration, labels
        )

    return PSTHResult(
        rate=rate,
        standard_error=standard_error,
        n_trials=spiketrain.n_intervals,
        sigma=sigma,
    )
