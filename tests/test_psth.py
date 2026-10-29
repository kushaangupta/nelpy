import numpy as np
import pytest

import nelpy as nel
from nelpy.analysis import PSTHResult, psth


def _two_trial_spiketrain(spikes):
    return nel.SpikeTrainArray(
        spikes,
        support=nel.EpochArray([[0, 1], [2, 3]]),
        series_labels=["unit-a" for _ in spikes],
    )


class TestPSTH:
    def test_aligns_support_intervals_and_estimates_gaussian_rate(self):
        spiketrain = _two_trial_spiketrain([[0.505, 2.505]])

        result = psth(spiketrain, ds=0.01, sigma=0.02)

        assert isinstance(result, PSTHResult)
        assert result.n_trials == 2
        assert result.standard_error is None
        assert result.rate.data.shape == (1, 100)
        np.testing.assert_allclose(result.rate.support.data, [[0, 1]])
        assert result.rate.labels.tolist() == ["unit-a"]

        peak_idx = np.flatnonzero(np.isclose(result.rate.abscissa_vals, 0.505))[0]
        expected_peak = 1 / (0.02 * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(result.rate.data[0, peak_idx], expected_peak)
        np.testing.assert_allclose(result.rate.data.sum() * 0.01, 1, atol=1e-10)

    def test_averages_over_trials_and_reports_poisson_standard_error(self):
        spiketrain = _two_trial_spiketrain([[0.505]])

        result = psth(spiketrain, ds=0.01, sigma=0.02, error="poisson")

        peak_idx = np.flatnonzero(np.isclose(result.rate.abscissa_vals, 0.505))[0]
        expected_peak = 1 / (2 * 0.02 * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(result.rate.data[0, peak_idx], expected_peak)
        expected_error = np.sqrt(expected_peak / (2 * 2 * 0.02 * np.sqrt(np.pi)))
        np.testing.assert_allclose(
            result.standard_error.data[0, peak_idx], expected_error
        )

    def test_bootstrap_is_reproducible(self):
        spiketrain = _two_trial_spiketrain([[0.3, 2.6]])

        first = psth(
            spiketrain,
            ds=0.01,
            sigma=0.03,
            error="bootstrap",
            n_boot=30,
            random_state=14,
        )
        second = psth(
            spiketrain,
            ds=0.01,
            sigma=0.03,
            error="bootstrap",
            n_boot=30,
            random_state=14,
        )

        np.testing.assert_allclose(
            first.standard_error.data, second.standard_error.data
        )
        assert np.all(first.standard_error.data >= 0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ds": 0}, "positive finite"),
            ({"sigma": 0}, "positive finite"),
            ({"error": "jackknife"}, "must be one of"),
            ({"error": "bootstrap", "n_boot": 1}, "at least two"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, match):
        spiketrain = _two_trial_spiketrain([[0.2, 2.2]])

        with pytest.raises(ValueError, match=match):
            psth(spiketrain, **kwargs)

    def test_rejects_invalid_trial_structure_and_input_type(self):
        unequal_trials = nel.SpikeTrainArray(
            [[0.2, 2.2]], support=nel.EpochArray([[0, 1], [2, 3.5]])
        )
        with pytest.raises(ValueError, match="equal duration"):
            psth(unequal_trials)

        with pytest.raises(TypeError, match="SpikeTrainArray"):
            psth([0.1, 0.2])

        one_trial = nel.SpikeTrainArray([[0.2]], support=nel.EpochArray([[0, 1]]))
        with pytest.raises(ValueError, match="at least two support intervals"):
            psth(one_trial, error="bootstrap")
