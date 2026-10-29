import numpy as np
import pytest
from scipy.special import erf
from scipy.stats import norm

import nelpy as nel


def _inhomogeneous_poisson(rng, duration, peak=35.0):
    def rate(t):
        return 2 + 25 * np.exp(-0.5 * ((t - duration / 4) / 0.6) ** 2)

    times = np.sort(rng.uniform(0, duration, rng.poisson(peak * duration)))
    return times[rng.uniform(0, peak, times.size) < rate(times)]


def _exact_cost(spikes_by_interval, width):
    """Shimazaki-Shinomoto cost from all spike pairs within each interval."""
    cost = 0.0
    for spikes in spikes_by_interval:
        lags = spikes[:, None] - spikes[None, :]
        both = norm.pdf(lags, scale=np.sqrt(2) * width).sum()
        cross = norm.pdf(lags, scale=width).sum() - spikes.size * norm.pdf(
            0, scale=width
        )
        cost += both - 2 * cross
    return cost


class TestKernelRate:
    def test_single_spike_gives_normalized_truncated_gaussian(self):
        st = nel.SpikeTrainArray([0.5], support=nel.EpochArray([0, 1]), fs=1000)

        rate = nel.utils.kernel_rate(st, ds=0.001, sigma=0.05)

        lags = rate.abscissa_vals - 0.5
        expected = np.where(
            np.abs(lags) <= 0.2,
            norm.pdf(lags, scale=0.05) / erf(4 / np.sqrt(2)),
            0,
        )
        np.testing.assert_allclose(rate.data[0], expected, atol=1e-12)
        assert rate.data.sum() * 0.001 == pytest.approx(1)

    def test_matches_direct_kernel_sum_per_unit_and_interval(self):
        rng = np.random.default_rng(0)
        support = nel.EpochArray([[0, 3], [4, 6]])
        spikes = [np.sort(rng.uniform(0, 6, n)) for n in (40, 15, 3)]
        spikes = [s[(s < 3) | (s >= 4)] for s in spikes]
        st = nel.SpikeTrainArray(spikes, support=support, fs=1000)
        sigmas = np.array([0.05, 0.2, 0.1])

        rate = nel.utils.kernel_rate(
            st, ds=0.01, sigma=sigmas, truncate=3, edge_correction=False
        )

        centers = rate.abscissa_vals
        expected = np.zeros((3, centers.size))
        for start, stop in support.data:
            grid = (centers >= start) & (centers < stop)
            for unit, (times, sigma) in enumerate(zip(spikes, sigmas)):
                inside = times[(times >= start) & (times < stop)]
                lags = centers[grid, None] - inside[None, :]
                density = norm.pdf(lags, scale=sigma) * (np.abs(lags) <= 3 * sigma)
                expected[unit, grid] = density.sum(axis=1) / erf(3 / np.sqrt(2))
        np.testing.assert_allclose(rate.data, expected, atol=1e-10)

    def test_edge_correction_keeps_constant_rate_at_support_boundaries(self):
        spikes = np.arange(0.0005, 2, 0.001)
        st = nel.SpikeTrainArray(spikes, support=nel.EpochArray([0, 2]), fs=10000)

        corrected = nel.utils.kernel_rate(st, ds=0.01, sigma=0.05)
        uncorrected = nel.utils.kernel_rate(
            st, ds=0.01, sigma=0.05, edge_correction=False
        )

        np.testing.assert_allclose(corrected.data, 1000, rtol=1e-3)
        assert uncorrected.data[0, 0] < 600
        assert uncorrected.data[0, 100] == pytest.approx(1000, rel=1e-3)

    def test_output_grid_matches_binning_and_is_non_negative(self):
        rng = np.random.default_rng(1)
        st = nel.SpikeTrainArray(
            [_inhomogeneous_poisson(rng, 10), _inhomogeneous_poisson(rng, 10)],
            support=nel.EpochArray([0, 10]),
            fs=1000,
        )

        rate = nel.utils.kernel_rate(st, ds=0.02, sigma=0.1)
        bst = st.bin(ds=0.02)

        np.testing.assert_allclose(rate.abscissa_vals, bst.bin_centers)
        np.testing.assert_array_equal(rate.support.data, bst.support.data)
        assert rate.fs == pytest.approx(50)
        np.testing.assert_array_equal(rate.labels, st.unit_labels)
        assert np.all(rate.data >= 0)

    def test_spikes_do_not_contribute_across_support_gaps(self):
        st = nel.SpikeTrainArray(
            [0.95, 0.97, 0.99], support=nel.EpochArray([[0, 1], [1.05, 2]]), fs=1000
        )

        rate = nel.utils.kernel_rate(st, ds=0.01, sigma=0.2)

        assert np.all(rate.data[0, rate.abscissa_vals > 1.05] == 0)
        assert rate.data[0, rate.abscissa_vals < 1].max() > 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ds": 0, "sigma": 0.1}, "ds must be positive"),
            ({"ds": 0.01, "sigma": -0.1}, "sigma must be positive"),
            ({"ds": 0.01, "sigma": [0.1, 0.2]}, "one value per unit"),
            ({"ds": 0.01, "sigma": 0.1, "truncate": 0}, "truncate must be positive"),
            ({"ds": 2, "sigma": 0.1}, "longer than every support interval"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, match):
        st = nel.SpikeTrainArray([0.1, 0.5], support=nel.EpochArray([0, 1]), fs=1000)

        with pytest.raises(ValueError, match=match):
            nel.utils.kernel_rate(st, **kwargs)

    def test_rejects_unsupported_input(self):
        asa = nel.AnalogSignalArray([0, 1, 0, 1], fs=10)

        with pytest.raises(NotImplementedError, match="not yet supported"):
            nel.utils.kernel_rate(asa, ds=0.1, sigma=0.1)


class TestKernelBandwidth:
    def test_matches_exact_pairwise_cost_within_each_interval(self):
        rng = np.random.default_rng(2)
        support = nel.EpochArray([[0, 20], [25, 40]])
        units = []
        for _ in range(2):
            first = _inhomogeneous_poisson(rng, 20)
            second = 25 + _inhomogeneous_poisson(rng, 15)
            units.append(np.r_[first, second])
        st = nel.SpikeTrainArray(units, support=support, fs=1000)
        candidates = np.geomspace(0.05, 5, 40)

        bandwidths = nel.utils.kernel_bandwidth(st, candidates=candidates)

        for unit, times in enumerate(units):
            by_interval = [times[times < 20], times[times >= 25]]
            exact = [_exact_cost(by_interval, width) for width in candidates]
            assert bandwidths[unit] == candidates[np.argmin(exact)]

    def test_modulated_rate_gets_narrower_bandwidth_than_constant_rate(self):
        rng = np.random.default_rng(3)
        constant = np.sort(rng.uniform(0, 40, 400))
        modulated = _inhomogeneous_poisson(rng, 40)
        st = nel.SpikeTrainArray(
            [constant, modulated], support=nel.EpochArray([0, 40]), fs=1000
        )

        constant_width, modulated_width = nel.utils.kernel_bandwidth(st)

        assert modulated_width < 1
        assert constant_width > 3 * modulated_width

    def test_returns_nan_for_units_with_fewer_than_two_spikes(self):
        st = nel.SpikeTrainArray(
            [[0.1, 0.2, 0.6, 0.65, 0.7], [0.3], []],
            support=nel.EpochArray([0, 1]),
            fs=1000,
        )

        bandwidths = nel.utils.kernel_bandwidth(st)

        assert bandwidths.shape == (3,)
        assert np.isfinite(bandwidths[0])
        assert np.isnan(bandwidths[1:]).all()

    @pytest.mark.parametrize(
        ("candidates", "match"),
        [([], "non-empty"), ([0.1, np.nan], "finite"), ([0, 0.1], "positive")],
    )
    def test_rejects_invalid_candidates(self, candidates, match):
        st = nel.SpikeTrainArray([0.1, 0.5], support=nel.EpochArray([0, 1]), fs=1000)

        with pytest.raises(ValueError, match=match):
            nel.utils.kernel_bandwidth(st, candidates=candidates)

    def test_rejects_unsupported_input(self):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            nel.utils.kernel_bandwidth(nel.AnalogSignalArray([0, 1, 0], fs=10))
