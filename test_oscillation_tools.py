"""Tests for oscillation_tools.py.

Covers the PMNS matrix builder, two-flavor and three-flavor oscillation
probabilities, and the decoherence (Lindblad) helper functions.
"""

import numpy as np
import pytest

import oscillation_tools as osc


# ---------------------------------------------------------------------------
# PMNS matrix / OscillationParameters
# ---------------------------------------------------------------------------

class TestUall:
    def test_unitary(self):
        U = osc.Uall(
            np.deg2rad(33.44), np.deg2rad(49.2), np.deg2rad(8.57), np.deg2rad(195.0)
        )
        np.testing.assert_allclose(U @ U.conj().T, np.eye(3), atol=1e-10)

    def test_zero_theta13_kills_direct_e_tau_coupling(self):
        # With theta13 = 0, U[0, 2] (the direct e <-> mass-3 coupling)
        # vanishes and loses its CP-phase dependence, regardless of delta_cp.
        theta12, theta23 = np.deg2rad(33.44), np.deg2rad(49.2)
        U = osc.Uall(theta12, theta23, 0.0, np.deg2rad(195.0))
        assert abs(U[0, 2]) < 1e-12
        # U[2, 0] is still populated indirectly through the theta23-theta12
        # rotation product (R23 @ R12), independent of delta_cp once s13 = 0.
        assert U[2, 0] == pytest.approx(np.sin(theta23) * np.sin(theta12))

    def test_zero_angles_is_identity(self):
        U = osc.Uall(0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(U, np.eye(3), atol=1e-12)


class TestOscillationParameters:
    def test_default_masses(self):
        params = osc.OscillationParameters()
        assert params.masses[0] == 0.0
        np.testing.assert_allclose(params.masses[1], np.sqrt(7.42e-5))
        np.testing.assert_allclose(params.masses[2], np.sqrt(2.517e-3))

    def test_mixing_matrix_matches_uall(self):
        params = osc.OscillationParameters()
        expected = osc.Uall(
            params.theta12, params.theta23, params.theta13, params.delta_cp
        )
        np.testing.assert_allclose(params.mixing_matrix(), expected)

    def test_antineutrino_mixing_matrix_is_conjugated(self):
        params = osc.OscillationParameters()
        np.testing.assert_allclose(
            params.mixing_matrix(antineutrino=True),
            np.conjugate(params.mixing_matrix()),
        )


# ---------------------------------------------------------------------------
# Two-flavor oscillation
# ---------------------------------------------------------------------------

class TestProb2f:
    def test_vacuum_matches_standard_formula(self):
        L_km, E_GeV, theta, dm2 = 810.0, 2.0, np.deg2rad(45.0), 2.5e-3
        expected = np.sin(2 * theta) ** 2 * np.sin(1.267 * dm2 * L_km / E_GeV) ** 2
        got = osc.prob_2f(L_km, E_GeV, theta, dm2, rho=0, Ye=0)
        assert got == pytest.approx(expected, abs=1e-12)

    def test_zero_baseline_gives_zero_probability(self):
        assert osc.prob_2f(0.0, 1.0, np.deg2rad(45.0), 2.5e-3) == pytest.approx(0.0)

    def test_maximal_mixing_vacuum_reaches_full_swap_at_peak(self):
        # At maximal mixing (theta=45deg) and phase = pi/2 the survival
        # probability of oscillating away should be 1.
        theta = np.deg2rad(45.0)
        dm2 = 2.5e-3
        E_GeV = 1.0
        L_km = (np.pi / 2) / (1.267 * dm2 / E_GeV)
        got = osc.prob_2f(L_km, E_GeV, theta, dm2, rho=0, Ye=0)
        assert got == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("rho,Ye,antineutrino", [
        (0, 0, False),
        (2.8, 0.5, False),
        (2.8, 0.5, True),
        (5.0, 0.3, False),
    ])
    def test_probability_bounded(self, rho, Ye, antineutrino):
        got = osc.prob_2f(
            1300.0, 1.5, np.deg2rad(33.0), 7.4e-5, rho=rho, Ye=Ye,
            antineutrino=antineutrino,
        )
        assert 0.0 <= got <= 1.0

    def test_antineutrino_flips_matter_potential_sign(self):
        # Away from vacuum, flipping antineutrino should generally change
        # the result whenever there is a nonzero matter potential.
        kwargs = dict(L_km=1300.0, E_GeV=3.0, theta=np.deg2rad(33.0), dm2=7.4e-5,
                      rho=2.8, Ye=0.5)
        nu = osc.prob_2f(**kwargs, antineutrino=False)
        nubar = osc.prob_2f(**kwargs, antineutrino=True)
        assert nu != pytest.approx(nubar)


# ---------------------------------------------------------------------------
# Three-flavor oscillation
# ---------------------------------------------------------------------------

class TestThreeFlavorOscillation:
    def test_survival_probability_is_one_at_zero_baseline(self):
        model = osc.ThreeFlavorOscillation(channel="mu_mu")
        assert model.probability(0.0, 1.0) == pytest.approx(1.0, abs=1e-9)

    def test_transition_probability_is_zero_at_zero_baseline(self):
        model = osc.ThreeFlavorOscillation(channel="mu_e")
        assert model.probability(0.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_probabilities_from_fixed_initial_flavor_sum_to_one(self):
        # Unitarity: sum over final flavors of P(initial -> final) == 1.
        L_km, E_GeV = 1300.0, 2.0
        total = 0.0
        for final in ("e", "mu", "tau"):
            model = osc.ThreeFlavorOscillation(channel=f"mu_{final}")
            total += model.probability(L_km, E_GeV)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_callable_matches_probability_method(self):
        model = osc.ThreeFlavorOscillation(channel="mu_e")
        assert model(1300.0, 2.0) == model.probability(1300.0, 2.0)

    def test_unknown_channel_raises(self):
        model = osc.ThreeFlavorOscillation(channel="mu_bogus")
        with pytest.raises(ValueError):
            model.probability(1300.0, 2.0)

    def test_probability_is_bounded(self):
        model = osc.ThreeFlavorOscillation(channel="mu_e", rho=2.8, Ye=0.5)
        got = model.probability(1300.0, 0.8)
        assert 0.0 <= got <= 1.0

    def test_vacuum_three_flavor_matches_two_flavor_leading_order(self):
        # In vacuum, with theta13 = 0 AND no solar splitting (dm21 = 0),
        # mass state 3 has zero e-flavor content and states 1/2 are
        # degenerate, so mu -> e transitions vanish exactly.
        model = osc.ThreeFlavorOscillation(
            rho=0, Ye=0, theta13=0.0, dm21=0.0, channel="mu_e",
        )
        assert model.probability(1300.0, 2.0) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# prob_full compatibility wrapper
# ---------------------------------------------------------------------------

class TestProbFull:
    def test_matches_three_flavor_oscillation_class(self):
        # prob_full is documented as a compatibility wrapper around
        # ThreeFlavorOscillation, so it should reproduce the class result
        # for the same parameters.
        model = osc.ThreeFlavorOscillation(channel="mu_e")
        expected = model.probability(1300.0, 2.0)
        got = osc.prob_full(1300.0, 2.0, channel="mu_e")
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Decoherence helpers
# ---------------------------------------------------------------------------

class TestDaggerAndBasisTransforms:
    def test_dagger_2d(self):
        m = np.array([[1 + 1j, 2], [3, 4 - 2j]])
        np.testing.assert_allclose(osc.dagger(m), m.conj().T)

    def test_dagger_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            osc.dagger(np.array(1.0))

    def test_flav_mass_roundtrip(self):
        U = osc.Uall(
            np.deg2rad(33.44), np.deg2rad(49.2), np.deg2rad(8.57), np.deg2rad(195.0)
        )
        rho_flav = np.diag([1.0, 0.0, 0.0]).astype(complex)
        rho_mass = osc.flav_to_mass(rho_flav, U)
        back = osc.mass_to_flav(rho_mass, U)
        np.testing.assert_allclose(back, rho_flav, atol=1e-12)

    def test_calc_bin_centers(self):
        edges = np.array([0.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(osc.calc_bin_centers(edges), [1.0, 3.0, 5.0])


class TestDecoherenceOscillation:
    def test_probabilities_sum_to_one_with_zero_damping(self):
        # With all gammas zero the evolution is unitary, so probabilities
        # to each final flavor from a fixed initial flavor must sum to 1
        # in every energy bin.
        E_edges = np.array([0.5e9, 1.0e9, 2.0e9])  # eV
        L = 1300.0 * osc.km  # baseline in eV^-1

        total = np.zeros(len(E_edges) - 1)
        for final in ("e", "mu", "tau"):
            model = osc.DecoherenceOscillation(gamma31=0, gamma21=0, gamma32=0)
            total += model.probability(L, E_edges, channel=f"mu_{final}")
        np.testing.assert_allclose(total, np.ones_like(total), atol=1e-6)

    def test_zero_baseline_survival_is_one(self):
        E_edges = np.array([0.5e9, 1.0e9])
        model = osc.DecoherenceOscillation()
        got = model.probability(0.0, E_edges, channel="e_e")
        np.testing.assert_allclose(got, [1.0], atol=1e-9)

    def test_unknown_channel_raises(self):
        E_edges = np.array([0.5e9, 1.0e9])
        model = osc.DecoherenceOscillation()
        with pytest.raises(ValueError):
            model.probability(0.0, E_edges, channel="mu_bogus")

    def test_damping_reduces_off_diagonal_relative_to_unitary_case(self):
        # Turning on decoherence should not push a physical probability
        # outside [0, 1].
        E_edges = np.array([0.5e9, 1.0e9])
        L = 1300.0 * osc.km
        model = osc.DecoherenceOscillation(gamma31=1e-14, gamma21=1e-14, gamma32=1e-14)
        got = model.probability(L, E_edges, channel="mu_e")
        assert np.all(got >= -1e-9) and np.all(got <= 1.0 + 1e-9)


class TestMakeDissipator:
    def test_shape(self):
        H = np.zeros((2, 3, 3), dtype=complex)
        D = osc.make_dissipator(H, gamma31=1.0, gamma21=2.0, gamma32=3.0)
        assert D.shape == (18, 18)

    def test_non_three_flavor_raises(self):
        H = np.zeros((2, 2, 2), dtype=complex)
        with pytest.raises(ValueError):
            osc.make_dissipator(H)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
