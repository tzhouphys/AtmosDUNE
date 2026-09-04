#!/usr/bin/env python
# coding: utf-8

"""
    Oscillation Tool Functions
    * Calculate two-flavor neutrino oscillation probabilities in both vacuum and matter(constant density)
    * Calculate basic three-flavor neutrino oscillation probabilities in both vacuum and matter(constant density)
    * Calculate dissipative neutrino oscillation probabilities in both vacuum and matter(constant density) using the Lindblad master equation
    TODO: Add varying matter density for DUNE parametric resonance oscillation
    TODO: write class for oscillation parameters and methods to calculate probabilities
"""

import numpy as np
from scipy.linalg import expm

class OscillationParameters:
    """Global default oscillation parameters.

    Masses are stored in eV, with the lightest mass set to zero. The
    corresponding mass-squared splittings are stored in eV^2. Angles and
    the CP phase are stored in radians. Decoherence helpers use energies in
    eV and lengths in eV^-1.
    """

    def __init__(
        self,
        dm21=7.42e-5,
        dm31=2.517e-3,
        theta12=np.deg2rad(33.44),
        theta23=np.deg2rad(49.2),
        theta13=np.deg2rad(8.57),
        delta_cp=np.deg2rad(195.0),
    ):
        self.dm21 = float(dm21)
        self.dm31 = float(dm31)
        self.theta12 = float(theta12)
        self.theta23 = float(theta23)
        self.theta13 = float(theta13)
        self.delta_cp = float(delta_cp)
        self.masses = np.array(
            [0.0, np.sqrt(abs(dm21)), np.sqrt(abs(dm31))],
            dtype=float,
        )

    def mixing_matrix(self, antineutrino=False):
        """Return the PMNS matrix, conjugated for antineutrinos."""
        matrix = Uall(
            self.theta12,
            self.theta23,
            self.theta13,
            self.delta_cp,
        )
        return np.conjugate(matrix) if antineutrino else matrix


OSC_PARAMS = OscillationParameters()

# Backwards-compatible global aliases used by the decoherence examples.
masses = OSC_PARAMS.masses
theta12 = OSC_PARAMS.theta12
theta23 = OSC_PARAMS.theta23
theta13 = OSC_PARAMS.theta13
deltaCP = OSC_PARAMS.delta_cp

def Uall(theta12, theta23, theta13, deltaCP):

    d_ = np.exp(-1j * deltaCP)
    d  = np.exp( 1j * deltaCP)

    s12, c12 = np.sin(theta12), np.cos(theta12)
    s23, c23 = np.sin(theta23), np.cos(theta23)
    s13, c13 = np.sin(theta13), np.cos(theta13)

    U = np.linalg.multi_dot(([[1, 0,    0   ],
                               [0, c23,  s23 ],
                               [0, -s23, c23 ]],
                                                [[c13,      0, s13 * d_],
                                                 [0,        1, 0       ],
                                                 [-s13 * d, 0, c13    ]],
                              [[c12,  s12, 0],
                               [-s12, c12, 0],
                               [0,    0,   1]]))
    return U

U = OSC_PARAMS.mixing_matrix()

###### Two-flavor SM neutrino oscillation probability function ######

def prob_2f(L_km, E_GeV, theta, dm2, rho=0, Ye=0, antineutrino=False):
    """
    Two-flavor neutrino oscillation probability in constant-density matter.

    Parameters
    ----------
    L_km : float
        Baseline in km.
    E_GeV : float
        Neutrino energy in GeV.
    theta : float
        Mixing angle in radians.
    dm2 : float
        Mass-squared splitting in eV^2.
    rho : float
        Matter density in g/cm^3.
    Ye : float
        Electron fraction of the matter.
    antineutrino : bool
        If True, reverse the sign of the matter potential.

    Returns
    -------
    float
        Probability
    """
    # Matter potential in mass-squared units, with E in GeV.
    a = 1.526494e-4 * rho * Ye * E_GeV
    if antineutrino:
        a = -a

    sin2theta = np.sin(2 * theta)
    cos2theta = np.cos(2 * theta)
    dm2_m = dm2 * np.sqrt(
        (cos2theta - a / dm2) ** 2 + sin2theta ** 2
    )
    sin2_2theta_m = dm2 * sin2theta / dm2_m

    phase = 1.267 * dm2_m * L_km / E_GeV
    prob = sin2_2theta_m ** 2 * np.sin(phase) ** 2
    return float(np.clip(prob, 0.0, 1.0))

###### Three-flavor SM neutrino oscillation probability ######

class ThreeFlavorOscillation:
    """Three-flavor oscillations in constant-density matter.

    The constructor stores the physical setup. ``L_km`` and ``E_GeV`` are
    supplied to :meth:`probability`, so one instance can be reused for many
    baselines and energies.
    """

    def __init__(
        self,
        rho=2.8,
        Ye=0.5,
        theta12=OSC_PARAMS.theta12,
        theta13=OSC_PARAMS.theta13,
        theta23=OSC_PARAMS.theta23,
        delta_cp=OSC_PARAMS.delta_cp,
        dm21=OSC_PARAMS.dm21,
        dm31=OSC_PARAMS.dm31,
        channel = "mu_e",
        antineutrino=False,
    ):
        self.rho = float(rho)
        self.Ye = float(Ye)
        self.theta12 = float(theta12)
        self.theta13 = float(theta13)
        self.theta23 = float(theta23)
        self.delta_cp = float(delta_cp)
        self.dm21 = float(dm21)
        self.dm31 = float(dm31)
        self.channel = str(channel)
        self.antineutrino = bool(antineutrino)

    def probability(self, L_km, E_GeV, channel=None):
        """Return the requested flavor-transition probability."""
        channel = self.channel if channel is None else str(channel)
        U = Uall(
            self.theta12,
            self.theta23,
            self.theta13,
            self.delta_cp,
        )
        if self.antineutrino:
            U = np.conjugate(U)

        M2_vac = np.diag([0.0, self.dm21, self.dm31])

        # Matter potential in eV^2; E_GeV is converted by the coefficient.
        a = 1.526494e-4 * self.rho * self.Ye * E_GeV
        if self.antineutrino:
            a = -a
        M2_flavor = U @ M2_vac @ U.conj().T + np.diag([a, 0.0, 0.0])

        # H is in 1/km because L_km is supplied in km.
        H = 2 * 1.267 * M2_flavor / E_GeV
        S = expm(-1j * H * L_km)

        flavor_index = {"e": 0, "mu": 1, "tau": 2}
        try:
            initial, final = channel.split("_")
            amplitude = S[flavor_index[final], flavor_index[initial]]
        except (KeyError, ValueError):
            raise ValueError(f"Unknown channel: {channel}")

        return float(np.clip(np.abs(amplitude) ** 2, 0.0, 1.0))

    __call__ = probability


def prob_full(
    L_km,
    E_GeV,
    rho=2.8,
    Ye=0.5,
    theta12=OSC_PARAMS.theta12,
    theta13=OSC_PARAMS.theta13,
    theta23=OSC_PARAMS.theta23,
    delta_cp=OSC_PARAMS.delta_cp,
    dm21=OSC_PARAMS.dm21,
    dm31=OSC_PARAMS.dm31,
    antineutrino=False,
    channel="mu_e",
):
    """Compatibility wrapper for :class:`ThreeFlavorOscillation`."""
    model = ThreeFlavorOscillation(
        rho=rho,
        Ye=Ye,
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        delta_cp=delta_cp,
        dm21=dm21,
        dm31=dm31,
        antineutrino=antineutrino,
    )
    return model.probability(L_km, E_GeV, channel=channel)

######## Decoherence tools adapted from nudice ########


# this code runs internally with eV so these might be useful
meter = 5.06773093741e6        # [eV^-1/m]
km    = 1.0e3*meter            # [eV^-1/km]
MeV   = 1.0e6                  # [eV/MeV]

def sq(x): return x * x
def cube(x): return x * x * x
def ht(x): return np.heaviside(x, 0)
def dagger(x):
    if np.ndim(x) == 2:
        return np.conj(x).T
    if np.ndim(x) == 3:
        return np.transpose(np.conj(x), axes=(0, 2, 1))
    if np.ndim(x) == 4:
        return np.transpose(np.conj(x), axes=(0, 1, 3, 2))
    if np.ndim(x) == 5:
        return np.transpose(np.conj(x), axes=(0, 1, 2, 4, 3))
    raise ValueError("dagger only supports arrays with 2 to 5 dimensions")

# rho_m =  U_dagger * rho_f * U (for neutrinos i_nu == 0)
def flav_to_mass(rho, U, i_nu=0):

    if i_nu == 0:
        return np.linalg.multi_dot((dagger(U), rho, U))
    return np.linalg.multi_dot((U, rho, dagger(U)))

# rho_f = U * rho_m * U_dagger (for neutrinos i_nu == 0)
def mass_to_flav(rho, U, i_nu=0):

    if i_nu == 0:
        return np.linalg.multi_dot((U, rho, dagger(U)))
    return np.linalg.multi_dot((dagger(U), rho, U))

def calc_bin_centers(bin_edges):
    return 0.5 * (bin_edges[1:] + bin_edges[:-1])

def _hamiltonian(e_edges, masses, hamiltonian=None):
    """Helper function to build the vacuum Hamiltonian or validate a user-supplied Hamiltonian."""

    e_centr = calc_bin_centers(e_edges)
    n_bins  = len(e_centr)
    Ndim    = len(masses)

    if hamiltonian is None:
        H = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
        for k, m in enumerate(masses):
            H[:, k, k] = (sq(m) - sq(masses[0])) / (2 * e_centr)
        return H

    H = np.asarray(hamiltonian, dtype=np.complex128)
    if H.shape == (Ndim, Ndim):
        return np.broadcast_to(H, (n_bins, Ndim, Ndim)).copy()
    if H.shape == (n_bins, Ndim, Ndim):
        return H
    raise ValueError(
        "hamiltonian must have shape (Ndim, Ndim) or "
        "(n_bins, Ndim, Ndim)."
    )

def unravelled_master_eqn_general(L, H, D):

    n_bins = len(H)
    Ndim   = H.shape[-1]

    L_super = D
    if L_super.shape != (sq(Ndim) * n_bins, sq(Ndim) * n_bins):
        raise ValueError("Dissipator D must have shape (sq(Ndim) * n_bins, sq(Ndim) * n_bins)")
    I = np.eye(Ndim)

    for n in range(n_bins):
        block  = -1j * (np.kron(np.eye(Ndim), H[n]) - np.kron(H[n].T, I))
        L_super[
            sq(Ndim) * n : sq(Ndim) * (n + 1),
            sq(Ndim) * n : sq(Ndim) * (n + 1),
        ] += block

    return expm(L_super * L)

def dynam_general(initial_value, L, e_edges, masses, dissipator=None, hamiltonian=None):
    """Solve the Lindblad equation via the dynamical map.

    The Liouvillian is exponentiated once and applied to the vectorized initial
    state.  This implementation is for per-energy-bin Lindblad evolution; it
    does not redistribute probability between energy bins.

    Here the lindblad (jump) operator (eqn 27 in 2604.09776) is in eV^1/2
    """

    H = _hamiltonian(e_edges, masses, hamiltonian)

    vec_rho = np.asarray(initial_value, dtype=np.complex128).reshape(-1)
    dy_map = unravelled_master_eqn_general(L, H, dissipator)
    solution = dy_map @ vec_rho

    return solution.reshape(H.shape)

def matter_hamiltonian(
    e_edges,
    U,
    masses,
    rho=2.8,
    Ye=0.5,
    antineutrino=False,
):
    E_eV = calc_bin_centers(e_edges)          # already in eV
    n_bins = len(E_eV)
    Ndim = len(masses)

    H = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
    for i, E in enumerate(E_eV):
        a = 1.526494e-13 * rho * Ye * E     # eV^2 matter potential
        if antineutrino:
            a = -a
        V = np.diag([a, 0.0, 0.0])

        M2_vac = np.diag([0.0, masses[1]**2, masses[2]**2])
        H_flav = U @ M2_vac @ U.conj().T + V

        H[i] = dagger(U) @ H_flav @ U / (2.0 * E)

    return H

def make_dissipator(H, gamma31=0, gamma21=0, gamma32=0):
    """
    D corresponding to Gamma_31 damping in the eigenbasis of H.

    Uses column-major vectorization:
        vec(rho) = [rho11,rho21,rho31,rho12,...]
    """

    n_bins = len(H)
    Ndim = H.shape[-1]

    if Ndim != 3:
        raise ValueError("This implementation assumes 3 flavors.")

    D = np.zeros(
        (9 * n_bins, 9 * n_bins),
        dtype=np.complex128
    )

    for n in range(n_bins):

        # H = U diag(E_i) U^\dagger
        evals, U = np.linalg.eigh(H[n])

        # Dissipator in propagation eigenbasis
        D_eig = np.zeros((9, 9), dtype=np.complex128)

        # column-major indices:
        # rho31 -> 2
        # rho13 -> 6
        D_eig[2, 2] = -gamma31
        D_eig[6, 6] = -gamma31
        D_eig[1, 1] = -gamma21   # rho_21
        D_eig[3, 3] = -gamma21   # rho_12
        D_eig[5, 5] = -gamma32   # rho_32
        D_eig[7, 7] = -gamma32   # rho_23

        # D_eig is defined in the propagation eigenbasis. Transform it
        # back to the mass-basis coordinates used by dynam_general.
        # from_eig = np.kron(U.conj(), U)
        # to_eig = np.kron(U.T, U.conj().T)
        # D_block = from_eig @ D_eig @ to_eig

        D[
            9*n : 9*(n+1),
            9*n : 9*(n+1)
        ] = D_eig

    return D


class DecoherenceOscillation:
    """Three-flavor Lindblad oscillations with fixed physical parameters.

    Energies in ``E_edges`` are in eV and the baseline ``L`` is in eV^-1,
    matching the natural-unit convention used by the decoherence helpers.
    The matter Hamiltonian is represented in the vacuum mass basis, while
    damping rates are in eV.
    """

    def __init__(
        self,
        params=None,
        rho=3.0,
        Ye=0.5,
        gamma31=0.0,
        gamma21=0.0,
        gamma32=0.0,
        antineutrino=False,
    ):
        self.params = OSC_PARAMS if params is None else params
        self.rho = float(rho)
        self.Ye = float(Ye)
        self.gamma31 = float(gamma31)
        self.gamma21 = float(gamma21)
        self.gamma32 = float(gamma32)
        self.antineutrino = bool(antineutrino)

    @staticmethod
    def _initial_flavor_state(channel):
        flavor_index = {"e": 0, "mu": 1, "tau": 2}
        try:
            initial, final = channel.split("_")
            if final not in flavor_index:
                raise KeyError(final)
            state = np.zeros(3, dtype=complex)
            state[flavor_index[initial]] = 1.0
            return state
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"Unknown channel: {channel}")

    def probability(self, L, E_edges, channel="mu_e"):
        """Return the channel probability for every energy bin."""
        e_centers = calc_bin_centers(E_edges)
        n_bins = len(e_centers)
        U = self.params.mixing_matrix(antineutrino=self.antineutrino)
        masses = self.params.masses

        nu = self._initial_flavor_state(channel)
        rho_flav_0 = np.outer(nu, nu.conj())
        rho_mass_0 = flav_to_mass(rho_flav_0, U)
        rho0 = np.broadcast_to(rho_mass_0, (n_bins, 3, 3)).copy()

        H_matter = matter_hamiltonian(
            E_edges,
            U,
            masses,
            rho=self.rho,
            Ye=self.Ye,
            antineutrino=self.antineutrino,
        )
        D = make_dissipator(
            H_matter,
            gamma31=self.gamma31,
            gamma21=self.gamma21,
            gamma32=self.gamma32,
        )
        rho_mass_L = dynam_general(
            initial_value=rho0,
            L=L,
            e_edges=E_edges,
            masses=masses,
            hamiltonian=H_matter,
            dissipator=D,
        )
        rho_flav_L = np.array(
            [mass_to_flav(rho, U) for rho in rho_mass_L]
        )
        final = channel.split("_")[1]
        flavor_index = {"e": 0, "mu": 1, "tau": 2}
        try:
            return np.real(
                np.diagonal(rho_flav_L, axis1=1, axis2=2)
            )[:, flavor_index[final]]
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"Unknown channel: {channel}")

    __call__ = probability

def prob_diss(
    L,
    E_edges,
    channel="mu_e",
    gamma31=0,
    gamma21=0,
    gamma32=0,
    rho=3,
    Ye=0.5,
    antineutrino=False,
):
    """Compatibility wrapper for :class:`DecoherenceOscillation`."""
    model = DecoherenceOscillation(
        rho=rho,
        Ye=Ye,
        gamma31=gamma31,
        gamma21=gamma21,
        gamma32=gamma32,
        antineutrino=antineutrino,
    )
    return model.probability(L, E_edges, channel=channel)

    
    #### piecewise constant matter density for DUNE parametric resonance oscillation ####
