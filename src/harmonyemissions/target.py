"""Target representations for the three emission regimes.

A :class:`Target` covers three very different kinds of plasma:
overdense solid surfaces (for ROM / CSE surface HHG), under-ionized gas jets
(for Lewenstein gas HHG), and underdense LWFA plasma (for betatron).
Each flavor is produced through a factory classmethod so that callers see
a single, unified type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["overdense", "gas", "underdense", "electron_beam"]


@dataclass(frozen=True)
class Target:
    """Plasma / gas target parameters.

    The same dataclass covers all four regimes to keep the public API simple;
    which fields matter depends on :attr:`kind`.

    Overdense (surface HHG, bremsstrahlung, K-α fluorescence)
        ``n_over_nc``: peak electron density / critical density.
        ``gradient_L_over_lambda``: density-gradient scale length in λ.
        ``hot_electron_temp_keV``: optional override for the Wilks-derived T_hot
        used by bremsstrahlung / K-α models.

    Gas (Lewenstein HHG)
        ``gas_species``: "He", "Ne", "Ar", "Kr", "Xe", "H".
        ``pressure_mbar``: backing pressure (informational).
        ``ionization_potential_eV``: overrides the species default if set.

    Underdense (LWFA betatron; also drives LWFA-ICS)
        ``n_over_nc``: plasma / critical density (≪ 1).
        ``bubble_radius_um``: LWFA bubble radius estimate.
        ``electron_energy_mev``: peak electron energy of the self-injected bunch.
        ``betatron_amplitude_um``: transverse oscillation amplitude.

    Electron_beam (externally-injected ICS / Thomson source)
        ``beam_energy_mev``: mean bunch energy.
        ``beam_charge_pc``: bunch charge in pC.
        ``beam_divergence_mrad``: RMS angular spread.
        ``beam_bunch_length_fs``: RMS bunch duration.
    """

    kind: Kind
    # overdense
    n_over_nc: float = 100.0
    gradient_L_over_lambda: float = 0.1
    material: str = "SiO2"              # target material (Timmis 2026 default)
    reflectivity: float = 0.6           # mirror reflectivity (from PIC) for denting
    # DPM contrast / prepulse — drive the contrast-model scale length.
    t_HDR_fs: float = 351.0             # high-dynamic-range rise time (paper's optimum)
    prepulse_intensity_rel: float = 0.0
    prepulse_delay_fs: float = 0.0
    # Hot-electron temperature override (defaults to Wilks scaling when None).
    hot_electron_temp_keV: float | None = None
    # gas
    gas_species: str = "Ar"
    pressure_mbar: float = 20.0
    ionization_potential_eV: float | None = None
    # underdense / LWFA
    bubble_radius_um: float = 10.0
    electron_energy_mev: float = 200.0
    betatron_amplitude_um: float = 1.0
    # electron_beam (external-beam ICS)
    beam_energy_mev: float = 100.0
    beam_charge_pc: float = 50.0
    beam_divergence_mrad: float = 1.0
    beam_bunch_length_fs: float = 50.0

    @classmethod
    def overdense(
        cls,
        n_over_nc: float,
        gradient_L_over_lambda: float = 0.1,
        material: str = "SiO2",
    ) -> Target:
        return cls(kind="overdense",
                   n_over_nc=n_over_nc,
                   gradient_L_over_lambda=gradient_L_over_lambda,
                   material=material)

    @classmethod
    def sio2(
        cls,
        t_HDR_fs: float = 351.0,
        prepulse_intensity_rel: float = 0.0,
        prepulse_delay_fs: float = 0.0,
        reflectivity: float = 0.6,
    ) -> Target:
        """Fused silica overdense target in the Timmis 2026 DPM configuration."""
        return cls(
            kind="overdense",
            n_over_nc=200.0,        # SiO₂ ionized → ~200 n_c at 800 nm
            gradient_L_over_lambda=0.14,
            material="SiO2",
            reflectivity=reflectivity,
            t_HDR_fs=t_HDR_fs,
            prepulse_intensity_rel=prepulse_intensity_rel,
            prepulse_delay_fs=prepulse_delay_fs,
        )

    @classmethod
    def gas(
        cls,
        species: str,
        pressure_mbar: float = 20.0,
        ionization_potential_eV: float | None = None,
    ) -> Target:
        return cls(kind="gas",
                   gas_species=species,
                   pressure_mbar=pressure_mbar,
                   ionization_potential_eV=ionization_potential_eV)

    @classmethod
    def underdense(
        cls,
        n_over_nc: float,
        bubble_radius_um: float = 10.0,
        electron_energy_mev: float = 200.0,
        betatron_amplitude_um: float = 1.0,
    ) -> Target:
        return cls(kind="underdense",
                   n_over_nc=n_over_nc,
                   bubble_radius_um=bubble_radius_um,
                   electron_energy_mev=electron_energy_mev,
                   betatron_amplitude_um=betatron_amplitude_um)

    @classmethod
    def electron_beam(
        cls,
        beam_energy_mev: float,
        beam_charge_pc: float = 50.0,
        beam_divergence_mrad: float = 1.0,
        beam_bunch_length_fs: float = 50.0,
    ) -> Target:
        """External-beam ICS / Thomson source (LINAC-style bunch on laser)."""
        return cls(
            kind="electron_beam",
            beam_energy_mev=beam_energy_mev,
            beam_charge_pc=beam_charge_pc,
            beam_divergence_mrad=beam_divergence_mrad,
            beam_bunch_length_fs=beam_bunch_length_fs,
        )


# Ionization potentials in eV (ground state) for noble gases commonly used in HHG.
IONIZATION_POTENTIALS_EV: dict[str, float] = {
    "H": 13.6,
    "He": 24.59,
    "Ne": 21.56,
    "Ar": 15.76,
    "Kr": 14.00,
    "Xe": 12.13,
}


def ionization_potential(target: Target) -> float:
    """Return the ionization potential [eV] for a gas target."""
    if target.ionization_potential_eV is not None:
        return target.ionization_potential_eV
    try:
        return IONIZATION_POTENTIALS_EV[target.gas_species]
    except KeyError as exc:
        raise ValueError(
            f"No default ionization potential for species {target.gas_species!r}. "
            "Set Target(ionization_potential_eV=...) explicitly."
        ) from exc
