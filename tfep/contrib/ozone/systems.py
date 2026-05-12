"""Ozone (O3) OpenMM toy systems + minimal MD runner.

OpenMM is an optional dependency of tfep; this file imports it lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _require_openmm():
    try:
        import openmm
        import openmm.unit as unit
        from openmm.app import Topology, Simulation, DCDReporter, StateDataReporter, PDBFile
        from openmm.app.element import Element
        return openmm, unit, Topology, Simulation, DCDReporter, StateDataReporter, PDBFile, Element
    except Exception as e:
        raise ImportError(
            "This ozone example requires OpenMM (`pip install openmm`) and its OpenMM app layer."
        ) from e


def ozone_topology():
    openmm, unit, Topology, Simulation, DCDReporter, StateDataReporter, PDBFile, Element = _require_openmm()
    top = Topology()
    chain = top.addChain()
    res = top.addResidue("O3", chain)
    O = Element.getBySymbol("O")
    a0 = top.addAtom("O", O, res)
    a1 = top.addAtom("O", O, res)
    a2 = top.addAtom("O", O, res)
    top.addBond(a0, a1)
    top.addBond(a1, a2)
    return top


def starting_positions(r0=None, theta0=None):
    openmm, unit, *_ = _require_openmm()
    if r0 is None:
        r0 = 1.278 * unit.angstrom
    if theta0 is None:
        theta0 = 2.038 * unit.radian

    th = theta0.value_in_unit(unit.radian)
    r = r0.value_in_unit(unit.angstrom)
    pos = np.zeros((3, 3), dtype=float)  # Å
    pos[0, 0] = r
    pos[2, 0] = np.cos(th) * r
    pos[2, 1] = np.sin(th) * r
    return unit.Quantity(pos, unit.angstrom)


def system_reference_harmonic(
    *,
    r0=None,
    k_bond=None,
    theta0=None,
    k_angle=None,
    mass=None,
):
    openmm, unit, *_ = _require_openmm()
    if r0 is None:
        r0 = 1.278 * unit.angstrom
    if k_bond is None:
        k_bond = 290.1 * unit.kilocalories_per_mole / (unit.angstrom**2)
    if theta0 is None:
        theta0 = 2.038 * unit.radian
    if k_angle is None:
        k_angle = 900.0 * unit.kilocalories_per_mole / (unit.radian**2)
    if mass is None:
        mass = 15.999 * unit.amu

    system = openmm.System()
    for _ in range(3):
        system.addParticle(mass)

    fb = openmm.HarmonicBondForce()
    fb.addBond(0, 1, r0, k_bond)
    fb.addBond(1, 2, r0, k_bond)
    system.addForce(fb)

    fa = openmm.HarmonicAngleForce()
    fa.addAngle(0, 1, 2, theta0, k_angle)
    system.addForce(fa)

    return system


def system_target_morse_anharm(
    *,
    r0=None,
    De=None,
    a=None,
    theta0=None,
    k2=None,
    k4=None,
    A13=None,
    mass=None,
):
    """Target: Morse bonds + anharmonic (quartic) angle + weak 1-3 repulsion."""
    openmm, unit, *_ = _require_openmm()
    if r0 is None:
        r0 = 1.330 * unit.angstrom
    if De is None:
        De = 120.0 * unit.kilocalories_per_mole
    if a is None:
        a = 2.2 / unit.angstrom
    if theta0 is None:
        theta0 = 2.038 * unit.radian
    if k2 is None:
        k2 = 700.0 * unit.kilocalories_per_mole
    if k4 is None:
        k4 = 2500.0 * unit.kilocalories_per_mole
    if A13 is None:
        A13 = 1e-6 * (unit.kilojoules_per_mole * unit.nanometer**12)
    if mass is None:
        mass = 15.999 * unit.amu

    system = openmm.System()
    for _ in range(3):
        system.addParticle(mass)

    morse = openmm.CustomBondForce("De*(1-exp(-a*(r-r0)))^2")
    morse.addPerBondParameter("De")
    morse.addPerBondParameter("a")
    morse.addPerBondParameter("r0")
    De_val = De.value_in_unit(unit.kilojoules_per_mole)
    a_val = a.value_in_unit(unit.nanometer**-1)
    r0_val = r0.value_in_unit(unit.nanometer)
    morse.addBond(0, 1, [De_val, a_val, r0_val])
    morse.addBond(1, 2, [De_val, a_val, r0_val])
    system.addForce(morse)

    ang = openmm.CustomAngleForce("0.5*k2*(theta-theta0)^2 + k4*(theta-theta0)^4")
    ang.addPerAngleParameter("k2")
    ang.addPerAngleParameter("k4")
    ang.addPerAngleParameter("theta0")
    k2_val = k2.value_in_unit(unit.kilojoules_per_mole)
    k4_val = k4.value_in_unit(unit.kilojoules_per_mole)
    th0_val = theta0.value_in_unit(unit.radian)
    ang.addAngle(0, 1, 2, [k2_val, k4_val, th0_val])
    system.addForce(ang)

    rep = openmm.CustomBondForce("A/(r^12)")
    rep.addPerBondParameter("A")
    A_val = A13.value_in_unit(unit.kilojoules_per_mole * unit.nanometer**12)
    rep.addBond(0, 2, [A_val])
    system.addForce(rep)

    return system


def get_platform(platform_name: str):
    openmm, *_ = _require_openmm()
    return openmm.Platform.getPlatformByName(platform_name)


def run_md(
    system,
    outdir: Path,
    *,
    temperature_K: float,
    n_steps: int,
    report_interval: int,
    platform,
    seed: int,
    timestep_fs: float = 2.0,
    friction_inv_ps: float = 1.0,
):
    openmm, unit, Topology, Simulation, DCDReporter, StateDataReporter, PDBFile, Element = _require_openmm()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    top = ozone_topology()
    pos = starting_positions()

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        friction_inv_ps / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(seed))

    sim = Simulation(top, system, integrator, platform=platform)
    sim.context.setPositions(pos)
    sim.minimizeEnergy()

    sim.reporters.append(DCDReporter(str(outdir / "traj.dcd"), int(report_interval)))
    sim.reporters.append(
        StateDataReporter(
            str(outdir / "energies.csv"),
            int(report_interval),
            step=True,
            potentialEnergy=True,
            separator=",",
        )
    )

    with open(outdir / "ozone.pdb", "w") as f:
        PDBFile.writeFile(top, pos, f)

    sim.step(int(n_steps))
