#!/usr/bin/env python3
"""
Zadatak 4.3 - Varijacijski Monte Carlo za 1D beskonacnu jamu
sa linearnom smetnjom V_s(x)=V0*x/L.

Unutar jame: 0 < x < L
Probna funkcija: psi(x, alpha) = A*x*(x-L)*exp(-alpha*x)

U kodu koristimo ekvivalentan oblik psi = A*x*(L-x)*exp(-alpha*x),
jer ukupni predznak ne mijenja |psi|^2 ni energiju.
"""

from __future__ import annotations

import argparse 
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1.0e-14


@dataclass
class VMCResult:
    alpha: float
    block_index: np.ndarray
    block_energy: np.ndarray
    running_energy: np.ndarray
    mean_energy: float
    error: float
    acceptance: float
    final_step: float


def psi_unnormalized(x: np.ndarray, alpha: float, L: float) -> np.ndarray:
    """postoji samo u [0,L]."""
    inside = (x > 0.0) & (x < L)
    psi = np.zeros_like(x, dtype=float)
    psi[inside] = x[inside] * (L - x[inside]) * np.exp(-alpha * x[inside])
    return psi


def psi_squared(x: np.ndarray, alpha: float, L: float) -> np.ndarray:
    psi = psi_unnormalized(x, alpha, L)
    return psi * psi


def local_energy(
    x: np.ndarray,
    alpha: float,
    L: float,
    V0: float,
    kinetic_prefactor: float,
) -> np.ndarray:
    """
    Lokalna energija:

    E_L = K * [2 + 2*alpha*(L - 2x) - alpha^2*x*(L-x)]/[x*(L-x)]
          + V0*x/L

    gdje je K = hbar^2/(2m).
    """
    q = np.maximum(x * (L - x), EPS)
    numerator = 2.0 + 2.0 * alpha * (L - 2.0 * x) - alpha * alpha * q
    return kinetic_prefactor * numerator / q + V0 * x / L


def normalization_constant(alpha: float, L: float, n_grid: int = 20_001) -> float:
    x = np.linspace(0.0, L, n_grid)
    y = psi_unnormalized(x, alpha, L)
    norm2 = np.trapezoid(y * y, x) #int |psi|^2 dx=1
    return 1.0 / np.sqrt(norm2)


def quadrature_energy(
    alpha: float,
    L: float,
    V0: float,
    kinetic_prefactor: float,
    n_grid: int = 20_001,
) -> float:
    """Direktna numericka integracija <psi|H|psi>/<psi|psi>, za  referencu."""
    x = np.linspace(0.0, L, n_grid)[1:-1]
    psi2 = psi_squared(x, alpha, L)
    e_loc = local_energy(x, alpha, L, V0, kinetic_prefactor)
    return float(np.trapezoid(psi2 * e_loc, x) / np.trapezoid(psi2, x))


#1 MC simulacija za 1 alfa

def vmc_infinite_well(
    *,
    alpha: float,
    L: float = 1.0,
    V0: float = 1.0,
    kinetic_prefactor: float = 1.0,
    n_walkers: int = 500,
    n_steps: int = 500,
    n_blocks: int = 210,
    n_skip: int = 10,
    step_size: float = 0.5,
    seed: int = 430,
    tune_step: bool = True,
) -> VMCResult:
    if n_skip >= n_blocks:
        raise ValueError("n_skip mora biti manji od n_blocks.")
    if L <= 0:
        raise ValueError("L mora biti pozitivan.")

    rng = np.random.default_rng(seed)

    # Uniformna inicijalizacija u jami. Rubovi se izbjegavaju da E_L ne divergira.
    x = rng.uniform(0.05 * L, 0.95 * L, size=n_walkers)
    prob = psi_squared(x, alpha, L)
    energy = local_energy(x, alpha, L, V0, kinetic_prefactor)

    dk = float(step_size)
    accepted_total = 0
    attempted_total = 0
    block_energies: list[float] = []
    running_energies: list[float] = []

    for ib in range(1, n_blocks + 1):
        accepted_block = 0
        attempted_block = 0
        step_energy_sum = 0.0

        for _ in range(n_steps):
            proposal = x + rng.uniform(-dk, dk, size=n_walkers)
            prob_new = psi_squared(proposal, alpha, L)
        #za svaki korak predoži novi položaj i izračunaj gustoću-Metropolis

            ratio = prob_new / np.maximum(prob, EPS)
            accept = (ratio >= 1.0) | (rng.random(n_walkers) <= ratio)

            x[accept] = proposal[accept]
            prob[accept] = prob_new[accept]
            energy[accept] = local_energy(x[accept], alpha, L, V0, kinetic_prefactor)

            accepted_step = int(np.count_nonzero(accept))
            accepted_block += accepted_step
            attempted_block += n_walkers
            accepted_total += accepted_step
            attempted_total += n_walkers

            if ib > n_skip:
                step_energy_sum += float(np.mean(energy))

        block_acceptance = accepted_block / attempted_block
        if tune_step and ib <= n_skip:
            if block_acceptance > 0.5:
                dk *= 1.05
            else:
                dk *= 0.95

        if ib > n_skip:
            block_energy = step_energy_sum / n_steps
            block_energies.append(block_energy)
            running_energies.append(float(np.mean(block_energies)))

    block_array = np.array(block_energies)
    running_array = np.array(running_energies)
    mean_energy = float(np.mean(block_array))

    if len(block_array) > 1:
        error = float(np.std(block_array, ddof=1) / np.sqrt(len(block_array)))
    else:
        error = float("nan")

    return VMCResult(
        alpha=alpha,
        block_index=np.arange(1, len(block_array) + 1),
        block_energy=block_array,
        running_energy=running_array,
        mean_energy=mean_energy,
        error=error,
        acceptance=accepted_total / attempted_total,
        final_step=dk,
    )


def save_equilibration_plot(result: VMCResult, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(result.block_index, result.block_energy, color="deepskyblue", lw=1.2, label=r"$E_b$")
    plt.plot(result.block_index, result.running_energy, color="navy", lw=2.0, label=r"$\langle E\rangle$")
    plt.axhline(result.mean_energy, color="crimson", ls="--", lw=1.4)
    plt.xlabel("blokovi")
    plt.ylabel("E")
    plt.title(
        rf"Ravnotezno uzorkovanje, $\alpha={result.alpha:.4g}$: "
        rf"$E={result.mean_energy:.6f}\pm{result.error:.2g}$"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_energy_scan_plot(scan_data: np.ndarray, smooth_data: np.ndarray, path: Path) -> None:
    alpha = scan_data[:, 0]
    energy = scan_data[:, 1]
    error = scan_data[:, 2]
    best = int(np.argmin(energy))

    plt.figure(figsize=(10, 6))
    plt.errorbar(alpha, energy, yerr=error, fmt="o", capsize=3, label="VMC")
    plt.plot(smooth_data[:, 0], smooth_data[:, 1], color="black", lw=1.4, alpha=0.8, label="numericki integral")
    plt.scatter([alpha[best]], [energy[best]], color="crimson", zorder=5, label="najmanja VMC energija")
    plt.xlabel(r"varijacijski parametar $\alpha$")
    plt.ylabel("E")
    plt.title(r"Ovisnost energije o parametru $\alpha$")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_wavefunction_plot(
    alpha: float,
    L: float,
    path: Path,
    sample_positions: np.ndarray | None = None,
) -> None:
    x = np.linspace(0.0, L, 1000)
    A = normalization_constant(alpha, L)
    psi = A * psi_unnormalized(x, alpha, L)

    plt.figure(figsize=(10, 6))
    plt.plot(x, psi, color="navy", lw=2.0, label=rf"$\psi(x),\ \alpha={alpha:.4g}$")

    if sample_positions is not None and len(sample_positions) > 0:
        hist, bins = np.histogram(sample_positions, bins=60, range=(0.0, L), density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        # Histogram je za |psi|^2; skaliramo ga samo kao vizualnu provjeru oblika.
        hist_scaled = hist / np.max(hist) * np.max(psi)
        plt.plot(centers, hist_scaled, color="deepskyblue", alpha=0.7, lw=1.2, label="uzorkovani oblik, skaliran")

    plt.xlabel("x")
    plt.ylabel(r"normirana $\psi(x)$")
    plt.title("Normirana probna funkcija osnovnog stanja")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_alpha_values(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Lista alpha vrijednosti je prazna.")
    return np.array(values, dtype=float)


def save_block_table(result: VMCResult, path: Path) -> None:
    table = np.column_stack((result.block_index, result.block_energy, result.running_energy))
    np.savetxt(
        path,
        table,
        header="blok E_bloka E_kumulativno",
        fmt=["%d", "%.10e", "%.10e"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VMC za Zadatak 4.3: beskonacna jama + linearna smetnja.")
    parser.add_argument("--L", type=float, default=1.0, help="Sirina jame.")
    parser.add_argument("--V0", type=float, default=1.0, help="Jacina linearne smetnje.")
    parser.add_argument(
        "--K",
        type=float,
        default=1.0,
        help="Kineticki prefaktor K=hbar^2/(2m). Default: 1.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Alpha za glavni run. Ako se izostavi, koristi se najbolji iz skena.")
    parser.add_argument("--walkers", type=int, default=500, help="Broj setaca za glavni run.")
    parser.add_argument("--steps", type=int, default=500, help="Broj Metropolis koraka po bloku.")
    parser.add_argument("--blocks", type=int, default=210, help="Broj blokova.")
    parser.add_argument("--skip", type=int, default=10, help="Broj pocetnih blokova koji se odbacuju.")
    parser.add_argument("--step-size", type=float, default=0.5, help="Pocetna maksimalna duljina koraka.")
    parser.add_argument("--seed", type=int, default=430, help="Sjeme generatora slucajnih brojeva.")
    parser.add_argument("--outdir", type=Path, default=Path("z43_outputs"), help="Direktorij za izlazne datoteke.")
    parser.add_argument(
        "--scan-alpha",
        type=str,
        default="-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.03,0.10,0.25,0.5,1.0,1.5,2.0,2.5,3.0",
        help="Vrijednosti alpha za skeniranje, odvojene zarezima.",
    )
    parser.add_argument("--scan-walkers", type=int, default=300, help="Broj setaca za alpha-scan.")
    parser.add_argument("--scan-steps", type=int, default=250, help="Broj koraka po bloku za alpha-scan.")
    parser.add_argument("--scan-blocks", type=int, default=90, help="Broj blokova za alpha-scan.")
    parser.add_argument("--scan-skip", type=int, default=10, help="Broj odbacenih blokova za alpha-scan.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    alpha_values = parse_alpha_values(args.scan_alpha)
    print("Skeniram varijacijski parametar alpha...")
    scan_rows = []
    for i, alpha in enumerate(alpha_values, start=1):
        result = vmc_infinite_well(
            alpha=float(alpha),
            L=args.L,
            V0=args.V0,
            kinetic_prefactor=args.K,
            n_walkers=args.scan_walkers,
            n_steps=args.scan_steps,
            n_blocks=args.scan_blocks,
            n_skip=args.scan_skip,
            step_size=args.step_size,
            seed=args.seed + 1000 * i,
        )
        scan_rows.append((result.alpha, result.mean_energy, result.error, result.acceptance))
        print(
            f"  alpha={result.alpha: .4f}: "
            f"E={result.mean_energy:.8f} +/- {result.error:.2e}, "
            f"acc={100.0 * result.acceptance:.1f}%"
        )

    scan_data = np.array(scan_rows)
    np.savetxt(
        args.outdir / "E_vs_alpha.dat",
        scan_data,
        header="alpha E srednja_greska prihvacanje",
        fmt=["%.8f", "%.10e", "%.10e", "%.8f"],
    )

    smooth_alpha = np.linspace(float(np.min(alpha_values)), float(np.max(alpha_values)), 300)
    smooth_energy = np.array([quadrature_energy(a, args.L, args.V0, args.K) for a in smooth_alpha])
    smooth_data = np.column_stack((smooth_alpha, smooth_energy))
    np.savetxt(
        args.outdir / "E_vs_alpha_integral.dat",
        smooth_data,
        header="alpha E_integral",
        fmt=["%.8f", "%.10e"],
    )
    save_energy_scan_plot(scan_data, smooth_data, args.outdir / "E_vs_alpha.png")

    best_index = int(np.argmin(scan_data[:, 1]))
    best_alpha = float(scan_data[best_index, 0]) if args.alpha is None else float(args.alpha)
    print(f"\nGlavni run za alpha={best_alpha:.6g}")

    final = vmc_infinite_well(
        alpha=best_alpha,
        L=args.L,
        V0=args.V0,
        kinetic_prefactor=args.K,
        n_walkers=args.walkers,
        n_steps=args.steps,
        n_blocks=args.blocks,
        n_skip=args.skip,
        step_size=args.step_size,
        seed=args.seed,
    )
    save_block_table(final, args.outdir / "E_blocks.dat")
    save_equilibration_plot(final, args.outdir / "E_equilibration.png")
    save_wavefunction_plot(final.alpha, args.L, args.outdir / "wavefunction.png")

    print(
        f"E = {final.mean_energy:.8f} +/- {final.error:.2e}, "
        f"prihvacanje = {100.0 * final.acceptance:.1f}%, "
        f"konacni korak = {final.final_step:.5f}"
    )
    print(f"Izlazi su spremljeni u: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
