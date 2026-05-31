import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ucitaj_xy(putanja):
    """Ucita datoteku s dva stupca: x y."""
    podaci = np.loadtxt(putanja)
    return podaci[:, 0], podaci[:, 1]


def ucitaj_blokove(putanja):
    """
    f_E.dat i f_Eb.dat imaju blokove odvojene praznim redovima.
    Ova funkcija vraca listu blokova, a svaki blok je numpy matrica.
    """
    tekst = Path(putanja).read_text().strip()
    blokovi = []

    for blok in tekst.split("\n\n"):
        redovi = []
        for red in blok.splitlines():
            red = red.strip()
            if red:
                redovi.append([float(x) for x in red.split()])
        if redovi:
            blokovi.append(np.array(redovi))

    return blokovi


def spremi_graf(x, y, xlabel, ylabel, naslov, ime, out_dir, teorijski_tc=True):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-", linewidth=2, markersize=4)

    if teorijski_tc:
        plt.axvline(2.269, color="red", linestyle="--", linewidth=1.5, label="kT_c ~= 2.269")
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(naslov)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / ime, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plotanje rezultata 2D Ising simulacije.")
    parser.add_argument("--data-dir", default="ising_output_Tc", help="Folder s .dat datotekama.")
    parser.add_argument("--out-dir", default="ising_plots", help="Folder za spremljene grafove.")
    parser.add_argument("--l", type=int, default=32, help="Dimenzija resetke L x L.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_spinova = args.l * args.l

    # Datoteke s jednim redom po temperaturi.
    kt, m = ucitaj_xy(data_dir / "f_m.dat")
    kt_c, c = ucitaj_xy(data_dir / "c.txt")
    kt_sus, sus = ucitaj_xy(data_dir / "f_sus.dat")

    # Energija je zapisana blokovski u f_E.dat.
    # Uzimamo zadnji red svakog temperaturnog bloka:
    # stupci su: ib, <E>, sigma_E.
    e_blokovi = ucitaj_blokove(data_dir / "f_E.dat")
    e = np.array([blok[-1, 1] for blok in e_blokovi]) / n_spinova
    sigma_e = np.array([blok[-1, 2] for blok in e_blokovi]) / n_spinova

    # Ako se broj blokova i temperatura ne poklapa, skrati na najmanju duljinu.
    n = min(len(kt), len(e), len(c), len(sus))
    kt = kt[:n]
    m = m[:n]
    e = e[:n]
    sigma_e = sigma_e[:n]
    c = c[:n]
    sus = sus[:n]

    # 1) Potpisana magnetizacija.
    spremi_graf(
        kt,
        m,
        "kT",
        "<M>/N",
        "Potpisana magnetizacija",
        "magnetizacija_signed.png",
        out_dir,
    )

    # 2) Apsolutna magnetizacija.
    # Napomena: ovo je |<M>|, jer f_m.dat sadrzi vec usrednjenu potpisanu M.
    # Fizikalno je jos bolje u simulaciji racunati <|M|>.
    spremi_graf(
        kt,
        np.abs(m),
        "kT",
        "|<M>|/N",
        "Apsolutna magnetizacija",
        "magnetizacija_abs.png",
        out_dir,
    )

    # 3) Energija po spinu.
    plt.figure(figsize=(7, 5))
    plt.errorbar(kt, e, yerr=sigma_e, fmt="o-", linewidth=2, markersize=4, capsize=3)
    plt.axvline(2.269, color="red", linestyle="--", linewidth=1.5, label="kT_c ~= 2.269")
    plt.xlabel("kT")
    plt.ylabel("<E>/N")
    plt.title("Energija po spinu")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "energija.png", dpi=200)
    plt.close()

    # 4) Toplinski kapacitet po spinu.
    spremi_graf(
        kt_c[:n],
        c,
        "kT",
        "C/N",
        "Toplinski kapacitet po spinu",
        "toplinski_kapacitet.png",
        out_dir,
    )

    # 5) Susceptibilnost po spinu.
    spremi_graf(
        kt_sus[:n],
        sus,
        "kT",
        "chi/N",
        "Susceptibilnost po spinu",
        "susceptibilnost.png",
        out_dir,
    )

    # 6) Kombinirani pregled.
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    axs[0, 0].plot(kt, np.abs(m), "o-")
    axs[0, 0].set_title("Apsolutna magnetizacija")
    axs[0, 0].set_ylabel("|<M>|/N")

    axs[0, 1].errorbar(kt, e, yerr=sigma_e, fmt="o-", capsize=3)
    axs[0, 1].set_title("Energija")
    axs[0, 1].set_ylabel("<E>/N")

    axs[1, 0].plot(kt_c[:n], c, "o-")
    axs[1, 0].set_title("Toplinski kapacitet")
    axs[1, 0].set_ylabel("C/N")

    axs[1, 1].plot(kt_sus[:n], sus, "o-")
    axs[1, 1].set_title("Susceptibilnost")
    axs[1, 1].set_ylabel("chi/N")

    for ax in axs.flat:
        ax.axvline(2.269, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("kT")
        ax.grid(True, alpha=0.3)

    fig.suptitle("2D Ising model - Metropolis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "ising_sve.png", dpi=200)
    plt.close(fig)

    print("Grafovi su spremljeni u:", out_dir.resolve())
    print("Maksimum C je kod kT =", kt_c[np.argmax(c)])
    print("Maksimum susceptibilnosti je kod kT =", kt_sus[np.argmax(sus)])


if __name__ == "__main__":
    main()
