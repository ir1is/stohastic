import argparse
import math
from pathlib import Path

import numpy as np


# Default parametri su manji od C primjera da se program moze brzo pokrenuti.
# Za originalne parametre iz C koda pokreni program s opcijom --full.
L = 32
DKT = 0.1
KT0 = 1.0
NT = 40
JV = 1.0

NB_SKIP = 50
NK = 1000
NB = 200

WRITE_SPINS = False
SEED = 2


def energija(s, jv):
    """Ukupna energija 2D Ising modela s periodicnim rubnim uvjetima."""
    desno = np.roll(s, -1, axis=0)
    gore = np.roll(s, -1, axis=1)
    return -jv * np.sum(s * (desno + gore))


def magnetizacija(s):
    """Ukupna magnetizacija sustava."""
    return np.sum(s)


def simulacija(l, dkt, kt0, nt, jv, nb_skip, nk, nb, write_spins, seed, out_dir):
    if nb <= nb_skip:
        raise ValueError("NB mora biti veci od NB_SKIP.")

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pocetna konfiguracija: svi spinovi su +1.
    s = np.ones((l, l), dtype=np.int8)

    e = float(energija(s, jv))
    m = float(magnetizacija(s))
    n_spinova = l * l

    print("Pocetna energija:", e)
    print("Pocetna magnetizacija:", m)
    print("Izlazni folder:", out_dir.resolve())

    fm = open(out_dir / "f_m.dat", "w")
    fE = open(out_dir / "f_E.dat", "w")
    fm2 = open(out_dir / "f_m2.dat", "w")
    fEb = open(out_dir / "f_Eb.dat", "w")
    fc = open(out_dir / "c.txt", "w")
    fsus = open(out_dir / "f_sus.dat", "w")
    fsG = open(out_dir / "f_spinG.dat", "w")
    fsD = open(out_dir / "f_spinD.dat", "w")

    try:
        for it in range(nt + 1):
            kt = kt0 + it * dkt

            sb_e = 0.0
            sb_e2 = 0.0
            sb_m = 0.0
            sb_m2 = 0.0
            reject = 0
            nb_eff = 0

            for ib in range(1, nb + 1):
                sk_e = 0.0
                sk_e2 = 0.0
                sk_m = 0.0
                sk_m2 = 0.0

                for ik in range(1, nk + 1):
                    i = rng.integers(0, l)
                    j = rng.integers(0, l)

                    stari_spin = int(s[i, j])
                    susjedi = int(
                        s[(i + 1) % l, j]
                        + s[(i - 1) % l, j]
                        + s[i, (j + 1) % l]
                        + s[i, (j - 1) % l]
                    )

                    # Za pokusaj okretanja spina s -> -s:
                    # dE = 2 * J * s * suma_susjeda
                    de = 2.0 * jv * stari_spin * susjedi
                    dm = -2.0 * stari_spin

                    if de <= 0.0 or rng.random() < math.exp(-de / kt):
                        s[i, j] = -stari_spin
                        e += de
                        m += dm
                    else:
                        reject += 1

                    if ib > nb_skip:
                        sk_e += e
                        sk_e2 += e * e
                        sk_m += m
                        sk_m2 += m * m

                if ib > nb_skip:
                    nb_eff = ib - nb_skip

                    blok_e = sk_e / nk
                    blok_m = sk_m / nk

                    sb_e += blok_e
                    sb_e2 += sk_e2 / nk
                    sb_m += blok_m
                    sb_m2 += sk_m2 / nk

                    srednja_e = sb_e / nb_eff
                    srednja_m = sb_m / nb_eff

                    var_e = sb_e2 / nb_eff - srednja_e * srednja_e
                    var_m = sb_m2 / nb_eff - srednja_m * srednja_m

                    sigma_e = math.sqrt(max(var_e, 0.0)) / math.sqrt(nb_eff)
                    sigma_m = math.sqrt(max(var_m, 0.0)) / math.sqrt(nb_eff)

                    fE.write("{}\t{}\t{}\n".format(ib, srednja_e, sigma_e))
                    fm2.write("{}\t{}\t{}\n".format(ib, srednja_m, sigma_m))
                    fEb.write("{}\t{}\n".format(ib, blok_e))

                    if write_spins:
                        plus = np.argwhere(s == 1)
                        minus = np.argwhere(s == -1)

                        for poz in plus:
                            fsG.write("{}  {}\n".format(poz[0] + 1, poz[1] + 1))
                        for poz in minus:
                            fsD.write("{}  {}\n".format(poz[0] + 1, poz[1] + 1))

                        fsG.write("\n\n")
                        fsD.write("\n\n")

            srednja_e = sb_e / nb_eff
            srednja_m = sb_m / nb_eff

            c = (sb_e2 / nb_eff - srednja_e * srednja_e) / (kt * kt)
            sus = (sb_m2 / nb_eff - srednja_m * srednja_m) / kt
            prihvacanje = 1.0 - float(reject) / float(nb * nk)

            fc.write("{}\t{}\n".format(kt, c / n_spinova))
            fsus.write("{}\t{}\n".format(kt, sus / n_spinova))
            fm.write("{}\t{}\n".format(kt, srednja_m / n_spinova))

            fE.write("\n\n")
            fEb.write("\n\n")

            print(
                "kT={:.2f}  <E>/spin={:.6f}  <M>/spin={:.6f}  C/spin={:.6f}  sus/spin={:.6f}  prihvacanje={:.4f}".format(
                    kt,
                    srednja_e / n_spinova,
                    srednja_m / n_spinova,
                    c / n_spinova,
                    sus / n_spinova,
                    prihvacanje,
                )
            )

    finally:
        fm.close()
        fE.close()
        fm2.close()
        fEb.close()
        fc.close()
        fsus.close()
        fsG.close()
        fsD.close()

    print("Gotovo. Datoteke su zapisane u:", out_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description="2D Ising model - Metropolis algoritam")
    parser.add_argument("--l", type=int, default=L, help="Dimenzija resetke L x L")
    parser.add_argument("--dkt", type=float, default=DKT, help="Temperaturni korak")
    parser.add_argument("--kt0", type=float, default=KT0, help="Pocetna temperatura")
    parser.add_argument("--nt", type=int, default=NT, help="Broj temperaturnih koraka")
    parser.add_argument("--jv", type=float, default=JV, help="Konstanta izmjene energije")
    parser.add_argument("--nb-skip", type=int, default=NB_SKIP, help="Broj odbacenih blokova")
    parser.add_argument("--nk", type=int, default=NK, help="Broj koraka po bloku")
    parser.add_argument("--nb", type=int, default=NB, help="Ukupan broj blokova")
    parser.add_argument("--seed", type=int, default=SEED, help="Sjeme generatora")
    parser.add_argument("--out-dir", default="ising_output", help="Folder za izlazne datoteke")
    parser.add_argument("--write-spins", action="store_true", help="Spremi koordinate spinova")
    parser.add_argument("--full", action="store_true", help="Koristi parametre iz C primjera")
    args = parser.parse_args()

    if args.full:
        args.l = 128
        args.nb_skip = 1000
        args.nk = 10000
        args.nb = 11000

    simulacija(
        l=args.l,
        dkt=args.dkt,
        kt0=args.kt0,
        nt=args.nt,
        jv=args.jv,
        nb_skip=args.nb_skip,
        nk=args.nk,
        nb=args.nb,
        write_spins=args.write_spins,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
