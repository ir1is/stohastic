import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt


# PARAMETRI RESETKE
RO = 0.8        # gustoca = broj atoma / povrsina; 0.8 daje gusci sustav koji vise nalikuje tekucini
MY2 = 4         # broj parova redaka u y smjeru; povecano s 3 na 4 za sire podrucje

# PARAMETRI SIMULACIJE
NB_SKIP = 100   # broj blokova koji se preskacu kod usrednjavanja
NB = 300        # broj blokova
NK = 50         # broj koraka po bloku
NW = 4          # broj setaca / konfiguracija
KT = 0.1        # temperatura u jedinicama kB*T
D0 = 0.1        # pocetna maksimalna duljina pomaka
SEED = 12345    # sjeme generatora slucajnih brojeva

# PARAMETRI RAZDIOBE g(r)
NG = 500        # broj intervala za radijalnu distribucijsku funkciju


def napravi_trokutastu_resetku(ro=RO, my2=MY2):
    """Generira atome u vrhovima jednakostranicnih trokuta."""
    mx = int(my2 * math.sqrt(3.0) + 0.5)
    npart = 2 * mx * my2

    povrsina = npart / ro
    omjer = mx / (my2 * math.sqrt(3.0))  # Lx / Ly
    lx = math.sqrt(povrsina * omjer)
    ly = math.sqrt(povrsina / omjer)
    a = lx / mx

    x01 = 0.25 * a
    y01 = 0.25 * a * math.sqrt(3.0)
    x02 = x01 + 0.5 * a
    y02 = y01 + 0.5 * a * math.sqrt(3.0)

    polozaji = []
    for nx in range(mx):
        for ny in range(my2):
            x1 = x01 + nx * a
            y1 = y01 + ny * a * math.sqrt(3.0)
            x2 = x02 + nx * a
            y2 = y02 + ny * a * math.sqrt(3.0)
            polozaji.append([x1, y1])
            polozaji.append([x2, y2])

    return polozaji, lx, ly, a, mx, npart


def minimalna_slika(dx, dy, lx, ly):
    """Aproksimacija minimalnih slika za periodicne rubne uvjete."""
    lxp = lx / 2.0
    lyp = ly / 2.0

    if dx > lxp:
        dx -= lx
    if dx < -lxp:
        dx += lx
    if dy > lyp:
        dy -= ly
    if dy < -lyp:
        dy += ly

    return dx, dy


def energija(konfiguracija, lx, ly):
    """Racuna Lennard-Jonesovu potencijalnu energiju sustava."""
    npart = len(konfiguracija)
    lminp2 = (min(lx, ly) / 2.0) ** 2
    suma = 0.0

    for i in range(npart - 1):
        xi, yi = konfiguracija[i]
        for j in range(i + 1, npart):
            dx = xi - konfiguracija[j][0]
            dy = yi - konfiguracija[j][1]
            dx, dy = minimalna_slika(dx, dy, lx, ly)
            r2 = dx * dx + dy * dy

            if 0.0 < r2 <= lminp2:
                sr2 = 1.0 / r2
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6
                suma += sr12 - sr6

    return 4.0 * suma


def spremi_xy(polozaji, putanja):
    with open(putanja, "w", encoding="utf-8") as f:
        for x, y in polozaji:
            f.write(f"{x:.8f}\t{y:.8f}\n")


def simuliraj():
    pocetak = time.time()
    random.seed(SEED)
    try:
        izlaz = Path(__file__).resolve().parent
    except NameError:
        izlaz = Path.cwd()

    pocetni_polozaji, lx, ly, a, mx, npart = napravi_trokutastu_resetku()
    povrsina = lx * ly
    lminp = min(lx, ly) / 2.0
    dr = lminp / NG

    spremi_xy(pocetni_polozaji, izlaz / "xy.dat")

    konfiguracije = [
        [p[:] for p in pocetni_polozaji]
        for _ in range(NW)
    ]
    energije = [energija(konf, lx, ly) for konf in konfiguracije]

    gr = [0.0 for _ in range(NG)]
    broj_udaljenosti = 0.0
    broj_udaljenosti_unutar = 0.0

    dx_max = D0
    odbijeno = 0.0
    sb_e = 0.0
    sb_e2 = 0.0

    podaci_e = []

    print("TROKUTASTA RESETKA")
    print(f"{npart} atoma rasporedjeno u {2 * MY2} redaka i {mx} stupaca")
    print(f"Lx = {lx:.6f}, Ly = {ly:.6f}, gustoca = {npart / povrsina:.6f}")
    print(f"Prvi susjedi udaljeni su za a = {a:.6f}")
    print(f"Pocetna energija jedne konfiguracije: {energije[0]:.6f}")

    for ib in range(1, NB + 1):
        sk_e = 0.0
        sk_e2 = 0.0

        for _ in range(NK):
            for iw in range(NW):
                stara = konfiguracije[iw]
                probna = []

                for x, y in stara:
                    xp = x + dx_max * (2.0 * random.random() - 1.0)
                    yp = y + dx_max * (2.0 * random.random() - 1.0)

                    if xp < 0.0:
                        xp += lx
                    if xp > lx:
                        xp -= lx
                    if yp < 0.0:
                        yp += ly
                    if yp > ly:
                        yp -= ly

                    probna.append([xp, yp])

                e_probna = energija(probna, lx, ly)
                de = e_probna - energije[iw]

                prihvati = True
                if de > 0.0 and math.exp(-de / KT) <= random.random():
                    prihvati = False

                if prihvati:
                    konfiguracije[iw] = probna
                    energije[iw] = e_probna
                else:
                    odbijeno += 1.0 / NK / NW

                if ib > NB_SKIP:
                    sk_e += energije[iw] / NW
                    sk_e2 += energije[iw] * energije[iw] / NW

                    konf = konfiguracije[iw]
                    for i in range(npart - 1):
                        xi, yi = konf[i]
                        for j in range(i + 1, npart):
                            rxij = xi - konf[j][0]
                            ryij = yi - konf[j][1]

                            # ZAD1: aproksimacija minimalnih slika
                            rxij, ryij = minimalna_slika(rxij, ryij, lx, ly)

                            # ZAD2: udaljenost cestica i-j
                            rij2 = rxij * rxij + ryij * ryij
                            rij = math.sqrt(rij2)

                            # ZAD3: spremanje udaljenosti u odgovarajuci interval
                            if rij < lminp:
                                ir = int(rij / dr)
                                if 0 <= ir < NG:
                                    gr[ir] += 1.0
                                    broj_udaljenosti_unutar += 1.0

                            # ZAD4: brojanje svih izracunatih udaljenosti
                            broj_udaljenosti += 1.0

        acc_ib = 1.0 - odbijeno / ib
        if acc_ib > 0.5:
            dx_max *= 1.05
        if acc_ib < 0.5:
            dx_max *= 0.95

        if ib > NB_SKIP:
            nb_eff = ib - NB_SKIP
            sb_e += sk_e / NK
            sb_e2 += sk_e2 / NK

            if nb_eff == 1:
                sigma_e = 0.0
            else:
                var = sb_e2 / nb_eff - (sb_e / nb_eff) ** 2
                sigma_e = math.sqrt(max(var, 0.0)) / math.sqrt(nb_eff - 1.0)

            podaci_e.append((ib, sk_e / NK, sb_e / nb_eff, sigma_e))

        if ib % 10 == 0 or ib == 1 or ib == NB:
            proteklo = time.time() - pocetak
            postotak = 100.0 * ib / NB
            procjena_ukupno = proteklo * NB / ib
            preostalo = max(procjena_ukupno - proteklo, 0.0)
            print(
                f"Blok {ib:4d}/{NB} "
                f"({postotak:5.1f} %) | "
                f"prihvacanje = {acc_ib * 100.0:6.2f} % | "
                f"dx = {dx_max:.5f} | "
                f"preostalo oko {preostalo / 60.0:.1f} min"
            )

    with open(izlaz / "E.dat", "w", encoding="utf-8") as f:
        f.write("# blok\tE_blok\tE_srednje\tsigma_E\n")
        for ib, e_blok, e_srednje, sigma_e in podaci_e:
            f.write(f"{ib}\t{e_blok:.10f}\t{e_srednje:.10f}\t{sigma_e:.10f}\n")

    # ZAD5: normiranje i ispis radijalne distribucijske funkcije
    podaci_gr = []
    with open(izlaz / "gr.dat", "w", encoding="utf-8") as f:
        f.write("# r\tg(r)\n")
        for ir in range(NG):
            r = (ir + 0.5) * dr
            skv = math.pi * (((ir + 1) * dr) ** 2 - (ir * dr) ** 2)
            g_r = gr[ir] * povrsina / (broj_udaljenosti * skv) if broj_udaljenosti > 0 else 0.0
            podaci_gr.append((r, g_r))
            f.write(f"{r:.10f}\t{g_r:.10f}\n")

    # Dodatna verzija normalizacije slicna prilozenom tudjem C rjesenju.
    with open(izlaz / "gr_norma_kao_u_primjeru.dat", "w", encoding="utf-8") as f:
        f.write("# r\tg_normirano_po_unutarnjim_udaljenostima\n")
        for ir in range(1, NG):
            r = ir * dr
            nazivnik = broj_udaljenosti_unutar * (2.0 * math.pi * ir * dr * dr)
            g_r = gr[ir] / nazivnik if nazivnik > 0.0 else 0.0
            f.write(f"{r:.10f}\t{g_r:.10f}\n")

    napravi_grafove(podaci_e, podaci_gr, izlaz)

    evani = 4.0 * npart * math.pi * RO * (0.2 / lminp**10 - 0.5 / lminp**4)
    print(f"Postotak prihvacenih pomaka: {acc_ib * 100.0:.2f} %")
    print(f"Srednja energija unutar Lmin/2: {podaci_e[-1][2]:.6f}")
    print(f"Procjena energije van Lmin/2: {evani:.6f}")
    print(f"Datoteke su spremljene u: {izlaz}")


def napravi_grafove(podaci_e, podaci_gr, izlaz):
    blokovi = [x[0] for x in podaci_e]
    e_blok = [x[1] for x in podaci_e]
    e_srednje = [x[2] for x in podaci_e]

    plt.figure(figsize=(9, 5))
    plt.plot(blokovi, e_blok, label="E po bloku", linewidth=1.0)
    plt.plot(blokovi, e_srednje, label="Srednja E", linewidth=2.0)
    plt.xlabel("Blok")
    plt.ylabel("Energija")
    plt.legend()
    plt.tight_layout()
    plt.savefig(izlaz / "energija.png", dpi=160)
    plt.close()

    r = [x[0] for x in podaci_gr]
    g = [x[1] for x in podaci_gr]

    plt.figure(figsize=(9, 5))
    plt.plot(r, g, linewidth=1.5)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.tight_layout()
    plt.savefig(izlaz / "gr.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    simuliraj()
