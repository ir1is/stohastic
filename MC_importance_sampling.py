import numpy as np
import time

def f(x):
    return np.exp(-x**2)


# UNIFORM p(x)=1
def mc_uniform(N):
    start = time.time()

    x = np.random.rand(N)
    values = f(x)

    Fn = np.mean(values)
    sigma = np.std(values)

    t = time.time() - start

    return Fn, sigma, sigma/np.sqrt(N), t

# IMPORTANCE SAMPLING
def mc_importance(N):
    start = time.time()

    # generiranje x
    r = np.random.rand(N)
    x = -np.log(1 - r*(1 - np.exp(-1)))   # ✔ ispravno

    A = np.e / (np.e - 1)
    p = A * np.exp(-x)

    values = f(x)/p

    Fn = np.mean(values)
    sigma = np.std(values)

    t = time.time() - start

    return Fn, sigma, sigma/np.sqrt(N), t

# -----------------------------
# RUN
# -----------------------------
N1 = int(5e6)
N2 = int(4e5)

Fn1, s1, e1, t1 = mc_uniform(N1)
Fn2, s2, e2, t2 = mc_importance(N2)

# spremanje u datoteku
with open("MC_python.txt", "w") as f:
    f.write("              p(x)=1           p(x)=Ae^-x\n")
    f.write(f"N           {N1}           {N2}\n")
    f.write(f"Fn          {Fn1:.6f}       {Fn2:.6f}\n")
    f.write(f"sigma       {s1:.6f}       {s2:.6f}\n")
    f.write(f"sigma/sqrtN {e1:.6f}       {e2:.6f}\n")
    f.write(f"time        {t1:.3f}       {t2:.3f}\n")