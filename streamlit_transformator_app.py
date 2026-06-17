
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

EPS = 1e-12
MAX_STROOM_FACTOR = 1.0e6
CURVE_PUNTEN = 40

MU0 = 4.0 * math.pi * 1e-7
MU_R_STANDAARD = 2000.0
KERN_OPPERVLAK_STANDAARD = 8.0e-4
MAGNETISCHE_LENGTE_STANDAARD = 0.30


def cabs(z: complex) -> float:
    waarde = float(abs(z))
    if np.isnan(waarde):
        return 0.0
    return waarde


def hoek_graden(z: complex) -> float:
    if cabs(z) <= EPS:
        return 0.0
    return float(np.degrees(np.angle(z)))


def normaliseer_hoek(hoek_deg: float) -> float:
    if not np.isfinite(hoek_deg):
        return float("nan")
    hoek = ((float(hoek_deg) + 180.0) % 360.0) - 180.0
    if hoek <= -180.0:
        hoek += 360.0
    return hoek


def fasehoek_stroom_tov_spanning(spanning: complex, stroom: complex) -> float:
    if cabs(spanning) <= EPS or cabs(stroom) <= EPS:
        return float("nan")
    return normaliseer_hoek(hoek_graden(stroom) - hoek_graden(spanning))


def format_hoek_signed(waarde_deg: float, decimalen: int = 1) -> str:
    if not np.isfinite(waarde_deg):
        return "—"
    return f"{waarde_deg:+.{decimalen}f}°"


def format_getal(waarde: float, eenheid: str = "", decimalen: int = 2) -> str:
    if not np.isfinite(waarde):
        return f"∞ {eenheid}".strip()
    return f"{waarde:.{decimalen}f} {eenheid}".strip()


def format_stroom(waarde: float, drempel_ma: float = 1.0) -> str:
    if not np.isfinite(waarde):
        return "∞ A"
    if abs(waarde) < drempel_ma:
        return f"{1000.0 * waarde:.1f} mA"
    return f"{waarde:.2f} A"


def format_ohm(waarde: float) -> str:
    if not np.isfinite(waarde):
        return "∞ Ω"
    if abs(waarde) < 1.0:
        return f"{1000.0 * waarde:.2f} mΩ"
    return f"{waarde:.2f} Ω"


def format_watt(waarde: float) -> str:
    if not np.isfinite(waarde):
        return "∞ W"
    if abs(waarde) >= 1000.0:
        return f"{waarde / 1000.0:.2f} kW"
    if abs(waarde) < 1.0:
        return f"{1000.0 * waarde:.2f} mW"
    return f"{waarde:.2f} W"


def format_va(waarde: float) -> str:
    if not np.isfinite(waarde):
        return "∞ VA"
    if abs(waarde) >= 1000.0:
        return f"{waarde / 1000.0:.2f} kVA"
    return f"{waarde:.2f} VA"


def polar_tekst(z: complex, eenheid: str = "") -> str:
    grootte = cabs(z)
    hoek = hoek_graden(z)
    if eenheid == "A":
        if grootte < 1.0:
            return f"{1000.0 * grootte:.1f} ∠ {hoek:.0f}° mA"
        return f"{grootte:.2f} ∠ {hoek:.0f}° A"
    suffix = f" {eenheid}" if eenheid else ""
    return f"{grootte:.2f} ∠ {hoek:.0f}°{suffix}"


def format_cosphi(waarde: float) -> str:
    if not np.isfinite(waarde):
        return "—"
    return format_getal(float(np.clip(waarde, -1.0, 1.0)), "", 3)


def parallel(z1: complex, z2: complex) -> complex:
    a1 = abs(z1)
    a2 = abs(z2)
    if a1 <= EPS or a2 <= EPS:
        return 0.0 + 0.0j
    if np.isinf(a1):
        return z2
    if np.isinf(a2):
        return z1
    y = 1.0 / z1 + 1.0 / z2
    if abs(y) <= EPS:
        return complex(np.inf)
    return 1.0 / y


def veilige_stroom(spanning: complex, impedantie: complex, limiet: float) -> Tuple[complex, bool]:
    if np.isinf(abs(impedantie)):
        return 0.0 + 0.0j, False
    if abs(impedantie) <= EPS:
        if abs(spanning) <= EPS:
            return 0.0 + 0.0j, False
        return limiet * spanning / abs(spanning), True
    stroom = spanning / impedantie
    grootte = abs(stroom)
    if not np.isfinite(grootte):
        return limiet * spanning / max(abs(spanning), EPS), True
    if grootte > limiet:
        return limiet * stroom / grootte, True
    return stroom, False


def eenheidsvector(z: complex, standaard: complex = 1.0 + 0.0j) -> complex:
    if cabs(z) <= EPS:
        return standaard
    return z / abs(z)


def belastingstype_phi2(phi2_deg: float) -> str:
    if phi2_deg < -0.5:
        return "inductief"
    if phi2_deg > 0.5:
        return "capacitief"
    return "ohms"


def cosinus_tussen(spanning: complex, stroom: complex) -> float:
    noemer = cabs(spanning) * cabs(stroom)
    if noemer <= EPS or not np.isfinite(noemer):
        return float("nan")
    waarde = float(np.real(spanning * np.conj(stroom)) / noemer)
    if not np.isfinite(waarde):
        return float("nan")
    return float(np.clip(waarde, -1.0, 1.0))


@dataclass
class Invoer:
    N1: int
    N2: int
    U1n: float
    I1n: float
    U1: float
    P_fe_n: float
    cos_phi0n: float
    P_cu_n: float
    u_k_pct: float
    phi2_deg: float
    Z_load_abs: float
    modus: str
    diagrammodus: str


@dataclass
class Resultaat:
    invoer: Invoer
    waarschuwingen: List[str]
    k: float
    U2n: float
    I2n: float
    S_n: float
    Z_n2: float
    R_v: float
    X_mu: float
    I0n: float
    I_v_n: float
    I_mu_n: float
    cos_phi0n: float
    R_k1: float
    X_k1: float
    Z_k1: float
    u_k_berekend: float
    U_k: float
    R1: float
    X1: float
    R2: float
    X2: float
    ZL: complex
    U1: complex
    minus_E1: complex
    I1: complex
    I0: complex
    I_v: complex
    I_mu: complex
    I1_prime: complex
    E2: complex
    U2: complex
    I2: complex
    P1: float
    Q1: float
    S1: float
    P2: float
    Q2: float
    S2: float
    P_cu: float
    P_fe: float
    eta: float
    d_u_pct: float
    belastingsgraad: float
    cos_phi2: float
    belastingstype: str
    status: str
    vermogensfout: float


def bereken_model(inp: Invoer) -> Resultaat:
    waarschuwingen: List[str] = []
    k = inp.N1 / max(inp.N2, 1)
    U1n = max(float(inp.U1n), EPS)
    I1n = max(float(inp.I1n), EPS)
    U2n = U1n / k
    I2n = k * I1n
    S_n = U1n * I1n
    Z_n2 = U2n / max(I2n, EPS)

    P_fe_n = max(inp.P_fe_n, 0.0)
    cos_phi0n = float(np.clip(inp.cos_phi0n, 0.20, 0.25))
    if P_fe_n <= EPS:
        I_v_n = 0.0; I0n = 0.0; I_mu_n = 0.0
        R_v = float("inf"); X_mu = float("inf")
    else:
        I_v_n = P_fe_n / U1n
        I0n = I_v_n / max(cos_phi0n, EPS)
        I_mu_n = math.sqrt(max(I0n**2 - I_v_n**2, 0.0))
        R_v = U1n**2 / P_fe_n
        X_mu = U1n / I_mu_n if I_mu_n > EPS else float("inf")
        if I0n > 2.0 * I1n:
            waarschuwingen.append("De ingestelde PFe,n veroorzaakt een zeer grote nullaststroom.")

    Z_Rv = complex(R_v, 0.0) if np.isfinite(R_v) else complex(np.inf)
    Z_Xmu = complex(0.0, X_mu) if np.isfinite(X_mu) else complex(np.inf)
    Z_m = parallel(Z_Rv, Z_Xmu)

    Z_k_gewenst = max(inp.u_k_pct, 0.0) / 100.0 * U1n / I1n
    R_k1 = max(inp.P_cu_n, 0.0) / I1n**2
    if R_k1 > Z_k_gewenst + 1e-10:
        Z_k1 = R_k1
        X_k1 = 0.0
        waarschuwingen.append("PCu,n is te groot voor de ingestelde uk; de effectieve uk werd verhoogd.")
    else:
        Z_k1 = Z_k_gewenst
        X_k1 = math.sqrt(max(Z_k1**2 - R_k1**2, 0.0))

    u_k_berekend = 100.0 * Z_k1 * I1n / U1n
    U_k = Z_k1 * I1n
    R1 = 0.5 * R_k1
    X1 = 0.5 * X_k1
    R2 = 0.5 * R_k1 / k**2
    X2 = 0.5 * X_k1 / k**2
    Z1 = complex(R1, X1)
    Z2 = complex(R2, X2)
    Z2_primair = k**2 * Z2

    modus = inp.modus.lower()
    phi2_deg = float(np.clip(inp.phi2_deg, -90.0, 90.0))
    cos_phi2 = math.cos(math.radians(phi2_deg))
    belastingstype = belastingstype_phi2(phi2_deg)

    if modus == "nullastproef":
        ZL = complex(np.inf)
        Z_tak_primair = complex(np.inf)
        belastingstype = "open kring"
    elif modus == "kortsluitproef":
        ZL = 0.0 + 0.0j
        Z_tak_primair = Z2_primair
        belastingstype = "kortsluiting"
    else:
        z_abs = max(inp.Z_load_abs, 0.0)
        phi_z = -math.radians(phi2_deg)
        ZL = z_abs * complex(math.cos(phi_z), math.sin(phi_z))
        Z_tak_primair = Z2_primair + k**2 * ZL

    Z_parallel = Z_m if modus == "nullastproef" else parallel(Z_m, Z_tak_primair)
    Z_in = Z1 + Z_parallel
    U1 = complex(max(inp.U1, 0.0), 0.0)

    stroomlimiet_primair = MAX_STROOM_FACTOR * I1n
    I1, begrensd_1 = veilige_stroom(U1, Z_in, stroomlimiet_primair)
    if begrensd_1:
        waarschuwingen.append("De theoretisch onbegrensde stroom werd alleen voor de weergave begrensd.")

    minus_E1 = U1 - I1 * Z1
    I_v = 0.0 + 0.0j if not np.isfinite(R_v) else minus_E1 / R_v
    I_mu = 0.0 + 0.0j if not np.isfinite(X_mu) else minus_E1 / complex(0.0, X_mu)
    I0 = I_v + I_mu

    if modus == "nullastproef":
        I1_prime = 0.0 + 0.0j
        I2 = 0.0 + 0.0j
        E2 = -minus_E1 / k
        U2 = E2
    else:
        I1_prime, begrensd_tak = veilige_stroom(minus_E1, Z_tak_primair, stroomlimiet_primair)
        if begrensd_tak and not begrensd_1:
            waarschuwingen.append("De belastingsstroom werd alleen voor de weergave begrensd.")
        I2 = -k * I1_prime
        E2 = -minus_E1 / k
        U2 = E2 - I2 * Z2
        if modus == "kortsluitproef":
            U2 = 0.0 + 0.0j

    S1_complex = U1 * np.conj(I1)
    S2_complex = U2 * np.conj(I2)
    P1 = float(np.real(S1_complex)); Q1 = float(np.imag(S1_complex)); S1 = cabs(U1) * cabs(I1)
    P2 = float(np.real(S2_complex)); Q2 = float(np.imag(S2_complex)); S2 = cabs(U2) * cabs(I2)
    P_cu = R1 * cabs(I1) ** 2 + R2 * cabs(I2) ** 2
    P_fe = 0.0 if (not np.isfinite(R_v) or R_v <= EPS) else cabs(minus_E1) ** 2 / R_v
    verliesnoemer = P2 + P_cu + P_fe
    eta = 100.0 * P2 / verliesnoemer if P2 > EPS and verliesnoemer > EPS else 0.0
    eta = float(np.clip(eta, 0.0, 100.0)) if np.isfinite(eta) else 0.0
    d_u_pct = 100.0 * (U2n - cabs(U2)) / U2n if U2n > EPS else 0.0
    belastingsgraad = cabs(I2) / I2n if I2n > EPS else 0.0
    vermogensfout = P1 - (P2 + P_cu + P_fe)

    if modus == "nullastproef":
        status = "nullastproef"
    elif modus == "kortsluitproef":
        status = "kortsluitproef"
    elif cabs(I1) > 1.05 * I1n or cabs(I2) > 1.05 * I2n:
        status = "overbelasting"
    else:
        status = "normale belasting"

    return Resultaat(inp, waarschuwingen, k, U2n, I2n, S_n, Z_n2, R_v, X_mu, I0n, I_v_n, I_mu_n,
                     cos_phi0n, R_k1, X_k1, Z_k1, u_k_berekend, U_k, R1, X1, R2, X2,
                     ZL, U1, minus_E1, I1, I0, I_v, I_mu, I1_prime, E2, U2, I2,
                     P1, Q1, S1, P2, Q2, S2, P_cu, P_fe, eta, d_u_pct, belastingsgraad,
                     cos_phi2, belastingstype, status, vermogensfout)


def maak_diagramfasoren(res: Resultaat):
    referentie = res.U2 if cabs(res.U2) > EPS else (res.E2 if cabs(res.E2) > EPS else -1.0j)
    rotatiehoek = -math.pi / 2.0 - np.angle(referentie)
    rot = complex(math.cos(rotatiehoek), math.sin(rotatiehoek))
    return {"U1": res.U1 * rot, "minus_E1": res.minus_E1 * rot, "U2": res.U2 * rot, "E2": res.E2 * rot,
            "I1": res.I1 * rot, "I0": res.I0 * rot, "I_v": res.I_v * rot, "I_mu": res.I_mu * rot,
            "I1_prime": res.I1_prime * rot, "I2": res.I2 * rot}


def teken_vector(ax, start, delta, label, color, lw=0.95, linestyle="-", offset=(0.08, 0.08), visible=True, alpha=1.0):
    if not visible or cabs(delta) <= 1e-9 or not np.isfinite(cabs(delta)):
        return
    end = start + delta
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=linestyle, alpha=alpha, mutation_scale=8), zorder=5)
    ax.text(end.real + offset[0], end.imag + offset[1], label, color=color, fontsize=7.4, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.64, pad=0.28), zorder=6)


def teken_fasehoek(ax, radius, a_spanning, a_stroom, label):
    delta = normaliseer_hoek(a_stroom - a_spanning)
    if not np.isfinite(delta) or abs(delta) < 0.5:
        return
    if delta >= 0:
        theta1 = a_spanning; theta2 = a_spanning + delta; mid_deg = a_spanning + 0.5 * delta
    else:
        theta1 = a_stroom; theta2 = a_spanning; mid_deg = a_stroom - 0.5 * delta
    ax.add_patch(patches.Arc((0, 0), 2 * radius, 2 * radius, theta1=theta1, theta2=theta2,
                             color="#777777", lw=0.70, ls="--"))
    a_end = math.radians(a_stroom)
    tangent = a_end + (math.pi / 2.0 if delta > 0 else -math.pi / 2.0)
    end = np.array([radius * math.cos(a_end), radius * math.sin(a_end)])
    seg_len = 0.14 * radius
    start = end - seg_len * np.array([math.cos(tangent), math.sin(tangent)])
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=0.55, color="#777777", mutation_scale=7), zorder=7)
    mid = math.radians(mid_deg)
    ax.text(1.18 * radius * math.cos(mid), 1.18 * radius * math.sin(mid), label, color="#777777", fontsize=7.2)


def bereken_flux_lengtes(res: Resultaat):
    Rm = MAGNETISCHE_LENGTE_STANDAARD / (MU0 * MU_R_STANDAARD * KERN_OPPERVLAK_STANDAARD)
    phi1_abs = res.invoer.N1 * cabs(res.I1_prime) / Rm
    phi2_abs = res.invoer.N2 * cabs(res.I2) / Rm
    return phi1_abs, phi2_abs


def plot_vectordiagram(res: Resultaat):
    d = maak_diagramfasoren(res)
    dR1 = res.R1 * d["I1"]; dX1 = complex(0.0, res.X1) * d["I1"]
    dR2 = res.R2 * d["I2"]; dX2 = complex(0.0, res.X2) * d["I2"]
    hoofd_max = max(cabs(d["minus_E1"]), cabs(d["U1"]), cabs(d["U2"]), cabs(d["E2"]), 1.0)
    val_max = max(cabs(dR1), cabs(dX1), cabs(dR2), cabs(dX2), EPS)
    vergroting = float(np.clip(0.19 * hoofd_max / val_max, 1.0, 8.0)) if res.invoer.diagrammodus == "didactisch" else 1.0
    nullastmodus = res.invoer.modus == "nullastproef"
    U1_constructie = d["U1"] if nullastmodus else d["minus_E1"] + vergroting * dR1 + vergroting * dX1
    E2_constructie = d["U2"] + vergroting * dR2 + vergroting * dX2
    spanning_max = max(cabs(d["minus_E1"]), cabs(d["U1"]), cabs(U1_constructie), cabs(d["U2"]), cabs(E2_constructie), 1.0)
    schaal_U = 7.0 / spanning_max
    minusE1p = d["minus_E1"] * schaal_U; U1p = U1_constructie * schaal_U
    U2p = d["U2"] * schaal_U; E2p = E2_constructie * schaal_U
    dR1p = vergroting * dR1 * schaal_U; dX1p = vergroting * dX1 * schaal_U
    dR2p = vergroting * dR2 * schaal_U; dX2p = vergroting * dX2 * schaal_U
    stroom_max = max(cabs(d["I1"]), cabs(d["I0"]), cabs(d["I_v"]), cabs(d["I_mu"]), cabs(d["I1_prime"]), cabs(d["I2"]), 1e-9)
    schaal_I = min(4.2 / stroom_max, 12.0)
    I1p = d["I1"] * schaal_I; I0p = d["I0"] * schaal_I; Ivp = d["I_v"] * schaal_I
    Imup = d["I_mu"] * schaal_I; I1primep = d["I1_prime"] * schaal_I; I2p = d["I2"] * schaal_I

    phi1_abs, phi2_abs = bereken_flux_lengtes(res)
    phi_ref = max(phi1_abs, phi2_abs, 1e-12)
    phi1_len = min(3.0, 1.65 * phi1_abs / phi_ref) if phi1_abs > EPS else 0.0
    phi2_len = min(3.0, 1.65 * phi2_abs / phi_ref) if phi2_abs > EPS else 0.0

    zero = 0.0 + 0.0j
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)
    ax.set_aspect("equal", adjustable="box"); ax.set_xlim(-7.2, 7.2); ax.set_ylim(-7.45, 7.45); ax.axis("off")
    ax.axhline(0, color="#bbbbbb", lw=0.7, ls=":", zorder=0); ax.axvline(0, color="#bbbbbb", lw=0.7, ls=":", zorder=0)

    teken_vector(ax, zero, minusE1p, r"$-\overline{E}_1$", "#2c7be5", lw=1.05, offset=(-0.66, 0.18))
    teken_vector(ax, minusE1p, dR1p, r"$R_1\overline{I}_1$", "#d64532", lw=0.85, visible=not nullastmodus)
    teken_vector(ax, minusE1p + dR1p, dX1p, r"$jX_1\overline{I}_1$", "#ff7f00", lw=0.85, visible=not nullastmodus)
    teken_vector(ax, zero, U1p, r"$\overline{U}_1$", "#005eb8", lw=1.10, offset=(-0.66, 0.22))
    teken_vector(ax, zero, U2p, r"$\overline{U}_2$", "#e0ad21", lw=1.05, offset=(-0.66, -0.22), visible=cabs(U2p) > 1e-8)
    teken_vector(ax, U2p, dR2p, r"$R_2\overline{I}_2$", "#d64532", lw=0.85, offset=(-0.62, -0.20), visible=cabs(dR2p) > 1e-8)
    teken_vector(ax, U2p + dR2p, dX2p, r"$jX_2\overline{I}_2$", "#ff7f00", lw=0.85, offset=(0.12, -0.30), visible=cabs(dX2p) > 1e-8)
    teken_vector(ax, zero, E2p, r"$\overline{E}_2$", "#9a7d00", lw=1.00, offset=(0.15, -0.26), visible=cabs(E2p) > 1e-8)
    teken_vector(ax, zero, I1p, r"$\overline{I}_1$", "#ef7f1a", lw=1.00, offset=(0.12, 0.18))
    teken_vector(ax, zero, I0p, r"$\overline{I}_0$", "#8e44ad", lw=0.95, offset=(0.12, 0.18))
    teken_vector(ax, zero, Ivp, r"$\overline{I}_v$", "#e74c3c", lw=0.75, linestyle="--", offset=(-0.54, 0.18), alpha=0.95)
    teken_vector(ax, zero, Imup, r"$\overline{I}_{\mu}$", "#d2527f", lw=0.75, linestyle="--", offset=(0.17, -0.44), alpha=0.95)
    teken_vector(ax, zero, I1primep, r"$\overline{I}'_1$", "#7d3c98", lw=0.85, linestyle="--", offset=(-0.58, 0.18), visible=cabs(I1primep) > 1e-8, alpha=0.95)
    teken_vector(ax, zero, I2p, r"$\overline{I}_2$", "#e74c3c", lw=0.95, offset=(-0.56, -0.44), visible=cabs(I2p) > 1e-8)
    teken_vector(ax, zero, phi2_len * eenheidsvector(d["I2"], -1.0j), r"$\overline{\Phi}_2$", "#13a85a", lw=0.85, offset=(-0.72, -0.26), visible=cabs(d["I2"]) > 1e-8 and cabs(d["I1_prime"]) > 1e-8)
    teken_vector(ax, zero, phi1_len * eenheidsvector(d["I1_prime"], 1.0j), r"$\overline{\Phi}_1$", "#13a85a", lw=0.85, offset=(0.16, 0.28), visible=cabs(d["I2"]) > 1e-8 and cabs(d["I1_prime"]) > 1e-8)
    phi_richting = eenheidsvector(d["I_mu"], eenheidsvector(-1.0j * d["minus_E1"], 1.0 + 0.0j))
    teken_vector(ax, zero, 3.10 * phi_richting, r"$\overline{\Phi}$", "#16b85f", lw=1.05, offset=(0.17, -0.44))
    teken_fasehoek(ax, 1.65, hoek_graden(U1p), hoek_graden(I1p), r"$\varphi_1$")
    if cabs(I2p) > EPS and cabs(U2p) > EPS:
        teken_fasehoek(ax, 1.18, hoek_graden(U2p), hoek_graden(I2p), r"$\varphi_2$")
    ax.add_patch(patches.FancyArrowPatch((5.35, -0.80), (5.35, 0.80), connectionstyle="arc3,rad=0.48",
                                         arrowstyle="-|>", mutation_scale=8, color="black", linewidth=0.85, zorder=2))
    ax.text(5.90, 0.08, r"$\omega$", fontsize=8.2)
    if nullastmodus:
        ax.text(-7.55, 8.10, "Nullast: R1I1 en jX1I1 bestaan, maar zijn te klein om zinvol te tekenen.",
                fontsize=7.8, color="#8a5a00", ha="left", va="top",
                bbox=dict(facecolor="#fff3c4", edgecolor="#d9a300", linewidth=0.6, alpha=0.90, pad=2.0))
    ax.text(-6.95, -7.20, f"schaal: U × {schaal_U:.3f}  |  I × {schaal_I:.3f}  |  spanningsvallen × {vergroting:.2f}", fontsize=7.2)
    return fig


def plot_rendement(res: Resultaat):
    inp = res.invoer; z_n = max(res.Z_n2, EPS)
    z_reeks = np.concatenate((np.geomspace(100.0 * z_n, 0.02 * z_n, CURVE_PUNTEN), np.array([0.0])))
    stromen, rendementen = [], []
    for z in z_reeks:
        rr = bereken_model(replace(inp, Z_load_abs=float(z), modus="belasting", diagrammodus="werkelijk"))
        if np.isfinite(cabs(rr.I2)) and np.isfinite(rr.eta):
            stromen.append(cabs(rr.I2)); rendementen.append(rr.eta)
    stromen.append(0.0); rendementen.append(0.0)
    i_arr = np.asarray(stromen); e_arr = np.asarray(rendementen)
    volgorde = np.argsort(i_arr); i_arr = i_arr[volgorde]; e_arr = e_arr[volgorde]
    uniek = np.concatenate(([True], np.diff(i_arr) > max(1e-9, 1e-7 * res.I2n)))
    i_arr = i_arr[uniek]; e_arr = e_arr[uniek]
    fig, ax = plt.subplots(figsize=(1.85, 1.05), dpi=180)
    ax.plot(i_arr, e_arr, linewidth=0.85)
    ax.plot([cabs(res.I2)], [res.eta], marker="o", markersize=3.2)
    ax.axvline(cabs(res.I2), linewidth=0.50, linestyle="--", alpha=0.55)
    ax.axhline(res.eta, linewidth=0.50, linestyle="--", alpha=0.55)
    ax.set_title(r"$\eta$ in functie van $|I_2|$", fontsize=7); ax.set_xlabel(r"$|I_2|$ (A)", fontsize=6); ax.set_ylabel(r"$\eta$ (%)", fontsize=6)
    ax.tick_params(axis="both", labelsize=5.5); ax.grid(True, linewidth=0.35, alpha=0.45); ax.set_ylim(0.0, 100.5)
    gewenst_max = max(1.5 * res.I2n, 1.08 * cabs(res.I2), 1e-6)
    if res.invoer.modus != "kortsluitproef":
        gewenst_max = min(max(1.5 * res.I2n, 1.08 * cabs(res.I2)), max(2.2 * res.I2n, 1e-6))
    ax.set_xlim(0.0, gewenst_max)
    ax.text(0.98, 0.06, f"|I₂| = {format_stroom(cabs(res.I2))}\nη = {res.eta:.2f} %",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
    return fig


st.set_page_config(page_title="Simulator niet-ideale eenfasige transformator", layout="wide")
st.markdown(
    """
    <style>
    section.main div.block-container { max-width: 1680px; padding-top: 1.6rem; }
    div[data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
    }
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    @media (max-width: 1100px) {
        div[data-testid="stImage"] img {
            max-width: 100%;
        }
        section.main div.block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Simulator niet-ideale eenfasige transformator")
st.caption("Didactische simulator voor het equivalent schema, het vectordiagram en het rendement.")

if "U1n_waarde" not in st.session_state:
    st.session_state.U1n_waarde = 230.0
if "U1n_slider" not in st.session_state:
    st.session_state.U1n_slider = 230.0
if "I1n_waarde" not in st.session_state:
    st.session_state.I1n_waarde = 1.5
if "I1n_slider" not in st.session_state:
    st.session_state.I1n_slider = 1.5

def sync_u_slider():
    st.session_state.U1n_waarde = st.session_state.U1n_slider
def sync_u_input():
    st.session_state.U1n_slider = float(np.clip(st.session_state.U1n_waarde, 10.0, 5000.0))
def sync_i_slider():
    st.session_state.I1n_waarde = st.session_state.I1n_slider
def sync_i_input():
    st.session_state.I1n_slider = float(np.clip(st.session_state.I1n_waarde, 0.05, 100.0))

with st.sidebar:
    st.header("Instellingen")
    st.subheader("Wikkelingen")
    c1, c2 = st.columns(2)
    with c1:
        N1 = st.number_input("N₁", min_value=1, max_value=1_000_000, value=1000, step=1)
    with c2:
        N2 = st.number_input("N₂", min_value=1, max_value=1_000_000, value=1000, step=1)

    st.subheader("Nominale gegevens")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.slider("U₁,n slider (V)", 10.0, 5000.0, key="U1n_slider", step=1.0, on_change=sync_u_slider)
    with c2:
        st.number_input("U₁,n (V)", min_value=10.0, max_value=5000.0, key="U1n_waarde", step=1.0, on_change=sync_u_input)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.slider("I₁,n slider (A)", 0.05, 100.0, key="I1n_slider", step=0.05, on_change=sync_i_slider)
    with c2:
        st.number_input("I₁,n (A)", min_value=0.05, max_value=100.0, key="I1n_waarde", step=0.05, on_change=sync_i_input)

    U1n = float(st.session_state.U1n_waarde); I1n = float(st.session_state.I1n_waarde)
    k_temp = N1 / max(N2, 1); U2n_temp = U1n / k_temp; I2n_temp = k_temp * I1n
    Zn_temp = U2n_temp / max(I2n_temp, EPS)

    st.subheader("Proefmodus")
    modus = st.radio("Modus", ["belasting", "nullastproef", "kortsluitproef"], index=0)
    st.subheader("Belasting")
    phi2_deg = st.slider("φ₂ = ∠I₂ − ∠U₂ (°)", -90, 90, 0, step=1)
    Z_load_abs = st.slider("|ZL| (Ω)", 0.0, max(2.0 * Zn_temp, 1.0), min(Zn_temp, max(2.0 * Zn_temp, 1.0)), step=max(Zn_temp / 300.0, 0.01))

    st.subheader("Primaire spanning")
    U1 = st.number_input("U₁ (V)", min_value=0.0, max_value=max(1.5 * U1n, 1.0), value=float(U1n), step=1.0, disabled=(modus != "belasting"))

    st.subheader("Nullastproef")
    P_fe_n = st.slider("PFe,n (W)", 0.0, max(0.1 * U1n * I1n, 1.0), 0.05 * U1n * I1n, step=max(0.001 * U1n * I1n, 0.01))
    cos_phi0n = st.slider("cos φ₀,n", 0.20, 0.25, 0.225, step=0.001)
    st.subheader("Kortsluitproef")
    P_cu_n = st.slider("PCu,n (W)", 0.0, max(0.1 * U1n * I1n, 1.0), 0.05 * U1n * I1n, step=max(0.001 * U1n * I1n, 0.01))
    u_k_pct = st.slider("uk (%)", 0.0, 12.0, 6.0, step=0.1)

    if modus == "kortsluitproef":
        R_k_pct = 100.0 * max(P_cu_n, 0.0) / max(U1n * I1n, EPS)
        U1 = U1n * max(u_k_pct, R_k_pct, 0.0) / 100.0
    elif modus == "nullastproef":
        U1 = U1n

    st.subheader("Vectordiagram")
    diagrammodus = st.radio("Diagrammodus", ["werkelijk", "didactisch"], index=1)

inp = Invoer(int(N1), int(N2), float(U1n), float(I1n), float(U1), float(P_fe_n), float(cos_phi0n),
             float(P_cu_n), float(u_k_pct), float(phi2_deg), float(Z_load_abs), modus, diagrammodus)
res = bereken_model(inp)
d = maak_diagramfasoren(res)

for w in res.waarschuwingen:
    st.warning(w)

top1, top2, top3, top4 = st.columns(4)
top1.metric("U₂", format_getal(cabs(res.U2), "V", 2))
top2.metric("I₂", format_stroom(cabs(res.I2)))
top3.metric("η", f"{res.eta:.2f} %")
top4.metric("status", res.status)


# =============================================================================
# Weergave
# =============================================================================

st.markdown("### Vectordiagram en rendement")

# Bovenaan: duidelijk vectordiagram met kleine rendementsgrafiek ernaast.
col_vec, col_right = st.columns([3.2, 0.85], gap="medium")

with col_vec:
    st.pyplot(plot_vectordiagram(res), clear_figure=True, use_container_width=True)

with col_right:
    st.markdown("##### Rendement")
    st.pyplot(plot_rendement(res), clear_figure=True, use_container_width=False)

    st.markdown("##### Hoeken")
    phi1_deg = fasehoek_stroom_tov_spanning(res.U1, res.I1)
    phi2_actueel = fasehoek_stroom_tov_spanning(res.U2, res.I2) if res.invoer.modus == "belasting" else float("nan")

    hoek_col1, hoek_col2 = st.columns(2)
    hoek_col1.metric("φ₁", format_hoek_signed(phi1_deg, 1))
    hoek_col2.metric("φ₂", format_hoek_signed(phi2_actueel, 1) if res.invoer.modus == "belasting" else "—")
    st.caption("Negatief = inductief. Positief = capacitief.")

st.markdown("---")
st.markdown("### Tabellen")

cos_phi1 = cosinus_tussen(res.U1, res.I1)
cos_phi0_actueel = cosinus_tussen(res.U1, res.I0)
phi1_deg = fasehoek_stroom_tov_spanning(res.U1, res.I1)
phi2_actueel = fasehoek_stroom_tov_spanning(res.U2, res.I2) if res.invoer.modus == "belasting" else float("nan")

tabellen = {
    "Nominale gegevens": pd.DataFrame({
        "Grootheid": ["N₁/N₂", "k", "U₁,n", "U₂,n", "I₁,n", "I₂,n", "Sₙ"],
        "Waarde": [
            f"{res.invoer.N1}/{res.invoer.N2}",
            format_getal(res.k, "", 3),
            format_getal(res.invoer.U1n, "V", 1),
            format_getal(res.U2n, "V", 1),
            format_stroom(res.invoer.I1n),
            format_stroom(res.I2n),
            format_va(res.S_n),
        ],
    }),
    "Primair": pd.DataFrame({
        "Grootheid": ["U₁", "-E₁", "I₁", "cos φ₁", "φ₁", "I′₁", "P₁", "S₁"],
        "Waarde": [
            polar_tekst(d["U1"], "V"),
            polar_tekst(d["minus_E1"], "V"),
            polar_tekst(d["I1"], "A"),
            format_cosphi(cos_phi1),
            format_hoek_signed(phi1_deg, 1),
            polar_tekst(d["I1_prime"], "A"),
            format_watt(res.P1),
            format_va(res.S1),
        ],
    }),
    "Secundair": pd.DataFrame({
        "Grootheid": ["E₂", "U₂", "I₂", "|ZL|", "cos φ₂", "φ₂", "karakter", "P₂", "S₂"],
        "Waarde": [
            polar_tekst(d["E2"], "V"),
            polar_tekst(d["U2"], "V"),
            polar_tekst(d["I2"], "A"),
            format_ohm(abs(res.ZL)),
            format_getal(res.cos_phi2, "", 3) if res.invoer.modus == "belasting" else "—",
            format_hoek_signed(phi2_actueel, 1) if res.invoer.modus == "belasting" else "—",
            res.belastingstype,
            format_watt(res.P2),
            format_va(res.S2),
        ],
    }),
    "Proefgegevens en equivalent schema": pd.DataFrame({
        "Grootheid": ["PFe,n", "cos φ₀", "I₀", "Iv", "Iμ", "PCu,n", "uk,calc", "Uk", "R₁", "X₁", "R₂", "X₂", "Rv", "Xμ"],
        "Waarde": [
            format_watt(res.invoer.P_fe_n),
            format_cosphi(cos_phi0_actueel),
            polar_tekst(d["I0"], "A"),
            polar_tekst(d["I_v"], "A"),
            polar_tekst(d["I_mu"], "A"),
            format_watt(res.invoer.P_cu_n),
            format_getal(res.u_k_berekend, "%", 2),
            format_getal(res.U_k, "V", 2),
            format_ohm(res.R1),
            format_ohm(res.X1),
            format_ohm(res.R2),
            format_ohm(res.X2),
            format_ohm(res.R_v),
            format_ohm(res.X_mu),
        ],
    }),
    "Verliezen en werking": pd.DataFrame({
        "Grootheid": ["du", "I₂/I₂,n", "PCu", "PFe", "PCu/PFe", "η", "Pgem", "status"],
        "Waarde": [
            format_getal(res.d_u_pct, "%", 2),
            format_getal(res.belastingsgraad, "", 3),
            format_watt(res.P_cu),
            format_watt(res.P_fe),
            format_getal(res.P_cu / res.P_fe, "", 3) if res.P_fe > EPS else "∞",
            format_getal(res.eta, "%", 2),
            format_watt(res.vermogensfout),
            res.status,
        ],
    }),
}

t1, t2 = st.columns(2, gap="large")

with t1:
    st.subheader("Nominale gegevens")
    st.dataframe(tabellen["Nominale gegevens"], hide_index=True, use_container_width=True)

    st.subheader("Primair")
    st.dataframe(tabellen["Primair"], hide_index=True, use_container_width=True)

with t2:
    st.subheader("Secundair")
    st.dataframe(tabellen["Secundair"], hide_index=True, use_container_width=True)

    st.subheader("Proefgegevens en equivalent schema")
    st.dataframe(tabellen["Proefgegevens en equivalent schema"], hide_index=True, use_container_width=True)

st.subheader("Verliezen en werking")
st.dataframe(tabellen["Verliezen en werking"], hide_index=True, use_container_width=True)
