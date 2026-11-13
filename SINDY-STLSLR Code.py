#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CSV_FILE = "W20P.csv"

TRAIN_N_RAW = 44

NUM_POINTS =90

DEGREE_MIN = 1
DEGREE_MAX = 9

ALPHA_SET = np.logspace(-5, 1, 50)                   
DELTA_SET = np.array([10.0 ** (-k) for k in range(0, 9)])  

MAX_STLSLR_ITER = 50

STATE_LIMIT = 1e6
DERIVATIVE_LIMIT = 1e6

NOISE_SEED = 123
NOISE_SCALE = 1.0

PLOT_HYPERPARAMETER_SURFACE = True
PLOT_ACTIVE_TERMS_SURFACE = True  
PLOT_VALIDATION = True
PLOT_SYNTHETIC_VALIDATION = True
PLOT_HEATMAPS = True

PNG_DPI = 600

SCI_DECIMALS = 2
ZERO_COLOR = "#5a5a5a"
CMAP_NAME = "coolwarm"
TRUNC_LO = 0.10
TRUNC_HI = 0.90

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": PNG_DPI,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

CSV_BASE = os.path.splitext(os.path.basename(CSV_FILE))[0]

STATE_NAMES = ["CODs", "NH3", "NO3", "PO4", "DO", "ORP", "pH"]

CSV_COLUMNS = {
    "CODs": "CODs",
    "NH3": "Ammonia",
    "NO3": "Nitrates",
    "PO4": "Phosphates",
    "DO": "DO",
    "ORP": "REDOX",
    "pH": "pH",
}

THETA_TERMS = [
    "CODs", "NH3", "NO3", "PO4", "DO", "ORP", "pH",
    "CODs^2",
    "CODs*NH3", "CODs*NO3", "CODs*PO4", "CODs*DO", "CODs*ORP", "CODs*pH",
    "NH3^2",
    "NH3*NO3", "NH3*PO4", "NH3*DO", "NH3*ORP", "NH3*pH",
    "NO3^2",
    "NO3*PO4", "NO3*DO", "NO3*ORP", "NO3*pH",
    "PO4^2",
    "PO4*DO", "PO4*ORP", "PO4*pH",
    "DO^2",
    "DO*ORP", "DO*pH",
    "ORP^2",
    "ORP*pH",
    "pH^2",
]

RAW_TERMS = ["1"] + THETA_TERMS

PLOT_LABELS = {
    "CODs": r"$\mathrm{COD}_s$",
    "NH3": r"$\mathrm{NH}_3$",
    "NO3": r"$\mathrm{NO}_3$",
    "PO4": r"$\mathrm{PO}_4$",
    "DO": r"$\mathrm{DO}$",
    "ORP": r"$\mathrm{ORP}$",
    "pH": r"$\mathrm{pH}$",
}

def theta(z):
    z = np.asarray(z, dtype=float).reshape(7)
    z = np.clip(z, -STATE_LIMIT, STATE_LIMIT)

    CODs, NH3, NO3, PO4, DO, ORP, pH = z

    return np.array([
        CODs, NH3, NO3, PO4, DO, ORP, pH,
        CODs * CODs,
        CODs * NH3,
        CODs * NO3,
        CODs * PO4,
        CODs * DO,
        CODs * ORP,
        CODs * pH,
        NH3 * NH3,
        NH3 * NO3,
        NH3 * PO4,
        NH3 * DO,
        NH3 * ORP,
        NH3 * pH,
        NO3 * NO3,
        NO3 * PO4,
        NO3 * DO,
        NO3 * ORP,
        NO3 * pH,
        PO4 * PO4,
        PO4 * DO,
        PO4 * ORP,
        PO4 * pH,
        DO * DO,
        DO * ORP,
        DO * pH,
        ORP * ORP,
        ORP * pH,
        pH * pH,
    ], dtype=float)

def library_matrix(Z):
    Z = np.asarray(Z, dtype=float)
    return np.vstack([theta(z_row) for z_row in Z])

def minmax_fit(values):
    values = np.asarray(values, dtype=float)
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    scale = xmax - xmin
    if np.isclose(scale, 0.0):
        scale = 1.0
    return xmin, xmax, float(scale)

def minmax_transform(values, xmin, scale):
    return (np.asarray(values, dtype=float) - xmin) / scale

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def l2_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.linalg.norm(y_true - y_pred))

def frobenius_error(Z, Z_tilde):
    if Z_tilde is None:
        return np.inf
    return float(np.linalg.norm(np.asarray(Z) - np.asarray(Z_tilde), ord="fro"))

def mean_frobenius_error(Z, Z_tilde):
    if Z_tilde is None:
        return np.inf
    return frobenius_error(Z, Z_tilde) / np.asarray(Z).size

def interpolate_at_times(t_grid, values_grid, t_query):
    return np.interp(
        np.asarray(t_query, dtype=float),
        np.asarray(t_grid, dtype=float),
        np.asarray(values_grid, dtype=float),
    )

def polynomial_derivative(model, poly, t):
    t = np.asarray(t, dtype=float).reshape(-1)
    powers = np.asarray(poly.powers_[:, 0], dtype=int)
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)

    derivative = np.zeros_like(t, dtype=float)
    for coefficient, power in zip(coefficients, powers):
        if power > 0:
            derivative += coefficient * power * t ** (power - 1)

    return derivative

def fit_polynomial_trajectory(t_raw, z_raw, degree_min, degree_max):
    t_raw = np.asarray(t_raw, dtype=float).reshape(-1, 1)
    z_raw = np.asarray(z_raw, dtype=float)

    best = None

    for degree in range(degree_min, degree_max + 1):
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        T_poly = poly.fit_transform(t_raw)
        model = LinearRegression().fit(T_poly, z_raw)
        z_hat = model.predict(T_poly)
        error = np.sqrt(mean_squared_error(z_raw, z_hat))

        if best is None or error < best[0]:
            best = (float(error), degree, model, poly)

    return best  

def solve_lasso(Theta_active, zdot_j, alpha):
    Theta_active = np.asarray(Theta_active, dtype=float)
    zdot_j = np.asarray(zdot_j, dtype=float).reshape(-1)

    M, n_terms = Theta_active.shape
    xi = cp.Variable(n_terms)

    objective = cp.Minimize(
        (1.0 / M) * cp.sum_squares(Theta_active @ xi - zdot_j)
        + alpha * cp.norm1(xi)
    )
    problem = cp.Problem(objective)

    for solver in (cp.OSQP, cp.ECOS, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
            if xi.value is not None and np.all(np.isfinite(xi.value)):
                return np.asarray(xi.value, dtype=float).reshape(-1)
        except Exception:
            continue

    return None

def stlslr_equation(Theta_scaled, zdot_j, alpha, delta, column_norms, N):
    n_library_terms = Theta_scaled.shape[1]
    active = np.ones(n_library_terms, dtype=bool)

    xi_scaled_full = np.zeros(n_library_terms, dtype=float)
    converged = False
    iterations = 0

    for k in range(N):
        iterations = k + 1
        active_idx = np.flatnonzero(active)

        if active_idx.size == 0:
            converged = True
            break

        xi_active = solve_lasso(
            Theta_active=Theta_scaled[:, active_idx],
            zdot_j=zdot_j,
            alpha=alpha,
        )

        if xi_active is None:
            return None, {
                "iterations": iterations,
                "converged": False,
                "active_terms": 0,
                "solver_failed": True,
            }

        keep = np.abs(xi_active) >= delta

        xi_scaled_full[:] = 0.0
        xi_scaled_full[active_idx[keep]] = xi_active[keep]

        new_active = np.zeros_like(active)
        new_active[active_idx[keep]] = True

        if np.array_equal(new_active, active):
            converged = True
            active = new_active
            break

        active = new_active

    xi_j = xi_scaled_full / column_norms

    info = {
        "iterations": iterations,
        "converged": converged,
        "active_terms": int(np.count_nonzero(xi_j)),
        "solver_failed": False,
    }

    return xi_j, info

def stlslr_model(Theta_scaled, Zdot, alpha, delta, column_norms, N):
    Xi = np.zeros((Theta_scaled.shape[1], 7), dtype=float)
    equation_info = {}

    for j, state in enumerate(STATE_NAMES):
        xi_j, info_j = stlslr_equation(
            Theta_scaled=Theta_scaled,
            zdot_j=Zdot[:, j],
            alpha=alpha,
            delta=delta,
            column_norms=column_norms,
            N=N,
        )

        if xi_j is None:
            return None, equation_info

        Xi[:, j] = xi_j
        equation_info[state] = info_j

    return Xi, equation_info

def coupled_rhs(z, Xi):
    z = np.asarray(z, dtype=float).reshape(7)

    if np.any(~np.isfinite(z)) or np.max(np.abs(z)) > STATE_LIMIT:
        return None

    zdot = theta(z) @ Xi

    if np.any(~np.isfinite(zdot)):
        return None

    return np.clip(zdot, -DERIVATIVE_LIMIT, DERIVATIVE_LIMIT)

def rk4_step(z, dt, Xi):
    k1 = coupled_rhs(z, Xi)
    if k1 is None:
        return None

    k2 = coupled_rhs(z + 0.5 * dt * k1, Xi)
    if k2 is None:
        return None

    k3 = coupled_rhs(z + 0.5 * dt * k2, Xi)
    if k3 is None:
        return None

    k4 = coupled_rhs(z + dt * k3, Xi)
    if k4 is None:
        return None

    z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if np.any(~np.isfinite(z_next)) or np.max(np.abs(z_next)) > STATE_LIMIT:
        return None

    return z_next

def ode_solve(Xi, z0, t):
    t = np.asarray(t, dtype=float)
    z0 = np.asarray(z0, dtype=float).reshape(7)

    Z_tilde = np.zeros((len(t), 7), dtype=float)
    Z_tilde[0] = z0

    for n in range(1, len(t)):
        dt = t[n] - t[n - 1]
        if dt <= 0:
            return None

        z_next = rk4_step(Z_tilde[n - 1], dt, Xi)
        if z_next is None:
            return None

        Z_tilde[n] = z_next

    return Z_tilde

def select_model(Z, Zdot, Theta_scaled, column_norms, t):
    records = []

    alpha_star = None
    delta_star = None
    Xi_star = None
    info_star = None
    Z_tilde_star = None
    error_star = np.inf

    for alpha in ALPHA_SET:
        for delta in DELTA_SET:

            Xi_candidate, info_candidate = stlslr_model(
                Theta_scaled=Theta_scaled,
                Zdot=Zdot,
                alpha=float(alpha),
                delta=float(delta),
                column_norms=column_norms,
                N=MAX_STLSLR_ITER,
            )

            if Xi_candidate is None:
                Z_tilde_candidate = None
                error = np.inf
                active_terms = np.nan
                status = "solver_failed"
            else:
                Z_tilde_candidate = ode_solve(
                    Xi=Xi_candidate,
                    z0=Z[0],
                    t=t,
                )
                error = mean_frobenius_error(Z, Z_tilde_candidate)
                active_terms = int(np.count_nonzero(Xi_candidate))
                status = "ok" if np.isfinite(error) else "diverged"

            records.append({
                "alpha": float(alpha),
                "delta": float(delta),
                "Mean_Frobenius_error": float(error),
                "Active_coefficients": active_terms,
                "Status": status,
            })

            if status == "ok" and error < error_star:
                alpha_star = float(alpha)
                delta_star = float(delta)
                Xi_star = Xi_candidate.copy()
                info_star = {key: value.copy() for key, value in info_candidate.items()}
                Z_tilde_star = Z_tilde_candidate.copy()
                error_star = float(error)

    if Xi_star is None:
        raise ValueError("No stable SINDy model was found over A x D.")

    search_df = pd.DataFrame(records)

    return {
        "alpha_star": alpha_star,
        "delta_star": delta_star,
        "Xi": Xi_star,
        "equation_info": info_star,
        "Z_tilde": Z_tilde_star,
        "Mean_Frobenius_error": error_star,
        "search_table": search_df,
    }

def build_raw_basis_index():
    exponents = []

    exponents.append((0, 0, 0, 0, 0, 0, 0))

    for i in range(7):
        e = [0] * 7
        e[i] = 1
        exponents.append(tuple(e))

    term_exponents = [
        (2,0,0,0,0,0,0),
        (1,1,0,0,0,0,0), (1,0,1,0,0,0,0), (1,0,0,1,0,0,0),
        (1,0,0,0,1,0,0), (1,0,0,0,0,1,0), (1,0,0,0,0,0,1),
        (0,2,0,0,0,0,0),
        (0,1,1,0,0,0,0), (0,1,0,1,0,0,0), (0,1,0,0,1,0,0),
        (0,1,0,0,0,1,0), (0,1,0,0,0,0,1),
        (0,0,2,0,0,0,0),
        (0,0,1,1,0,0,0), (0,0,1,0,1,0,0), (0,0,1,0,0,1,0),
        (0,0,1,0,0,0,1),
        (0,0,0,2,0,0,0),
        (0,0,0,1,1,0,0), (0,0,0,1,0,1,0), (0,0,0,1,0,0,1),
        (0,0,0,0,2,0,0),
        (0,0,0,0,1,1,0), (0,0,0,0,1,0,1),
        (0,0,0,0,0,2,0),
        (0,0,0,0,0,1,1),
        (0,0,0,0,0,0,2),
    ]
    exponents.extend(term_exponents)

    return {exp: i for i, exp in enumerate(exponents)}

RAW_EXP_TO_INDEX = build_raw_basis_index()

TERM_DESCRIPTION = [
    ("lin", 0), ("lin", 1), ("lin", 2), ("lin", 3), ("lin", 4), ("lin", 5), ("lin", 6),
    ("sq", 0),
    ("cross", 0, 1), ("cross", 0, 2), ("cross", 0, 3), ("cross", 0, 4), ("cross", 0, 5), ("cross", 0, 6),
    ("sq", 1),
    ("cross", 1, 2), ("cross", 1, 3), ("cross", 1, 4), ("cross", 1, 5), ("cross", 1, 6),
    ("sq", 2),
    ("cross", 2, 3), ("cross", 2, 4), ("cross", 2, 5), ("cross", 2, 6),
    ("sq", 3),
    ("cross", 3, 4), ("cross", 3, 5), ("cross", 3, 6),
    ("sq", 4),
    ("cross", 4, 5), ("cross", 4, 6),
    ("sq", 5),
    ("cross", 5, 6),
    ("sq", 6),
]

def denormalize_equation(xi_normalized, target_state, normalization):
    xi_normalized = np.asarray(xi_normalized, dtype=float).reshape(35)

    target_scale = normalization[target_state]["scale"]
    mins = np.array([normalization[s]["min"] for s in STATE_NAMES], dtype=float)
    scales = np.array([normalization[s]["scale"] for s in STATE_NAMES], dtype=float)

    raw_coefficients = np.zeros(36, dtype=float)

    def add(exponent, value):
        raw_coefficients[RAW_EXP_TO_INDEX[tuple(exponent)]] += float(value)

    for coefficient, description in zip(xi_normalized, TERM_DESCRIPTION):
        if coefficient == 0.0:
            continue

        c = target_scale * coefficient
        kind = description[0]

        if kind == "lin":
            i = description[1]
            mi, si = mins[i], scales[i]
            add((0,0,0,0,0,0,0), c * (-mi / si))
            e = [0] * 7
            e[i] = 1
            add(e, c / si)

        elif kind == "sq":
            i = description[1]
            mi, si = mins[i], scales[i]
            factor = c / (si ** 2)
            add((0,0,0,0,0,0,0), factor * mi ** 2)
            e1 = [0] * 7
            e1[i] = 1
            add(e1, factor * (-2.0 * mi))
            e2 = [0] * 7
            e2[i] = 2
            add(e2, factor)

        elif kind == "cross":
            i, j = description[1], description[2]
            mi, mj = mins[i], mins[j]
            si, sj = scales[i], scales[j]
            factor = c / (si * sj)
            add((0,0,0,0,0,0,0), factor * mi * mj)
            ei = [0] * 7
            ei[i] = 1
            add(ei, factor * (-mj))
            ej = [0] * 7
            ej[j] = 1
            add(ej, factor * (-mi))
            eij = [0] * 7
            eij[i] = 1
            eij[j] = 1
            add(eij, factor)

    return raw_coefficients

def denormalize_Xi(Xi, normalization):
    Xi_raw = np.zeros((36, 7), dtype=float)
    for j, state in enumerate(STATE_NAMES):
        Xi_raw[:, j] = denormalize_equation(Xi[:, j], state, normalization)
    return Xi_raw

def print_equations(Xi, title, terms):
    print(f"\n{title}")

    for j, state in enumerate(STATE_NAMES):
        print(f"\n{CSV_BASE} — d({state})/dt =")
        active = np.flatnonzero(Xi[:, j] != 0.0)

        if active.size == 0:
            print("  0")
            continue

        for i in active:
            if terms[i] == "1":
                print(f"  {Xi[i, j]:+.6g}")
            else:
                print(f"  {Xi[i, j]:+.6g} * {terms[i]}")

def plot_coefficient_heatmap(df, title, filename):
    values = df.to_numpy(dtype=float)
    zero_mask = values == 0.0

    if np.all(zero_mask):
        print(f"Heatmap skipped because all coefficients are zero: {title}")
        return

    masked = np.ma.array(values, mask=zero_mask)
    base = plt.get_cmap(CMAP_NAME)
    colors = base(np.linspace(TRUNC_LO, TRUNC_HI, 256))
    cmap = ListedColormap(colors)
    cmap.set_bad(color=ZERO_COLOR)

    nonzero = values[~zero_mask]
    vmax = np.max(np.abs(nonzero))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=11)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=8.5)

    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fmt = f"{{:.{SCI_DECIMALS}e}}"
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            value = values[row, col]
            if value != 0.0 and np.isfinite(value):
                intensity = abs(value) / vmax
                text_color = "black" if intensity < 0.65 else "white"
                ax.text(col, row, fmt.format(value), ha="center", va="center",
                        fontsize=7.5, color=text_color)

    ax.set_title(title, pad=12)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(filename, dpi=PNG_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def selected_pair_marker(ax, alpha_idx, delta_idx):
    artist = ax.scatter(alpha_idx, delta_idx, marker="x", s=100,
                        linewidths=2.5, color="white", zorder=6)
    artist.set_path_effects([pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()])
    return artist

def selected_pair_legend_handle():
    handle = Line2D([0], [0], marker="x", linestyle="None", color="white",
                    markeredgecolor="white", markersize=9, markeredgewidth=2.5,
                    label="Selected pair")
    handle.set_path_effects([pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()])
    return handle

def _alpha_delta_heatmap_axes():
    n_alpha, n_delta = len(ALPHA_SET), len(DELTA_SET)
    fig_width = 12.0
    axes_width_fraction = 0.78
    axes_height_inches = fig_width * axes_width_fraction * n_delta / n_alpha
    fig_height = axes_height_inches / 0.60

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.09, 0.32, axes_width_fraction, 0.60])
    cax = fig.add_axes([0.895, 0.32, 0.018, 0.60])
    return fig, ax, cax

def _format_alpha_delta_ticks(ax):
    alpha_tick_idx = np.arange(0, len(ALPHA_SET), 5)
    ax.set_xticks(alpha_tick_idx)
    ax.set_xticklabels(
        [f"{ALPHA_SET[i]:.1e}" for i in alpha_tick_idx],
        rotation=90, fontsize=8
    )
    ax.set_yticks(np.arange(len(DELTA_SET)))
    ax.set_yticklabels([f"{d:.0e}" for d in DELTA_SET], fontsize=8)
    ax.set_xlabel(r"$\alpha \in \mathcal{A}$")
    ax.set_ylabel(r"$\delta \in \mathcal{D}$")

def plot_search_surface(search_df, alpha_star, delta_star, filename):
    pivot = search_df.pivot(index="delta", columns="alpha", values="Mean_Frobenius_error")
    pivot = pivot.reindex(index=DELTA_SET, columns=ALPHA_SET)
    values = pivot.to_numpy(dtype=float)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="red")
    values_plot = np.ma.masked_invalid(values)

    fig, ax, cax = _alpha_delta_heatmap_axes()
    image = ax.imshow(values_plot, aspect="equal", interpolation="nearest", cmap=cmap)
    _format_alpha_delta_ticks(ax)

    ax.text(
        0.975, 0.08, "(a)",
        transform=ax.transAxes,
        fontsize=22, fontweight="bold",
        color="white", ha="right", va="bottom", zorder=10,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="black")],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="white", alpha=0.35)
    )

    alpha_idx = int(np.argmin(np.abs(np.log(ALPHA_SET / alpha_star))))
    delta_idx = int(np.argmin(np.abs(np.log(DELTA_SET / delta_star))))
    selected_pair_marker(ax, alpha_idx, delta_idx)

    ax.legend(
        handles=[selected_pair_legend_handle(),
                 Patch(facecolor="red", edgecolor="red", label="Diverged trajectory")],
        loc="upper right", fontsize=8
    )
    fig.colorbar(image, cax=cax, label="Frobenius error per matrix entry")
    fig.savefig(filename, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)

def plot_active_terms_surface(search_df, alpha_star, delta_star, filename):
    pivot = search_df.pivot(index="delta", columns="alpha", values="Active_coefficients")
    pivot = pivot.reindex(index=DELTA_SET, columns=ALPHA_SET)
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#bcbcbc")

    fig, ax, cax = _alpha_delta_heatmap_axes()
    image = ax.imshow(masked, aspect="equal", interpolation="nearest",
                      cmap=cmap, vmin=0, vmax=35 * 7)
    _format_alpha_delta_ticks(ax)

    ax.text(
        0.975, 0.08, "(b)",
        transform=ax.transAxes,
        fontsize=22, fontweight="bold",
        color="white", ha="right", va="bottom", zorder=10,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="black")],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="white", alpha=0.35)
    )

    alpha_idx = int(np.argmin(np.abs(np.log(ALPHA_SET / alpha_star))))
    delta_idx = int(np.argmin(np.abs(np.log(DELTA_SET / delta_star))))
    selected_pair_marker(ax, alpha_idx, delta_idx)

    ax.legend(handles=[selected_pair_legend_handle()], loc="upper right", fontsize=8)
    fig.colorbar(image, cax=cax, label="Number of active coefficients")
    fig.savefig(filename, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)

def save_validation_figure(fig, basename):
    filename = basename + ".png"
    fig.savefig(filename, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    return filename

def plot_validation(t_full, Z_poly_full, Z_sindy_full, df_test_norm, sigma_poly):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5))
    axes = axes.flatten()
    split_t = df_train_norm["Time"].max()
    test_region = t_full >= split_t
    panel_labels = list("ABCDEFG")

    for j, state in enumerate(STATE_NAMES):
        ax = axes[j]
        sigma = sigma_poly[state]

        ax.plot(t_full, Z_poly_full[:, j], linewidth=2.2,
                color="0.25", label="Polynomial regression")
        ax.plot(t_full, Z_sindy_full[:, j], "--", linewidth=2.3,
                color="orange", label="SINDy")
        ax.scatter(
            df_test_norm["Time"],
            df_test_norm[CSV_COLUMNS[state]],
            s=35, color="tab:blue", marker="s", alpha=0.9,
            label="Held-out data points",
        )
        ax.axvline(split_t, linestyle=":", linewidth=1.8, color="black", alpha=0.8)

        if np.isfinite(sigma) and sigma > 0:
            ax.fill_between(
                t_full[test_region],
                Z_sindy_full[test_region, j] - 1.96 * sigma,
                Z_sindy_full[test_region, j] + 1.96 * sigma,
                alpha=0.19, color="lightcoral", label="Approx. 95% residual band",
            )

        ax.text(0.15, 0.88, panel_labels[j], transform=ax.transAxes,
                fontsize=22, fontweight="bold", ha="center", va="center")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(PLOT_LABELS[state] + " (normalized)")
        ax.grid(True, alpha=0.3)

    for j in range(len(STATE_NAMES), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=True, bbox_to_anchor=(0.5, 0.018))
    fig.tight_layout(rect=(0, 0.085, 1, 1), h_pad=2.0, w_pad=1.5)

    filename = save_validation_figure(fig, f"{CSV_BASE}_SINDy_forward_prediction")
    plt.show()
    plt.close(fig)
    return filename

def plot_synthetic(t_train, Z_noisy, Z_sindy_train, synthetic_df):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5))
    axes = axes.flatten()
    panel_labels = list("ABCDEFG")
    sigma_lookup = synthetic_df.set_index("Variable")["sigma"].to_dict()

    for j, state in enumerate(STATE_NAMES):
        ax = axes[j]
        sigma = sigma_lookup[state]

        ax.scatter(t_train, Z_noisy[:, j], s=24,
                   color="tab:blue", alpha=0.65, label="Synthetic noisy data")
        ax.plot(t_train, Z_sindy_train[:, j], "--", linewidth=2.3,
                color="orange", label="SINDy")

        if np.isfinite(sigma) and sigma > 0:
            ax.fill_between(
                t_train,
                Z_sindy_train[:, j] - 1.96 * sigma,
                Z_sindy_train[:, j] + 1.96 * sigma,
                alpha=0.15, color="lightcoral", label="Approx. 95% residual band",
            )

        ax.text(0.15, 0.88, panel_labels[j], transform=ax.transAxes,
                fontsize=22, fontweight="bold", ha="center", va="center")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(PLOT_LABELS[state] + " (normalized)")
        ax.grid(True, alpha=0.3)

    for j in range(len(STATE_NAMES), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=True, bbox_to_anchor=(0.5, 0.018))
    fig.tight_layout(rect=(0, 0.085, 1, 1), h_pad=2.0, w_pad=1.5)

    filename = save_validation_figure(fig, f"{CSV_BASE}_SINDy_synthetic_validation")
    plt.show()
    plt.close(fig)
    return filename

raw = pd.read_csv(CSV_FILE).sort_values("Time").reset_index(drop=True)
required_columns = ["Time"] + [CSV_COLUMNS[s] for s in STATE_NAMES]
missing = [column for column in required_columns if column not in raw.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

if TRAIN_N_RAW < 5 or TRAIN_N_RAW >= len(raw):
    raise ValueError(
        f"TRAIN_N_RAW must be between 5 and N-1. Got {TRAIN_N_RAW} with N={len(raw)}"
    )

df_train_raw = raw.iloc[:TRAIN_N_RAW].copy()
df_test_raw = raw.iloc[TRAIN_N_RAW:].copy()

t_train_raw = df_train_raw["Time"].to_numpy(dtype=float)
t_full_raw = raw["Time"].to_numpy(dtype=float)
t_train = np.linspace(t_train_raw.min(), t_train_raw.max(), NUM_POINTS)
t_full = np.linspace(t_full_raw.min(), t_full_raw.max(), NUM_POINTS)

normalization = {}
df_train_norm = df_train_raw.copy()
df_test_norm = df_test_raw.copy()
df_all_norm = raw.copy()

for state in STATE_NAMES:
    column = CSV_COLUMNS[state]
    xmin, xmax, scale = minmax_fit(df_train_raw[column])
    normalization[state] = {"min": xmin, "max": xmax, "scale": scale}

    df_train_norm[column] = minmax_transform(df_train_raw[column], xmin, scale)
    df_test_norm[column] = minmax_transform(df_test_raw[column], xmin, scale)
    df_all_norm[column] = minmax_transform(raw[column], xmin, scale)

polynomial_models = {}
polynomial_features = {}
best_degree = {}
polynomial_training_rmse = {}
sigma_poly = {}

Z = np.zeros((NUM_POINTS, 7), dtype=float)
Zdot = np.zeros((NUM_POINTS, 7), dtype=float)
Z_poly_full = np.zeros((NUM_POINTS, 7), dtype=float)

for j, state in enumerate(STATE_NAMES):
    column = CSV_COLUMNS[state]
    t_observed = df_train_norm["Time"].to_numpy(dtype=float)
    z_observed = df_train_norm[column].to_numpy(dtype=float)

    fit_rmse, degree, model, poly = fit_polynomial_trajectory(
        t_raw=t_observed,
        z_raw=z_observed,
        degree_min=DEGREE_MIN,
        degree_max=DEGREE_MAX,
    )

    polynomial_models[state] = model
    polynomial_features[state] = poly
    best_degree[state] = degree
    polynomial_training_rmse[state] = fit_rmse

    Z[:, j] = model.predict(poly.transform(t_train.reshape(-1, 1)))
    Zdot[:, j] = polynomial_derivative(model, poly, t_train)

    Z_poly_full[:, j] = model.predict(poly.transform(t_full.reshape(-1, 1)))

    z_fit_observed = model.predict(poly.transform(t_observed.reshape(-1, 1)))
    residuals = z_observed - z_fit_observed
    sigma_poly[state] = float(np.std(residuals, ddof=1))

Theta = library_matrix(Z)

library_column_norms = np.linalg.norm(Theta, axis=0)
library_column_norms[np.isclose(library_column_norms, 0.0)] = 1.0
Theta_scaled = Theta / library_column_norms

selection = select_model(
    Z=Z,
    Zdot=Zdot,
    Theta_scaled=Theta_scaled,
    column_norms=library_column_norms,
    t=t_train,
)

alpha_star = selection["alpha_star"]
delta_star = selection["delta_star"]
Xi = selection["Xi"]
selected_equation_info = selection["equation_info"]
Z_tilde_train = selection["Z_tilde"]
training_mean_frobenius_error = selection["Mean_Frobenius_error"]
search_df = selection["search_table"]

training_records = []
for j, state in enumerate(STATE_NAMES):
    info = selected_equation_info[state]
    training_records.append({
        "Variable": state,
        "RMSE_train": rmse(Z[:, j], Z_tilde_train[:, j]),
        "L2_train": l2_error(Z[:, j], Z_tilde_train[:, j]),
        "STLSLR_iterations": info["iterations"],
        "STLSLR_converged": info["converged"],
        "Active_terms": info["active_terms"],
    })

training_df = pd.DataFrame(training_records)

Z_tilde_full = ode_solve(Xi=Xi, z0=Z_poly_full[0], t=t_full)
if Z_tilde_full is None:
    raise ValueError("The selected model diverged during full forward simulation.")

validation_records = []
t_test = df_test_norm["Time"].to_numpy(dtype=float)

for j, state in enumerate(STATE_NAMES):
    z_test = df_test_norm[CSV_COLUMNS[state]].to_numpy(dtype=float)
    z_pred = interpolate_at_times(t_full, Z_tilde_full[:, j], t_test)
    residual = z_test - z_pred
    sigma = sigma_poly[state]

    validation_records.append({
        "Variable": state,
        "RMSE_val": rmse(z_test, z_pred),
        "sigma": sigma,
        "coverage_95": float(np.mean(np.abs(residual) <= 1.96 * sigma)) if sigma > 0 else np.nan,
    })

validation_df = pd.DataFrame(validation_records)

rng = np.random.default_rng(NOISE_SEED)
sigma_vector = np.array([sigma_poly[state] for state in STATE_NAMES], dtype=float)

noise = rng.normal(
    loc=0.0,
    scale=(NOISE_SCALE * sigma_vector).reshape(1, 7),
    size=Z.shape,
)
Z_noisy = Z + noise

synthetic_records = []
for j, state in enumerate(STATE_NAMES):
    z_noisy = Z_noisy[:, j]
    z_pred = Z_tilde_train[:, j]
    residual = z_noisy - z_pred
    sigma = sigma_vector[j]

    synthetic_records.append({
        "Variable": state,
        "RMSE_syn": rmse(z_noisy, z_pred),
        "sigma": sigma,
        "coverage_95": float(np.mean(np.abs(residual) <= 1.96 * sigma)) if sigma > 0 else np.nan,
    })

synthetic_df = pd.DataFrame(synthetic_records)

Xi_df = pd.DataFrame(Xi, index=THETA_TERMS, columns=STATE_NAMES)
Xi_raw = denormalize_Xi(Xi, normalization)
Xi_raw_df = pd.DataFrame(Xi_raw, index=RAW_TERMS, columns=STATE_NAMES)

normalized_matrix_file = f"{CSV_BASE}_SINDy_Xi_normalized.csv"
dimensional_matrix_file = f"{CSV_BASE}_SINDy_Xi_dimensional.csv"
validation_file = f"{CSV_BASE}_SINDy_forward_prediction_metrics.csv"

Xi_df.to_csv(normalized_matrix_file)
Xi_raw_df.to_csv(dimensional_matrix_file)
validation_df.to_csv(validation_file, index=False)

active_coefficients = int(np.count_nonzero(Xi))
total_coefficients = Xi.size
active_fraction = 100.0 * active_coefficients / total_coefficients

print("\nFINAL SINDy MODEL — STLSLR REGRESSION")
print(f"System: {CSV_BASE}")
print(f"Raw observations: {len(raw)}")
print(f"D_train: {len(df_train_raw)} observations")
print(f"D_test:  {len(df_test_raw)} observations")
print(f"M: {NUM_POINTS}")
print(f"A: {len(ALPHA_SET)} alpha values from {ALPHA_SET.min():.1e} to {ALPHA_SET.max():.1e}")
print(f"D: {', '.join(f'{d:.0e}' for d in DELTA_SET)}")
print(f"N: {MAX_STLSLR_ITER}")
print(f"Selected alpha*: {alpha_star:.6g}")
print(f"Selected delta*: {delta_star:.6g}")
print(f"Training Frobenius norm per matrix entry: {training_mean_frobenius_error:.6g}")
print(f"Active coefficients: {active_coefficients}/{total_coefficients} ({active_fraction:.2f}%)")
print(f"Mean training RMSE: {training_df['RMSE_train'].mean():.6g}")
print(f"Mean forward prediction RMSE: {validation_df['RMSE_val'].mean():.6g}")
print(f"Mean forward prediction 95% coverage: {100 * validation_df['coverage_95'].mean():.1f}%")
print(f"Mean synthetic RMSE: {synthetic_df['RMSE_syn'].mean():.6g}")
print(f"Mean synthetic 95% coverage: {100 * synthetic_df['coverage_95'].mean():.1f}%")

print("\nPolynomial degrees d_j:")
for state in STATE_NAMES:
    print(f"  {state:>4s}: d_j = {best_degree[state]}")

print_equations(Xi, "NORMALIZED FINAL EQUATIONS", THETA_TERMS)
print_equations(Xi_raw, "DIMENSIONAL FINAL EQUATIONS", RAW_TERMS)

print("\nSINDy forward prediction metrics:")
print(validation_df.to_string(index=False))

print("\nSINDy synthetic-perturbation metrics:")
print(synthetic_df.to_string(index=False))

saved_figures = []

if PLOT_HYPERPARAMETER_SURFACE:
    file = f"{CSV_BASE}_SINDy_alpha_delta_surface.png"
    plot_search_surface(search_df, alpha_star, delta_star, file)
    saved_figures.append(file)

if PLOT_ACTIVE_TERMS_SURFACE:
    file = f"{CSV_BASE}_SINDy_alpha_delta_active_coefficients.png"
    plot_active_terms_surface(search_df, alpha_star, delta_star, file)
    saved_figures.append(file)

if PLOT_VALIDATION:
    file = plot_validation(
        t_full=t_full,
        Z_poly_full=Z_poly_full,
        Z_sindy_full=Z_tilde_full,
        df_test_norm=df_test_norm,
        sigma_poly=sigma_poly,
    )
    saved_figures.append(file)

if PLOT_SYNTHETIC_VALIDATION:
    file = plot_synthetic(
        t_train=t_train,
        Z_noisy=Z_noisy,
        Z_sindy_train=Z_tilde_train,
        synthetic_df=synthetic_df,
    )
    saved_figures.append(file)

if PLOT_HEATMAPS:
    file = f"{CSV_BASE}_SINDy_Xi_normalized_heatmap.png"
    plot_coefficient_heatmap(
        Xi_df,
        f"{CSV_BASE} — normalized coefficient matrix Xi",
        file,
    )
    saved_figures.append(file)

print("\nSAVED FILES")

for file in [
    normalized_matrix_file,
    dimensional_matrix_file,
    validation_file,
]:
    print(f"  {file}")

for file in saved_figures:
    print(f"  {file}")

print("\nDone.")


# In[ ]:




