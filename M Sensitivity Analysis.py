#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SYSTEM_ORDER = ["W20P", "W20U", "C20P", "C20U", "W5P", "W5U", "C5P", "C5U"]
M_VALUES = np.arange(50, 201, 10)
TOTAL_COEFFICIENTS = 35 * 7
OUTPUT_FILE = Path("All_systems_combined_objective.png")

# Active counts from the supplied 35 x 7 SINDy coefficient matrices.
# RMSE values are the five-point forward-prediction RMSEs.
DATA = {
    "W20P": {
        "active": [23, 90, 24, 22, 17, 39, 39, 39, 182, 19, 139, 179, 181, 26, 29, 32],
        "rmse": [0.144697, 0.264813, 0.291816, 0.168737, 0.135233, 0.137683, 0.140794, 0.129088, 0.220875, 0.152254, 0.179203, 0.157654, 0.201078, 0.165177, 0.154999, 0.157846],
    },
    "W20U": {
        "active": [14, 15, 19, 13, 13, 14, 15, 15, 14, 12, 14, 14, 14, 14, 15, 13],
        "rmse": [0.261789, 0.252824, 0.256061, 0.27442, 0.252365, 0.251038, 0.259375, 0.267103, 0.260349, 0.242304, 0.247535, 0.252434, 0.257131, 0.261641, 0.264885, 0.265237],
    },
    "C20P": {
        "active": [14, 15, 17, 18, 15, 15, 17, 17, 18, 19, 19, 16, 17, 17, 17, 17],
        "rmse": [0.296141, 0.268678, 0.310035, 0.289842, 0.276334, 0.268259, 0.306982, 0.295068, 0.286117, 0.279818, 0.275449, 0.272838, 0.311179, 0.304149, 0.2981, 0.292905],
    },
    "C20U": {
        "active": [51, 20, 19, 22, 21, 21, 20, 23, 21, 21, 21, 19, 25, 25, 23, 24],
        "rmse": [0.182742, 0.188743, 0.199998, 0.167648, 0.170642, 0.182195, 0.188164, 0.168306, 0.167349, 0.170223, 0.177788, 0.183316, 0.194438, 0.182449, 0.1746, 0.169758],
    },
    "W5P": {
        "active": [6, 11, 14, 7, 12, 14, 9, 18, 12, 16, 15, 15, 10, 51, 50, 17],
        "rmse": [0.532963, 0.411903, 0.411903, 0.414367, 0.231716, 0.715377, 0.421197, 0.381097, 0.251142, 0.212253, 0.441717, 1.35007, 0.45994, 0.324053, 0.271287, 0.212423],
    },
    "W5U": {
        "active": [14, 46, 47, 10, 14, 15, 15, 14, 13, 15, 15, 15, 15, 14, 16, 16],
        "rmse": [0.281396, 0.276547, 0.275849, 0.264799, 0.263662, 0.258478, 0.260567, 0.252238, 0.247034, 0.254261, 0.247179, 0.242679, 0.239941, 0.23839, 0.243462, 0.238759],
    },
    "C5P": {
        "active": [20, 29, 14, 124, 58, 14, 14, 36, 35, 35, 18, 17, 17, 14, 14, 64],
        "rmse": [0.176543, 0.228962, 0.160625, 0.197376, 0.200704, 0.158632, 0.157726, 0.18079, 0.186891, 0.203638, 0.163708, 0.162186, 0.161455, 0.157168, 0.157024, 0.183181],
    },
    "C5U": {
        "active": [68, 153, 19, 19, 22, 22, 79, 99, 62, 98, 13, 13, 13, 11, 11, 11],
        "rmse": [0.693369, 0.74249, 0.199364, 0.215252, 0.193201, 0.308592, 0.763558, 0.719609, 0.715435, 0.686657, 0.196686, 0.194318, 0.191808, 0.186994, 0.18421, 0.181313],
    },
}


def combined_objective(active_counts, forward_rmse):
    rmse = np.asarray(forward_rmse, dtype=float)
    active_fraction = np.asarray(active_counts, dtype=float) / TOTAL_COEFFICIENTS
    if rmse.shape != active_fraction.shape or not np.all(np.isfinite(rmse)):
        raise ValueError("Active counts and forward-prediction RMSE must align and be finite.")
    if np.any(rmse < 0):
        raise ValueError("RMSE must be nonnegative.")
    low, high = rmse.min(), rmse.max()
    rmse_normalized = np.zeros_like(rmse) if np.isclose(low, high, rtol=1e-12, atol=0) else (rmse - low) / (high - low)
    return rmse_normalized + active_fraction


fig, axes = plt.subplots(3, 3, figsize=(13, 6.5))
axes = axes.flat

for i, system in enumerate(SYSTEM_ORDER):
    data = DATA[system]
    f = combined_objective(data["active"], data["rmse"])
    best = int(np.argmin(f))
    M_star = int(M_VALUES[best])
    ax = axes[i]
    ax.plot(M_VALUES, f, "o-", linewidth=1.8, markersize=4.5, label="Combined objective")
    ax.scatter(M_star, f[best], marker="*", s=160, color="black", zorder=5,
               label="Objective minimum")
    ax.text(0.03, 0.95, "ABCDEFGH"[i], transform=ax.transAxes,
            fontsize=17, fontweight="bold", va="top")
    ax.set_xticks(M_VALUES[::2])
    ax.grid(alpha=0.20)
    if i // 3 != 2:
        ax.set_xticklabels([])
    if i % 3 != 0:
        ax.set_yticklabels([])
    print(f"{system}: M* = {M_star}, f(M*) = {f[best]:.6f}")

axes[-1].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.005))
fig.supxlabel(r"Interpolation resolution, $M$", y=0.065, fontsize=12)
fig.supylabel(r"Combined objective, $f(M)$", x=0.025, fontsize=12)
fig.tight_layout(rect=[0.04, 0.10, 1.00, 0.99], h_pad=0.5, w_pad=0.5)
fig.savefig(OUTPUT_FILE, dpi=600, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Figure saved: {OUTPUT_FILE}")


# In[ ]:




