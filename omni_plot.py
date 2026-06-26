from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "solar_wind_data.csv", parse_dates=["datetime"])
public_private_boundary = pd.Timestamp("2024-01-01")

plots = [
    ("Speed (km/s)", "Speed (km/s)"),
    # ("Bx_GSE (nT)", "Bx GSE (nT)"),
    # ("By_GSE (nT)", "By GSE (nT)"),
    # ("Bz_GSE (nT)", "Bz GSE (nT)"),
    ("Density (1/cm^3)", "Density (1/cm^3)"),
    ("Temperature (K)", "Temperature (K)"),
    ("B (nT)", "B (nT)"),
    # ("Flow Pressure (nPa)", "Flow Pressure (nPa)"),
    # ("Electric Field (mV/m)", "Electric Field (mV/m)"),
    # ("Plasma Beta", "Plasma Beta"),
    # ("Alfven Mach Number", "Alfven Mach Number"),
    ("Sunspot Number", "Sunspot Number"),
    # ("F10.7 Flux (sfu)", "F10.7 Flux (sfu)"),
    ("Coronal Hole Area", "Coronal Hole Area"),
]

fig, axes = plt.subplots(len(plots), 1, figsize=(30, 15), sharex=True)

for ax, (column, label) in zip(axes, plots):
    ax.plot(data["datetime"], data[column], linewidth=0.5)
    ax.axvline(
        public_private_boundary,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

# 첫 번째 subplot 위에 두 기간의 이름을 표시한다.
public_center = data["datetime"].min() + (
    public_private_boundary - data["datetime"].min()
) / 2
private_center = public_private_boundary + (
    data["datetime"].max() - public_private_boundary
) / 2

axes[0].text(
    public_center,
    0.92,
    "PUBLIC (2011–2023)",
    transform=axes[0].get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=11,
    fontweight="bold",
)
axes[0].text(
    private_center,
    0.92,
    "PRIVATE (2024–2025)",
    transform=axes[0].get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=11,
    fontweight="bold",
    color="crimson",
)

axes[-1].set_xlabel("Year")

fig.suptitle("OMNI Solar Wind Data (2011–2025)", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(folder / "solar_wind_overview.png", dpi=180)
plt.close(fig)
