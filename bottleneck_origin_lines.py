"""Diagnostic: lines from the origin to the reconstructed spiral at a few
positions (t-fraction 0, 0.4, 0.7, 0.95, plus 0.85 highlighted in red).

Each line is colored to match the reconstruction's color at that point
(same sequential-blue encoding as autoencoder_spiral.py), making it easy to
see where each sampled position sits on the curve and how far apart they
are in angle around the origin. Same data/model setup as
autoencoder_spiral.py, duplicated here to keep this script standalone.
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# Style: chart chrome + a single-hue sequential ramp (blue, light->dark),
# per the project's data-viz palette.
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
RED = "#e34948"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "font.family": "sans-serif",
        "font.size": 10,
    }
)

# ----------------------------------------------------------------------
# 1. Data: a jittered 2D spiral (same construction as autoencoder_spiral.py)
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)

N = 2000
T_MIN, T_MAX = 1.0 * np.pi, 3.3 * np.pi
A = 0.2
JITTER_STD = 0.05

t = rng.uniform(T_MIN, T_MAX, size=N)
t.sort()
x = A * t * np.cos(t) + rng.normal(0, JITTER_STD, size=N)
y = A * t * np.sin(t) + rng.normal(0, JITTER_STD, size=N)
data = np.stack([x, y], axis=1)

mean = data.mean(axis=0)
std = data.std(axis=0)
data_n = (data - mean) / std
data_t = torch.tensor(data_n, dtype=torch.float32)

# ----------------------------------------------------------------------
# 2. Model: same MLP autoencoder as autoencoder_spiral.py
# ----------------------------------------------------------------------
HIDDEN_WIDTH = 4


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTH, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTH, 2),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


torch.manual_seed(0)
model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12000, eta_min=1e-5)
loss_fn = nn.MSELoss()

EPOCHS = 12000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    recon, _ = model(data_t)
    loss = loss_fn(recon, data_t)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 2000 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:5d}  mse {loss.item():.5f}")

# ----------------------------------------------------------------------
# 3. Evaluate: reconstruct the data, and a fine noise-free sweep used only
#    to locate the exact reconstructed point at each target t-fraction.
# ----------------------------------------------------------------------
model.eval()
with torch.no_grad():
    recon_n, _ = model(data_t)
recon = recon_n.numpy() * std + mean

t_fine = np.linspace(T_MIN, T_MAX, 2000)
sweep_clean = np.stack([A * t_fine * np.cos(t_fine), A * t_fine * np.sin(t_fine)], axis=1)
sweep_n = (sweep_clean - mean) / std
with torch.no_grad():
    recon_fine, _ = model(torch.tensor(sweep_n, dtype=torch.float32))
recon_fine = recon_fine.numpy() * std + mean

frac = (t_fine - T_MIN) / (T_MAX - T_MIN)
t_norm = (t - T_MIN) / (T_MAX - T_MIN)

# ----------------------------------------------------------------------
# 4. Visualize
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7))

ax.scatter(data[:, 0], data[:, 1], c=BASELINE, s=10, alpha=0.5, linewidths=0, zorder=1)
ax.scatter(
    recon[:, 0], recon[:, 1], c=t_norm, cmap=SEQUENTIAL_BLUE, s=10, alpha=0.9,
    linewidths=0, zorder=2,
)

TARGET_FRACS = [0.0, 0.4, 0.7, 0.95]
for tf in TARGET_FRACS:
    idx = np.argmin(np.abs(frac - tf))
    px, py = recon_fine[idx]
    color = SEQUENTIAL_BLUE(tf)
    ax.plot([0, px], [0, py], color=color, linewidth=2.2, zorder=3, solid_capstyle="round")
    ax.scatter([px], [py], color=color, s=60, zorder=4, edgecolor=INK_PRIMARY, linewidths=0.8)

# highlighted reference line at t-fraction 0.85, inside the "seam" region
idx85 = np.argmin(np.abs(frac - 0.85))
px85, py85 = recon_fine[idx85]
ax.plot([0, px85], [0, py85], color=RED, linewidth=2.2, zorder=3, solid_capstyle="round")
ax.scatter([px85], [py85], color=RED, s=60, zorder=4, edgecolor=INK_PRIMARY, linewidths=0.8)

ax.scatter([0], [0], color=INK_PRIMARY, s=25, zorder=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.grid(True, linewidth=0.6)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig("bottleneck_origin_lines.png", dpi=150, bbox_inches="tight")
print("saved bottleneck_origin_lines.png")
plt.show()
