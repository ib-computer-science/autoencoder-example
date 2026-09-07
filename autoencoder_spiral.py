"""MLP autoencoder with a 1D bottleneck, trained on a jittered 2D spiral.

Visualizes:
  1. the original (jittered) spiral data
  2. the autoencoder's reconstruction of that data
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# Style: chart chrome + a single-hue sequential ramp (blue, light->dark)
# and the series-1 blue, per the project's data-viz palette.
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"

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
# 1. Data: a jittered 2D spiral
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)

N = 2000
T_MIN, T_MAX = 1.0 * np.pi, 3.3 * np.pi
A = 0.2
JITTER_STD = 0.05

t = rng.uniform(T_MIN, T_MAX, size=N)
t.sort()
x_clean = A * t * np.cos(t)
y_clean = A * t * np.sin(t)
x = x_clean + rng.normal(0, JITTER_STD, size=N)
y = y_clean + rng.normal(0, JITTER_STD, size=N)
data = np.stack([x, y], axis=1)

# normalize using the training data's statistics
mean = data.mean(axis=0)
std = data.std(axis=0)
data_n = (data - mean) / std

data_t = torch.tensor(data_n, dtype=torch.float32)


# ----------------------------------------------------------------------
# 2. Model: MLP autoencoder, bottleneck width 1
# ----------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
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
# 3. Evaluate: reconstruct the data
# ----------------------------------------------------------------------
model.eval()
with torch.no_grad():
    recon_n, z_data = model(data_t)

recon = recon_n.numpy() * std + mean

# ----------------------------------------------------------------------
# 4. Visualize
# ----------------------------------------------------------------------
def draw_network_diagram(ax, layer_sizes, max_nodes=6):
    """Schematic of a fully-connected MLP. Layers wider than max_nodes are
    drawn with a capped number of nodes (the true width is labeled below)
    so the diagram stays legible."""
    n_layers = len(layer_sizes)
    bottleneck_idx = int(np.argmin(layer_sizes))
    xs = np.linspace(0.0, 1.0, n_layers)

    node_positions = []
    for n in layer_sizes:
        shown = min(n, max_nodes)
        ys = np.linspace(-1, 1, shown) if shown > 1 else np.array([0.0])
        node_positions.append(ys)

    # connections, drawn first so nodes sit on top
    for li in range(n_layers - 1):
        for y0 in node_positions[li]:
            for y1 in node_positions[li + 1]:
                ax.plot(
                    [xs[li], xs[li + 1]], [y0, y1],
                    color=BASELINE, linewidth=0.5, alpha=0.4, zorder=1,
                )

    for li, n in enumerate(layer_sizes):
        is_bottleneck = li == bottleneck_idx
        is_endpoint = li in (0, n_layers - 1)
        if is_bottleneck:
            face, edge, size = SERIES_BLUE, SERIES_BLUE, 190
        elif is_endpoint:
            face, edge, size = SURFACE, INK_SECONDARY, 120
        else:
            face, edge, size = BASELINE, BASELINE, 95
        ys = node_positions[li]
        ax.scatter(
            np.full_like(ys, xs[li]), ys, s=size, facecolor=face,
            edgecolor=edge, linewidths=1.1, zorder=2,
        )
        ax.text(
            xs[li], -1.32, str(n), ha="center", va="top", fontsize=8.5,
            color=INK_SECONDARY,
        )

    role_labels = ["input", "hidden", "hidden", "bottleneck", "hidden", "hidden", "output"]
    for li, role in enumerate(role_labels[:n_layers]):
        ax.text(
            xs[li], 1.32, role, ha="center", va="bottom", fontsize=7.5,
            color=INK_MUTED, rotation=30, rotation_mode="anchor",
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.55, 1.75)
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax_net, ax_orig, ax_recon = axes

draw_network_diagram(ax_net, layer_sizes=[2, 16, 16, 1, 16, 16, 2])
ax_net.set_title("Network architecture", color=INK_PRIMARY, fontsize=11, pad=18)

t_norm = (t - T_MIN) / (T_MAX - T_MIN)

# Panel 1: original jittered data, colored by position along the spiral
ax_orig.scatter(
    data[:, 0], data[:, 1], c=t_norm, cmap=SEQUENTIAL_BLUE, s=10, alpha=0.85,
    linewidths=0,
)
ax_orig.set_title("Original data (spiral + jitter)", color=INK_PRIMARY, fontsize=11)

# Panel 2: original (faint reference) + reconstruction, same color encoding
ax_recon.scatter(
    data[:, 0], data[:, 1], c=BASELINE, s=10, alpha=0.5, linewidths=0,
    label="original",
)
ax_recon.scatter(
    recon[:, 0], recon[:, 1], c=t_norm, cmap=SEQUENTIAL_BLUE, s=10, alpha=0.9,
    linewidths=0, label="reconstruction",
)
ax_recon.set_title("Reconstruction", color=INK_PRIMARY, fontsize=11)
ax_recon.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
    fontsize=9, labelcolor=INK_SECONDARY, markerscale=2,
)

for ax in (ax_orig, ax_recon):
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle(
    "1D-bottleneck MLP autoencoder learning a jittered 2D spiral",
    color=INK_PRIMARY, fontsize=13, y=1.02,
)
fig.tight_layout()
fig.savefig("autoencoder_spiral.png", dpi=150, bbox_inches="tight")
print("saved autoencoder_spiral.png")
plt.show()
