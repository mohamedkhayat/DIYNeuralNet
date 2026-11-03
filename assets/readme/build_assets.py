"""Build the README visuals by training with the library itself. No numbers are typed by hand.

  boundary.gif               an MLP learning the two-spirals problem, one frame per few epochs
  optimizers.json            MNIST validation accuracy and wall time for the four optimizers (same net, same seed)

Run:  .venv/bin/python assets/readme/build_assets.py   (uses CuPy when available, NumPy otherwise)
"""

import io
import json
import os
import platform
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as onp  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

from diy_neural_net import (  # noqa: E402
    BatchNorm, BCELoss, CrossEntropyLossWithLogits, Dense, NeuralNetwork, ReLU, Sigmoid, Tanh, get_numpy,
    is_gpu_available, utils,
)
from diy_neural_net.Optimizer import RMSProp, Adam, GradientDescent, GradientDescentWithMomentum  # noqa: E402

np = get_numpy()
HERE = Path(__file__).resolve().parent
FONTS = HERE / ".fonts"
FONT_URLS = {
    "Inter-400.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf",
    "Inter-600.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuGKYMZg.ttf",
    "JBMono-400.ttf": "https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPQ.ttf",
}
THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#d8dee4", "blue": "#2a78d6", "orange": "#eb6834"},
    "dark": {"bg": "#0d1117", "ink": "#f0f6fc", "muted": "#9198a1", "grid": "#262c36", "blue": "#3987e5", "orange": "#e8692f"},
}
CPU = lambda a: a.get() if hasattr(a, "get") else onp.asarray(a)  # noqa: E731
def _device():
    if is_gpu_available():
        import cupy

        return f"GPU (CuPy, {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode().replace('NVIDIA GeForce ', '')})"
    return f"CPU (NumPy, {platform.processor() or platform.machine()})"


DEVICE = _device()


def ensure_fonts():
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        if not (FONTS / name).exists():
            urllib.request.urlretrieve(url, FONTS / name)
        font_manager.fontManager.addfont(str(FONTS / name))
    plt.rcParams.update({"svg.fonttype": "path", "font.family": "Inter", "font.size": 10, "svg.hashsalt": "diy-neural-net",
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0})


# --------------------------------------------------------------- spirals GIF
def spirals(n=600, noise=0.25, seed=0):
    r = onp.random.default_rng(seed)
    t = onp.sqrt(r.uniform(0.05, 1, n)) * 3.5 * onp.pi
    a = onp.stack([t * onp.cos(t), t * onp.sin(t)]) / 11 + r.normal(0, noise / 11, (2, n))
    X = onp.concatenate([a, -a], axis=1)
    y = onp.concatenate([onp.zeros(n), onp.ones(n)]).reshape(1, -1)
    return np.asarray(X), np.asarray(y)


def boundary_gif():
    onp.random.seed(0)
    if hasattr(np, "random") and np is not onp:
        np.random.seed(0)
    X, y = spirals()
    model = NeuralNetwork.Sequential([Dense(2, 48), Tanh(), Dense(48, 48), Tanh(), Dense(48, 1, initializer="glorot"), Sigmoid()])
    model.set_loss(BCELoss())
    opt = Adam(model, lr=0.01)

    g = onp.linspace(-1.45, 1.45, 160)
    gx, gy = onp.meshgrid(g, g)
    grid = np.asarray(onp.stack([gx.ravel(), gy.ravel()]))
    t = THEMES["dark"]
    snap = sorted({int(round(e)) for e in onp.geomspace(1, 400, 46)})
    losses, accs, frames, epoch = [], [], [], 0
    Xc, yc = CPU(X), CPU(y).ravel()

    for target in snap:
        while epoch < target:
            h = model.fit(X, y, optimizer=opt, epochs=1, batch_size=64)
            losses.append(h["Train_losses"][0])
            accs.append(h["Train_accuracy"][0])
            epoch += 1
        p = CPU(model.forward(grid, train=False)).reshape(gx.shape)

        fig = plt.figure(figsize=(9.6, 4.6), dpi=100)
        fig.patch.set_facecolor(t["bg"])
        ax = fig.add_axes([0.02, 0.05, 0.42, 0.84])
        ax.contourf(gx, gy, p, levels=onp.linspace(0, 1, 21), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "bo", ["#1d4f8f", "#0d1117", "#9c4320"]), vmin=0, vmax=1)
        ax.contour(gx, gy, p, levels=[0.5], colors=[t["ink"]], linewidths=1.2)
        ax.scatter(Xc[0, yc == 0], Xc[1, yc == 0], s=7, color=t["blue"], linewidths=0)
        ax.scatter(Xc[0, yc == 1], Xc[1, yc == 1], s=7, color=t["orange"], linewidths=0)
        ax.set(xticks=[], yticks=[], xlim=(-1.45, 1.45), ylim=(-1.45, 1.45))
        for s in ax.spines.values():
            s.set_visible(False)
        fig.text(0.02, 0.93, f"epoch {epoch:>3}", color=t["ink"], fontsize=13, weight=600, family="JetBrains Mono")
        fig.text(0.44, 0.93, "decision boundary (P = 0.5)", color=t["muted"], fontsize=10, ha="right")

        for k, (vals, label, color, ylim) in enumerate([(losses, "train loss (BCE)", t["orange"], (0, 0.75)),
                                                         (accs, "train accuracy", t["blue"], (0.45, 1.0))]):
            a2 = fig.add_axes([0.53, 0.56 - k * 0.46, 0.44, 0.33])
            a2.set_facecolor(t["bg"])
            a2.plot(range(1, len(vals) + 1), vals, color=color, lw=1.8)
            a2.set(xlim=(1, 400), ylim=ylim, xscale="log")
            a2.tick_params(colors=t["muted"], labelsize=8.5)
            a2.grid(color=t["grid"], lw=0.6)
            for s in ("left", "bottom"):
                a2.spines[s].set_color(t["grid"])
            a2.text(0.0, 1.06, f"{label}   {vals[-1]:.3f}", transform=a2.transAxes, color=t["ink"], fontsize=10,
                    family="JetBrains Mono")
        a2.set_xlabel("epoch (log scale)", color=t["muted"], fontsize=9)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=t["bg"])
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))

    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    durations = [140] * len(q)
    durations[-1] = 2500
    q[0].save(HERE / "boundary.gif", save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)
    print(f"spirals: final train accuracy {accs[-1]:.3f}, loss {losses[-1]:.3f}")


# --------------------------------------------------------- optimizer study
def optimizer_study(epochs=12):
    X, y = utils.load_mnist()
    onp.random.seed(0)
    if np is not onp:
        np.random.seed(0)
    X_train, X_val, y_train, y_val = utils.train_test_split(X, y, test_size=0.15)
    runs = {
        "SGD (lr 0.1)": lambda m: GradientDescent(m, lr=0.1),
        "Momentum (lr 0.1, β 0.9)": lambda m: GradientDescentWithMomentum(m, lr=0.1, momentum=0.9),
        "RMSProp (lr 1e-3)": lambda m: RMSProp(m, lr=1e-3),
        "Adam (lr 1e-3)": lambda m: Adam(m, lr=1e-3),
    }
    out = {"device": DEVICE, "epochs": epochs, "train_samples": int(X_train.shape[1]), "val_samples": int(X_val.shape[1]),
           "architecture": "784-512-256-128-10, BatchNorm + ReLU, batch 64", "runs": {}}
    for name, make in runs.items():
        onp.random.seed(1)
        if np is not onp:
            np.random.seed(1)
        model = NeuralNetwork.Sequential([Dense(784, 512), BatchNorm(512), ReLU(), Dense(512, 256), BatchNorm(256), ReLU(),
                                          Dense(256, 128), BatchNorm(128), ReLU(), Dense(128, 10, initializer="glorot")])
        model.set_loss(CrossEntropyLossWithLogits())
        h = model.fit(X_train, y_train, optimizer=make(model), epochs=epochs, batch_size=64, validation_data=(X_val, y_val))
        out["runs"][name] = {"val_accuracy": [float(a) for a in h["Test_accuracy"]], "val_loss": [float(v) for v in h["Test_losses"]],
                             "seconds": h["Time_Elapsed"]}
        print(f"{name}: val acc {h['Test_accuracy'][-1]:.4f} in {h['Time_Elapsed']:.1f}s")
    (HERE / "optimizers.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    os.chdir(HERE)
    ensure_fonts()
    boundary_gif()
    if not (HERE / "optimizers.json").exists() or os.environ.get("RERUN"):
        optimizer_study()
