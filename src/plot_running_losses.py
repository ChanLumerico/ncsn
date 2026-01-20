import argparse
import os
import sys
from typing import Any, Dict, List

import torch


def _resolve_ckpt_path(ckpt: str, out_dir: str) -> str:
    lower = ckpt.lower()
    if lower in ("latest", "last", "ckpt_latest"):
        path = os.path.join(out_dir, "checkpoints", "ckpt_latest.pt")
    elif os.path.isdir(ckpt):
        candidate = os.path.join(ckpt, "ckpt_latest.pt")
        if os.path.exists(candidate):
            path = candidate
        else:
            path = os.path.join(ckpt, "checkpoints", "ckpt_latest.pt")
    else:
        path = ckpt

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: `{path}`")
    return path


def _extract_running_losses(ckpt_obj: Dict[str, Any]) -> List[float]:
    losses = ckpt_obj.get("running_losses", [])
    if losses is None:
        return []
    if isinstance(losses, list):
        return [float(v) for v in losses]
    return [float(v) for v in list(losses)]


def plot_running_losses(
    ckpt_path: str, save_path: str, show: bool = False, rolling: int = 0
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    running_losses = _extract_running_losses(ckpt)
    if not running_losses:
        raise RuntimeError(f"No `running_losses` found in `{ckpt_path}`")

    save_dir = os.path.dirname(save_path) or "."
    mpl_cfg = os.path.join(save_dir, ".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    os.environ.setdefault("XDG_CACHE_HOME", mpl_cfg)
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(running_losses, linewidth=0.8, label="loss")

    if rolling and rolling > 1:
        import numpy as np

        x = np.asarray(running_losses, dtype=np.float32)
        kernel = np.ones(int(rolling), dtype=np.float32) / float(rolling)
        smoothed = np.convolve(x, kernel, mode="valid")
        ax.plot(
            range(len(smoothed)),
            smoothed,
            linewidth=1.5,
            label=f"rolling_mean({rolling})",
        )

    dataset = str(ckpt.get("cfg", {}).get("data", {}).get("dataset", "dataset"))
    ax.set_title(f"Running Loss [{dataset}]")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[plot] saved `{save_path}`")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot running losses from an NCSN checkpoint."
    )
    parser.add_argument("--out_dir", type=str, default="out/pytorch")
    parser.add_argument(
        "--ckpt", type=str, default="latest", help="Path, directory, or 'latest'."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output image path (default: out_dir/loss_curve_ckpt.png).",
    )
    parser.add_argument(
        "--show", action="store_true", help="Call plt.show() after saving."
    )
    parser.add_argument(
        "--rolling", type=int, default=0, help="Optional rolling mean window."
    )
    args = parser.parse_args()

    ckpt_path = _resolve_ckpt_path(args.ckpt, out_dir=args.out_dir)
    save_path = args.save or os.path.join(args.out_dir, "loss_curve_ckpt.png")
    plot_running_losses(
        ckpt_path, save_path=save_path, show=bool(args.show), rolling=int(args.rolling)
    )


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
