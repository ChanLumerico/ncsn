import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
from torchvision.utils import make_grid
from tqdm.auto import tqdm


def _ensure_src_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.dirname(here)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _load_state_dict_into(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
    else:
        state = obj
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def _sample_with_intermediate_frames(
    model: torch.nn.Module,
    sigmas: torch.Tensor,
    n_samples: int,
    image_size: int,
    in_channels: int,
    n_steps_each: int,
    step_lr: float,
    device: torch.device,
    clamp: bool,
    init_distribution: str,
    denoise: bool,
    num_frames: int,
) -> List[torch.Tensor]:
    model.eval()

    if init_distribution not in ("uniform", "normal"):
        raise ValueError("init_distribution must be one of: 'uniform', 'normal'")

    total_steps = int(sigmas.numel()) * int(n_steps_each)
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    if num_frames >= total_steps:
        capture_steps = list(range(total_steps))
    elif num_frames == 1:
        capture_steps = [total_steps - 1]
    else:
        capture_steps = []
        for k in range(num_frames):
            idx = int(round(k * (total_steps - 1) / float(num_frames - 1)))
            capture_steps.append(idx)
        capture_steps = sorted(set(capture_steps))

    if init_distribution == "uniform":
        x = torch.empty(
            n_samples, in_channels, image_size, image_size, device=device
        ).uniform_(-1.0, 1.0)
    else:
        x = torch.randn(n_samples, in_channels, image_size, image_size, device=device)

    frames: List[torch.Tensor] = []
    capture_set = set(capture_steps)

    pbar = tqdm(total=total_steps, desc="Sampling", dynamic_ncols=True)
    global_step = 0
    try:
        for i, sigma in enumerate(sigmas):
            pbar.set_postfix(sigma=f"{sigma.item():.4f}")
            labels = torch.full((n_samples,), i, device=device, dtype=torch.long)
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for _ in range(int(n_steps_each)):
                grad = model(x, labels)
                noise = torch.randn_like(x)
                x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise
                if clamp:
                    x = x.clamp(-1.0, 1.0)
                if global_step in capture_set:
                    frames.append(x.detach().clone())
                global_step += 1
                pbar.update(1)
    finally:
        pbar.close()

    if denoise:
        last_label = torch.full(
            (n_samples,), sigmas.shape[0] - 1, device=device, dtype=torch.long
        )
        x = x + (sigmas[-1] ** 2) * model(x, last_label)
        if clamp:
            x = x.clamp(-1.0, 1.0)
        if frames:
            frames[-1] = x.detach().clone()
        else:
            frames.append(x.detach().clone())

    return frames


def _tensor_grid_to_uint8(frame: torch.Tensor, nrow: int) -> Tuple[torch.Tensor, Any]:
    grid = make_grid(frame, nrow=nrow, padding=2, pad_value=0.0)
    grid = (grid.clamp(-1.0, 1.0) + 1.0) / 2.0
    grid = (grid * 255.0).round().to(torch.uint8)
    return grid


def _grid_to_numpy_uint8(grid: torch.Tensor) -> torch.Tensor:
    if grid.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(grid.shape)}")
    c, h, w = grid.shape
    if c == 1:
        grid = grid.repeat(3, 1, 1)
    elif c != 3:
        grid = grid[:3]
    return grid.permute(1, 2, 0).contiguous().cpu().numpy()


def _default_config_path(dataset: str) -> str:
    return os.path.join(os.path.dirname(__file__), "configs", f"{dataset}.yml")


def main() -> None:
    _ensure_src_on_path()

    from src.model import NCSN, make_sigmas
    from src.utils import ensure_dir, load_config, seed_all

    parser = argparse.ArgumentParser(
        description="Create a sampling-trajectory GIF for a trained NCSN."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (used for config/model naming).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML path (default: src/pytorch/configs/{dataset}.yml)",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out_dir", type=str, default="out/pytorch")
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed (default: config `seed` or 42).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of intermediate frames to capture across the full sampling trajectory.",
    )
    parser.add_argument(
        "--n_steps_each",
        type=int,
        default=None,
        help="Override sampling.n_steps_each for GIF generation.",
    )
    parser.add_argument(
        "--step_lr",
        type=float,
        default=None,
        help="Override sampling.step_lr for GIF generation.",
    )

    args = parser.parse_args()

    dataset = str(args.dataset)
    config_path = args.config or _default_config_path(dataset)
    cfg: Dict[str, Any] = load_config(config_path)

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    seed_all(seed)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device(cfg.get("device", "cpu"))
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        device = torch.device("cpu")

    out_dir = str(args.out_dir)
    models_dir = str(args.models_dir or os.path.join(out_dir, "models"))
    fig_dir = str(args.fig_dir or os.path.join(out_dir, "fig"))
    ensure_dir(fig_dir)

    ckpt_path = os.path.join(models_dir, f"ncsn_{dataset}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: `{ckpt_path}`")

    image_size = int(cfg.get("data", {}).get("image_size", 32))
    in_channels = int(cfg.get("model", {}).get("in_channels", 3))

    num_scales = int(cfg.get("model", {}).get("num_scales", 10))
    sigmas = make_sigmas(
        sigma_begin=float(cfg.get("model", {}).get("sigma_begin", 1.0)),
        sigma_end=float(cfg.get("model", {}).get("sigma_end", 0.01)),
        num_scales=num_scales,
    ).to(device)

    model = NCSN(
        in_channels=in_channels,
        nf=int(cfg.get("model", {}).get("nf", 128)),
        num_classes=num_scales,
        dilations=tuple(cfg.get("model", {}).get("dilations", (1, 2, 4, 8))),
        scale_by_sigma=bool(cfg.get("model", {}).get("scale_by_sigma", True)),
    ).to(device)
    model.set_sigmas(sigmas)
    _load_state_dict_into(model, ckpt_path, device=device)

    sampling_cfg = cfg.get("sampling", {}) or {}
    n_steps_each = (
        int(args.n_steps_each)
        if args.n_steps_each is not None
        else int(sampling_cfg.get("n_steps_each", 100))
    )
    step_lr = (
        float(args.step_lr)
        if args.step_lr is not None
        else float(sampling_cfg.get("step_lr", 2e-5))
    )
    clamp = bool(sampling_cfg.get("clamp", True))
    denoise = bool(sampling_cfg.get("denoise", False))
    init_distribution = str(sampling_cfg.get("init_distribution", "uniform"))

    frames = _sample_with_intermediate_frames(
        model=model,
        sigmas=sigmas,
        n_samples=int(args.n_samples),
        image_size=image_size,
        in_channels=in_channels,
        n_steps_each=n_steps_each,
        step_lr=step_lr,
        device=device,
        clamp=clamp,
        init_distribution=init_distribution,
        denoise=denoise,
        num_frames=int(args.frames),
    )

    images = []
    for f in frames:
        grid = _tensor_grid_to_uint8(f, nrow=int(args.nrow))
        images.append(_grid_to_numpy_uint8(grid))

    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio

    out_path = os.path.join(fig_dir, f"{dataset}_sample.gif")
    imageio.mimsave(out_path, images, fps=int(args.fps))
    print(f"[gif] saved `{out_path}` ({len(images)} frames)")


if __name__ == "__main__":
    main()
