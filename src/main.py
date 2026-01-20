import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.losses import annealed_dsm_loss
from src.model import NCSN, make_sigmas
from src.runners import annealed_langevin_dynamics
from src.utils import AverageMeter, ensure_dir, load_config, seed_all

import importlib


def _import_object(path: str):
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path: {path}")

    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _build_dataset(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {}) or {}
    cls_path = data_cfg.get("dataset_cls", None)
    if not cls_path:
        raise ValueError("`data.dataset_cls` is required")
    ds_cls = _import_object(str(cls_path))

    kwargs = dict(data_cfg.get("dataset_kwargs", {}) or {})

    if "root" in data_cfg and "root" not in kwargs:
        kwargs["root"] = data_cfg["root"]
    if "image_size" in data_cfg and "image_size" not in kwargs:
        kwargs["image_size"] = int(data_cfg["image_size"])
    if "train" in data_cfg and "train" not in kwargs:
        kwargs["train"] = bool(data_cfg["train"])

    return ds_cls(**kwargs)


def _print_model_summary(
    model: torch.nn.Module,
    image_size: int,
    in_channels: int,
    device: torch.device,
    depth: int = 3,
) -> None:
    try:
        from torchinfo import summary

        model_was_training = model.training
        model.eval()
        x = torch.zeros(1, in_channels, image_size, image_size, device=device)
        labels = torch.zeros(1, dtype=torch.long, device=device)

        print(summary(model, input_data=(x, labels), depth=depth, verbose=0))
        model.train(model_was_training)
        return

    except Exception as e:
        print(
            f"[warn] torchinfo.summary failed ({type(e).__name__}: {e}); printing fallback."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"[model] params: {total_params:,} (trainable: {trainable_params:,})")


def _get_device(cfg: Dict[str, Any], override: Optional[str] = None) -> torch.device:
    if override:
        return torch.device(override)
    dev = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    if dev == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        dev = "cpu"
    return torch.device(dev)


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Dict[str, Any],
    running_losses: Optional[List[float]] = None,
) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
            "running_losses": running_losses if running_losses is not None else [],
        },
        path,
    )


def _resolve_ckpt_path(resume_arg: Optional[str], out_dir: str) -> Optional[str]:
    if not resume_arg:
        return None

    arg = str(resume_arg)
    lower = arg.lower()

    if lower in ("latest", "last", "ckpt_latest"):
        path = os.path.join(out_dir, "checkpoints", "ckpt_latest.pt")

    elif os.path.isdir(arg):
        # If a directory is provided, assume it is either the checkpoints directory
        # or the project output directory.
        candidate = os.path.join(arg, "ckpt_latest.pt")
        if os.path.exists(candidate):
            path = candidate
        else:
            path = os.path.join(arg, "checkpoints", "ckpt_latest.pt")

    else:
        path = arg

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: `{path}`")

    return path


def _load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)

    except Exception as e:
        ckpt_cfg = ckpt.get("cfg", {})
        model_cfg = ckpt_cfg.get("model", {})
        raise RuntimeError(
            "Failed to load checkpoint model weights. "
            "This usually means your current model config (e.g. nf/num_scales) "
            "doesn't match the checkpoint.\n"
            f"- checkpoint: `{path}`\n"
            f"- checkpoint model cfg: {model_cfg}\n"
            f"- error: {type(e).__name__}: {e}"
        ) from e

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt


@torch.no_grad()
def _save_samples_grid(samples: torch.Tensor, out_path: str, nrow: int = 8) -> None:
    ensure_dir(os.path.dirname(out_path))

    img = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(img, out_path, nrow=nrow)


def train(
    cfg: Dict[str, Any], device: torch.device, resume_ckpt: Optional[str] = None
) -> List[float]:
    seed = int(cfg.get("seed", 42))
    seed_all(seed)

    out_dir = str(cfg["training"].get("out_dir", "out"))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "gens")

    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    auto_resume = bool(cfg["training"].get("auto_resume", True))
    if resume_ckpt is None and auto_resume:
        latest = os.path.join(ckpt_dir, "ckpt_latest.pt")
        if os.path.exists(latest):
            resume_ckpt = "latest"

    image_size = int(cfg["data"].get("image_size", 32))
    dataset = _build_dataset(cfg)

    in_channels = int(cfg["model"].get("in_channels", 3))
    try:
        x0 = dataset[0]
        if torch.is_tensor(x0) and x0.dim() >= 3:
            if int(x0.shape[0]) != in_channels:
                raise ValueError(
                    f"Dataset channels ({int(x0.shape[0])}) != model.in_channels ({in_channels})."
                )
    except Exception:
        pass

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    num_scales = int(cfg["model"].get("num_scales", 10))
    sigmas = make_sigmas(
        sigma_begin=float(cfg["model"].get("sigma_begin", 1.0)),
        sigma_end=float(cfg["model"].get("sigma_end", 0.01)),
        num_scales=num_scales,
    ).to(device)

    model = NCSN(
        in_channels=in_channels,
        nf=int(cfg["model"].get("nf", 128)),
        num_classes=num_scales,
        dilations=tuple(cfg["model"].get("dilations", (1, 2, 4, 8))),
        scale_by_sigma=bool(cfg["model"].get("scale_by_sigma", True)),
    ).to(device)
    model.set_sigmas(sigmas)
    _print_model_summary(
        model, image_size=image_size, in_channels=in_channels, device=device, depth=2
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 2e-4)),
        betas=tuple(cfg["training"].get("betas", (0.9, 0.999))),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    start_epoch = 0
    running_losses: List[float] = []
    resume_path = _resolve_ckpt_path(resume_ckpt, out_dir=out_dir)

    if resume_path:
        ckpt = _load_checkpoint(resume_path, model, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0))
        prior_losses = ckpt.get("running_losses", [])

        if isinstance(prior_losses, list):
            running_losses = [float(v) for v in prior_losses]
        else:
            running_losses = [float(v) for v in list(prior_losses)]

        print(
            f"[resume] loaded `{resume_path}` (start_epoch={start_epoch}, "
            f"running_losses={len(running_losses)})"
        )

    else:
        print("[resume] starting fresh")

    epochs = int(cfg["training"].get("epochs", 100))
    grad_clip = cfg["training"].get("grad_clip", None)
    save_every = int(cfg["training"].get("save_every", 1))
    sample_every = int(cfg["training"].get("sample_every", 1))

    n_samples = int(cfg["sampling"].get("n_samples", 64))
    n_steps_each = int(cfg["sampling"].get("n_steps_each", 100))
    step_lr = float(cfg["sampling"].get("step_lr", 2e-5))
    clamp = bool(cfg["sampling"].get("clamp", True))
    denoise = bool(cfg["sampling"].get("denoise", False))

    for epoch in range(start_epoch, epochs):
        model.train()
        meter = AverageMeter()
        pbar = tqdm(
            loader,
            desc=f"Train {epoch + 1}/{epochs}",
            total=len(loader),
            dynamic_ncols=True,
        )

        for x in pbar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            loss, _ = annealed_dsm_loss(model, x, sigmas)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            cur_loss = float(loss.item())
            prev_loss = running_losses[-1] if running_losses else None
            delta_loss = cur_loss - prev_loss if prev_loss is not None else 0.0

            meter = meter.update(cur_loss, n=x.shape[0])
            running_losses.append(cur_loss)
            pbar.set_postfix(
                dloss=f"{delta_loss:+.4f}",
                loss=f"{cur_loss:.4f}",
                avg_loss=f"{meter.avg:.4f}",
            )

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            _save_checkpoint(
                os.path.join(ckpt_dir, "ckpt_latest.pt"),
                model,
                optimizer,
                epoch + 1,
                cfg,
                running_losses=running_losses,
            )
            _save_checkpoint(
                os.path.join(ckpt_dir, f"ckpt_epoch_{epoch + 1:04d}.pt"),
                model,
                optimizer,
                epoch + 1,
                cfg,
                running_losses=running_losses,
            )

        if (epoch + 1) % sample_every == 0 or (epoch + 1) == epochs:
            samples = annealed_langevin_dynamics(
                model=model,
                sigmas=sigmas,
                n_samples=n_samples,
                image_size=image_size,
                in_channels=in_channels,
                n_steps_each=n_steps_each,
                step_lr=step_lr,
                device=device,
                clamp=clamp,
                denoise=denoise,
                init_distribution=str(
                    cfg["sampling"].get("init_distribution", "uniform")
                ),
            )
            _save_samples_grid(
                samples, os.path.join(sample_dir, f"epoch_{epoch + 1:04d}.png")
            )

    # Save a lightweight final model artifact for downstream sampling utilities.
    dataset = str(cfg.get("data", {}).get("dataset", "dataset")).lower()
    models_dir = os.path.join(out_dir, "models")
    ensure_dir(models_dir)
    torch.save(
        {"model_state": model.state_dict(), "cfg": cfg},
        os.path.join(models_dir, f"ncsn_{dataset}.pt"),
    )

    if bool(cfg["training"].get("plot_losses", True)) and running_losses:
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(running_losses, linewidth=1.0)
            ax.set_title("Training Loss (per iteration)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig_path = os.path.join(out_dir, "loss_curve.png")
            fig.savefig(fig_path, dpi=150)

            plt.close(fig)
            print(f"[plot] saved `{fig_path}`")

        except Exception as e:
            print(f"[warn] failed to plot losses ({type(e).__name__}: {e})")

    return running_losses


def sample(cfg: Dict[str, Any], device: torch.device, ckpt: str, out_path: str) -> None:
    num_scales = int(cfg["model"].get("num_scales", 10))
    sigmas = make_sigmas(
        sigma_begin=float(cfg["model"].get("sigma_begin", 1.0)),
        sigma_end=float(cfg["model"].get("sigma_end", 0.01)),
        num_scales=num_scales,
    ).to(device)

    in_channels = int(cfg["model"].get("in_channels", 3))
    model = NCSN(
        in_channels=in_channels,
        nf=int(cfg["model"].get("nf", 128)),
        num_classes=num_scales,
        dilations=tuple(cfg["model"].get("dilations", (1, 2, 4, 8))),
        scale_by_sigma=bool(cfg["model"].get("scale_by_sigma", True)),
    ).to(device)

    _load_checkpoint(ckpt, model, optimizer=None, map_location=device)

    model.set_sigmas(sigmas)
    model.eval()

    samples = annealed_langevin_dynamics(
        model=model,
        sigmas=sigmas,
        n_samples=int(cfg["sampling"].get("n_samples", 64)),
        image_size=int(cfg["data"].get("image_size", 32)),
        in_channels=in_channels,
        n_steps_each=int(cfg["sampling"].get("n_steps_each", 100)),
        step_lr=float(cfg["sampling"].get("step_lr", 2e-5)),
        device=device,
        clamp=bool(cfg["sampling"].get("clamp", True)),
        denoise=bool(cfg["sampling"].get("denoise", False)),
        init_distribution=str(cfg["sampling"].get("init_distribution", "uniform")),
    )
    _save_samples_grid(samples, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Original NCSN - PyTorch")
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        help="Dataset preset name (used to pick default config when --config is not provided).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config YAML (default: configs/{dataset}.yml).",
    )
    parser.add_argument(
        "--mode", type=str, choices=("train", "sample"), default="train"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume training."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint path for sampling."
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output image path for sampling."
    )
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "configs", f"{args.dataset}.yml"
    )
    cfg = load_config(config_path)
    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = str(args.dataset)
    device = _get_device(cfg, override=args.device)

    if args.mode == "train":
        train(cfg, device, resume_ckpt=args.resume)
        return

    if args.ckpt is None:
        raise SystemExit("--ckpt is required for --mode sample")
    out_dir = str(cfg["training"].get("out_dir", "out"))
    ckpt_path = _resolve_ckpt_path(args.ckpt, out_dir=out_dir)

    if ckpt_path is None:
        raise SystemExit("--ckpt is required for --mode sample")

    out_path = args.out or os.path.join(
        out_dir,
        "samples",
        f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    sample(cfg, device, ckpt=ckpt_path, out_path=out_path)


if __name__ == "__main__":
    main()
