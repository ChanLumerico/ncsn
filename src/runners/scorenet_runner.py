from typing import Optional

import torch
from tqdm.auto import tqdm


@torch.no_grad()
def annealed_langevin_dynamics(
    model,
    sigmas: torch.Tensor,
    n_samples: int,
    image_size: int,
    in_channels: int,
    n_steps_each: int,
    step_lr: float,
    device: torch.device,
    clamp: bool = True,
    denoise: bool = False,
    init: Optional[torch.Tensor] = None,
    init_distribution: str = "uniform",
) -> torch.Tensor:
    model.eval()
    sigmas = sigmas.to(device)

    if init is None:
        if init_distribution not in ("uniform", "normal"):
            raise ValueError("init_distribution must be one of: 'uniform', 'normal'")
        if init_distribution == "uniform":
            x = torch.empty(
                n_samples, in_channels, image_size, image_size, device=device
            ).uniform_(-1.0, 1.0)
        else:
            x = torch.randn(
                n_samples, in_channels, image_size, image_size, device=device
            )
    else:
        x = init.to(device)
        if x.shape != (n_samples, in_channels, image_size, image_size):
            raise ValueError(
                f"init has shape {tuple(x.shape)} but expected {(n_samples, in_channels, image_size, image_size)}"
            )

    total = int(sigmas.shape[0]) * int(n_steps_each)
    pbar = tqdm(total=total, desc="Sampling", dynamic_ncols=True)

    for i, sigma in enumerate(sigmas):
        labels = torch.full((n_samples,), i, device=device, dtype=torch.long)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for _ in range(n_steps_each):
            grad = model(x, labels)
            noise = torch.randn_like(x)
            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise
            if clamp:
                x = x.clamp(-1.0, 1.0)
            pbar.update(1)
            pbar.set_postfix(sigma=f"{sigma.item():.4f}")

    if denoise:
        last_label = torch.full(
            (n_samples,), sigmas.shape[0] - 1, device=device, dtype=torch.long
        )
        x = x + (sigmas[-1] ** 2) * model(x, last_label)
        if clamp:
            x = x.clamp(-1.0, 1.0)

    pbar.close()
    return x
