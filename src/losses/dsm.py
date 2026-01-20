from typing import Tuple

import torch


def annealed_dsm_loss(
    model, x: torch.Tensor, sigmas: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sigmas.dim() != 1:
        raise ValueError("sigmas must be a 1-D tensor.")

    batch = x.shape[0]
    device = x.device
    labels = torch.randint(
        0, sigmas.shape[0], (batch,), device=device, dtype=torch.long
    )
    used_sigmas = sigmas[labels].view(batch, 1, 1, 1)

    noise = torch.randn_like(x)
    perturbed = x + used_sigmas * noise
    score = model(perturbed, labels)

    loss = torch.mean(torch.sum((score * used_sigmas + noise) ** 2, dim=(1, 2, 3)))
    return loss, labels
