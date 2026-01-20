import os
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTTensorDataset(Dataset):
    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        image_size: int = 32,
        transform: Optional[Callable] = None,
    ) -> None:
        tv_root = root
        if os.path.basename(os.path.normpath(root)).lower() == "mnist":
            tv_root = os.path.dirname(os.path.normpath(root)) or "."

        expected = os.path.join(tv_root, "MNIST", "raw")
        if not os.path.isdir(expected):
            raise FileNotFoundError(
                f"MNIST not found at `{expected}` (set `data.root` to your data parent folder)"
            )

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        image_size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        self.ds = datasets.MNIST(
            root=tv_root,
            train=train,
            download=False,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x, _y = self.ds[idx]
        return x
