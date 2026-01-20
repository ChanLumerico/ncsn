import os
from glob import glob
from typing import Callable, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 32,
        random_horizontal_flip: bool = True,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
        transform: Optional[Callable] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.root = root
        self.exts = tuple(e.lower() for e in exts)

        paths: List[str] = []
        for ext in self.exts:
            paths.extend(glob(os.path.join(root, f"*{ext}")))
            paths.extend(glob(os.path.join(root, f"*{ext.upper()}")))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(
                f"No images found under `{root}` with extensions: {', '.join(self.exts)}"
            )
        if limit is not None:
            paths = paths[: int(limit)]
        self.paths = paths

        if transform is not None:
            self.transform = transform
        else:
            tfs = [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
            ]
            if random_horizontal_flip:
                tfs.append(transforms.RandomHorizontalFlip())
            tfs.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.transform = transforms.Compose(tfs)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        return x
