"""The ConvNet without skip connections, the simple model compared."""

import itertools

import torch


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class ConvNet(torch.nn.Module):
    """Down and up convolutions with no skip connections, the simple model compared."""

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.down = torch.nn.ModuleList(_conv_block(a, b) for a, b in itertools.pairwise(widths))
        self.pool = torch.nn.MaxPool2d(2)
        reverse = channels[::-1]
        self.up = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Upsample(scale_factor=2), _conv_block(a, b))
            for a, b in itertools.pairwise(reverse)
        )
        self.head = torch.nn.Conv2d(channels[0], 1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.down):
            x = block(x)
            if index < len(self.down) - 1:
                x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        for block in self.up:
            x = block(x)
        return self.head(x)
