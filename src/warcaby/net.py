"""The policy-value network.

One network, two answers about a position:

* **policy** - for every possible move, how promising it looks (a vector of
  logits, one per action index),
* **value** - who is winning, as a number from -1 (the player to move loses)
  to +1 (the player to move wins).

The body is a small ResNet: a stack of 3x3 convolutions with skip connections.
Convolutions suit a board because a pattern - "a man with an empty square behind
it" - means the same thing wherever it appears. The skip connections let each
block learn a correction to what it receives, which keeps deeper stacks
trainable.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .game import Game


@dataclass
class NetConfig:
    blocks: int = 6
    channels: int = 64
    policy_channels: int = 8
    value_hidden: int = 64


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_size: int,
        config: NetConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NetConfig()
        self.observation_shape = tuple(observation_shape)
        self.action_size = action_size
        planes, height, width = self.observation_shape
        cfg = self.config

        self.stem = nn.Sequential(
            nn.Conv2d(planes, cfg.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*(ResidualBlock(cfg.channels) for _ in range(cfg.blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.policy_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cfg.policy_channels * height * width, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(cfg.channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(height * width, cfg.value_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(policy logits [batch, actions], value [batch])``."""
        features = self.body(self.stem(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    @classmethod
    def for_game(cls, game: Game, config: NetConfig | None = None) -> PolicyValueNet:
        return cls(game.observation_shape, game.action_size, config)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- saving ----------------------------------------------------------------------

    def export(self) -> dict:
        """Everything needed to rebuild this network, with weights on the CPU."""
        return {
            "config": asdict(self.config),
            "observation_shape": self.observation_shape,
            "action_size": self.action_size,
            "weights": {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()},
        }

    @classmethod
    def restore(cls, exported: dict) -> PolicyValueNet:
        net = cls(
            tuple(exported["observation_shape"]),
            exported["action_size"],
            NetConfig(**exported["config"]),
        )
        net.load_state_dict(exported["weights"])
        return net


def export_to_bytes(exported: dict) -> bytes:
    """An exported network as bytes, for handing to another process."""
    buffer = io.BytesIO()
    torch.save(exported, buffer)
    return buffer.getvalue()


def export_from_bytes(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def pick_device(preferred: str | None = None) -> torch.device:
    """``preferred`` if given, otherwise CUDA, then Apple's MPS, then the CPU."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
