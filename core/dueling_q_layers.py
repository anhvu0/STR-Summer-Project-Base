import os

import torch
from torch import nn


class DuelingDQN(nn.Module):
    """
    PyTorch dueling DQN used by training and inference.
    """

    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_size, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )

    def forward(self, states):
        trunk = self.trunk(states)
        value = self.value_stream(trunk)
        advantage = self.advantage_stream(trunk)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_dueling_dqn(state_size, action_size, device=None):
    model = DuelingDQN(state_size, action_size)
    if device is not None:
        model = model.to(device)
    return model


def save_torch_checkpoint(path, model, optimizer=None, **metadata):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "format": "str-pytorch-dueling-dqn-v1",
        "state_size": int(model.state_size),
        "action_size": int(model.action_size),
        "model_state_dict": model.state_dict(),
        **metadata,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_torch_checkpoint(path, device=None):
    if str(path).lower().endswith(".h5"):
        raise ValueError(
            f"{path} is a legacy Keras/TensorFlow .h5 checkpoint. "
            "Retrain or provide a PyTorch .pt checkpoint."
        )
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a STR PyTorch checkpoint.")
    model = build_dueling_dqn(
        checkpoint["state_size"],
        checkpoint["action_size"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
