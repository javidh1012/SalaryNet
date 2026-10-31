import torch
from torch import nn


class SalaryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu(self.hidden(inputs)))
