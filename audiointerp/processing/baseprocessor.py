import torch
from abc import ABC, abstractmethod


class BaseProcessor(torch.nn.Module, ABC):
    def __init__(self, return_phase=True):
        super().__init__()
        self.return_phase = return_phase

    @abstractmethod
    def forward(self, wav):
        pass

    @abstractmethod
    def inverse(self, features, phase=None):
        pass

    def extract_phase(self, stft_complex):
        return torch.atan2(stft_complex.imag, stft_complex.real)

    def apply_phase(self, magnitude, phase):
        return torch.polar(magnitude, phase)
