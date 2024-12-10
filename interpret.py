"""
Functionality for applying interpretation
methods and evaluating their quality.
"""

import torchaudio
import torchaudio.functional as F
import torch

def load_audio(filename, sampling_rate=None):
    audio, sr_source = torchaudio.load(filename)

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0)

    if sampling_rate is not None and sr_source != sampling_rate:
        audio = F.resample(audio, sr_source, sampling_rate)

    return audio

def interpret_audio(filename, sampling_rate=None):
    audio = load_audio(filename, sampling_rate)
    return audio

def evaluate_interpretation():
    pass