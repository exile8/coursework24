import torch
import torchaudio.transforms as T_audio
from .baseprocessor import BaseProcessor


class STFTSpectrogram(BaseProcessor):
    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window, power=None,
                 wkwargs=None, center=True, pad_mode='reflect',
                 return_phase=True):
        super().__init__(return_phase)
        
        self.stft = T_audio.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, window_fn=window_fn, power=None,
            normalized=False, wkwargs=wkwargs, center=center,
            pad_mode=pad_mode, onesided=True
        )

        self.stft_inverse = T_audio.InverseSpectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, window_fn=window_fn, normalized=False,
            wkwargs=wkwargs, center=center, pad_mode=pad_mode, onesided=True
        )

        self.griffinlim = T_audio.GriffinLim(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            window_fn=window_fn, power=power, wkwargs=wkwargs
        )

        self.power = power

    def forward(self, wav):
        stft_complex = self.stft(wav)

        if self.power is not None:
            if self.power == 1.:
                features = stft_complex.abs()
            else:
                features = stft_complex.abs().pow(self.power)
        else:
            features = stft_complex

        if self.return_phase:
            phase = self.extract_phase(stft_complex)
            return features, phase

        return features 

    def inverse(self, features, phase=None):

        if phase is None:
            print("Warning: no phase is provided. The inversion is performed with GriffinLim")
            wav = self.griffinlim(features)
            return wav
        
        if self.power is not None:
            if self.power == 1.:
                magnitude = features
            else:
                magnitude = features.pow(1./self.power)
            stft_complex = self.apply_phase(magnitude, phase)
        else:
            stft_complex = features

        wav = self.stft_inverse(stft_complex)

        return wav

        


        


class LogSTFTSpectrogram:
    def __init__(self):
        pass

    def forward(self):
        pass

    def inverse(self):
        pass


class MelSTFTSpectrogram:
    def __init__(self):
        pass

    def forward(self):
        pass

    def inverse(self):
        pass


class LogMelSTFTSpectrogram:
    def __init__(self):
        pass

    def forward(self):
        pass

    def inverse(self):
        pass