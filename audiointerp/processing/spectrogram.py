import torch
import torchaudio.transforms as T_audio
from .baseprocessor import BaseProcessor



def plot_spec(spec,is_mel=True, sample_rate: int = 16000, fmin=0, fmax=8000, cmap="magma"):
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    if spec.ndim == 3:
        spec = spec[0, :, :]
    elif spec.ndim != 2:
        raise ValueError("Spec format is wrong")

    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()

    n_bands, n_frames = spec.shape
    
    if is_mel:
        freqs = librosa.mel_frequencies(n_mels=n_bands, fmin=fmin, fmax=fmax)
    else:
        freqs = np.linspace(0, sample_rate / 2, n_bands)


    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect="auto", origin="lower", cmap=cmap)
    yt = np.linspace(0, n_bands-1, 6, dtype=int)
    plt.yticks(yt, [f"{round(freqs[i])}" for i in yt])
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time frame")
    plt.title("Mel Spectrogram" if is_mel else "STFT Spectrogram")
    plt.colorbar(label="Energy (dB)")
    plt.tight_layout()
    plt.show()

    

def _db_to_linear(spec_db, stype):
    if stype == "power":
        return 10.0 ** (spec_db * 0.1)
    elif stype == "magnitude":
        return 10.0 ** (spec_db * 0.05)
    else:
        raise ValueError(f"Unknown stype: {stype}")


class _LogLosslessInvMixin:
    def __init__(self, return_pre_db=False):
        self.return_pre_db = return_pre_db


class STFTSpectrogram(BaseProcessor):
    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window, power=2.,
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

        

class LogSTFTSpectrogram(STFTSpectrogram, _LogLosslessInvMixin):
    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window, power=2.,
                 wkwargs=None, center=True, pad_mode='reflect',
                 top_db=None, return_phase=True, return_full_db=True, return_pre_db=False):
        STFTSpectrogram.__init__(
            self, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, window_fn=window_fn, power=power, wkwargs=wkwargs,
            center=center, pad_mode=pad_mode, return_phase=return_phase
        )

        self.return_full_db = return_full_db

        self.stype = None
        if power == 2.:
            self.stype = "power"
        elif power == 1.:
            self.stype = "magnitude"

        self.amp_to_db = T_audio.AmplitudeToDB(
            stype=self.stype, top_db=top_db
        )

        self.amp_to_db_full = T_audio.AmplitudeToDB(
            stype=self.stype, top_db=None
        )

        _LogLosslessInvMixin.__init__(self, return_pre_db=return_pre_db)

    def forward(self, wav):
        
        if self.return_phase:
            spec, phase = super().forward(wav)
        else:
            spec = super().forward(wav)

        spec_db = self.amp_to_db(spec)

        out = [spec_db]

        if self.return_phase:
            out.append(phase)
        if self.return_full_db:
            full_db = self.amp_to_db_full(spec)
            out.append(full_db)
        if self.return_pre_db:
            out.append(spec)

        return tuple(out) if len(out) > 1 else out[0]

    def inverse(self, features, phase=None, full_db=None, pre_db=None):
        if pre_db is not None:
            linear_spec = pre_db
        elif full_db is not None:
            linear_spec = _db_to_linear(full_db, self.stype)
        else:
            print("Warning: non-clipped db values are not provided. The inversion will be lossy.")
            linear_spec = _db_to_linear(features, self.stype)

        wav = super().inverse(linear_spec, phase=phase)

        return wav



class MelSTFTSpectrogram(STFTSpectrogram):
    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window, power=2.,
                 wkwargs=None, center=True, pad_mode='reflect',
                 sample_rate=16000, n_mels=80, f_min=0.0, f_max=None, return_phase=True):
        
        super().__init__(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, window_fn=window_fn, power=power, wkwargs=wkwargs,
            center=center, pad_mode=pad_mode, return_phase=return_phase
        )

        n_stft = (n_fft // 2) + 1
        if f_max is None:
            f_max = sample_rate / 2.0

        self.mel_scale = T_audio.MelScale(
            n_mels=n_mels, sample_rate=sample_rate,
            f_min=f_min, f_max=f_max, n_stft=n_stft
        )

        self.inverse_mel_scale = T_audio.InverseMelScale(
            n_stft=n_stft, n_mels=n_mels, sample_rate=sample_rate,
            f_min=f_min, f_max=f_max
        )


    def forward(self, wav):
        if self.return_phase:
            linear_spec, phase = super().forward(wav)
        else:
            linear_spec = super().forward(wav)

        mel_spec = self.mel_scale(linear_spec)

        if self.return_phase:
            return mel_spec, phase
        else:
            return mel_spec

    def inverse(self, features, phase=None):
        print("Warning: melscale inversion is approximate")
        linear_spec_reconstructed = self.inverse_mel_scale(features)

        wav = super().inverse(linear_spec_reconstructed, phase=phase)

        return wav


class LogMelSTFTSpectrogram(MelSTFTSpectrogram, _LogLosslessInvMixin):
    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window, power=2.,
                 wkwargs=None, center=True, pad_mode='reflect',
                 sample_rate=16000, n_mels=80, f_min=0.0, f_max=None,
                 top_db=None, return_full_db=True, return_phase=True, return_pre_db=False):
        MelSTFTSpectrogram.__init__(
            self, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, window_fn=window_fn, power=power, wkwargs=wkwargs,
            sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_mels=n_mels,
            center=center, pad_mode=pad_mode, return_phase=return_phase
        )

        self.return_full_db = return_full_db

        self.stype = None
        if power == 2.:
            self.stype = "power"
        elif power == 1.:
            self.stype = "magnitude"

        self.amp_to_db = T_audio.AmplitudeToDB(
            stype=self.stype, top_db=top_db
        )

        self.amp_to_db_full = T_audio.AmplitudeToDB(
            stype=self.stype, top_db=None
        )

        _LogLosslessInvMixin.__init__(self, return_pre_db=return_pre_db)

    def forward(self, wav):
        if self.return_phase:
            mel_spec, phase = super().forward(wav)
        else:
            mel_spec = super().forward(wav)

        mel_spec_db = self.amp_to_db(mel_spec)

        out = [mel_spec_db]

        if self.return_phase:
            out.append(phase)
        if self.return_full_db:
            full_db = self.amp_to_db_full(mel_spec)
            out.append(full_db)
        if self.return_pre_db:
            out.append(mel_spec)

        return tuple(out) if len(out) > 1 else out[0]

    def inverse(self, features, phase=None, full_db=None, pre_db=None):
        if pre_db is not None:
            linear_mel_spec = pre_db
        elif full_db is not None:
            linear_mel_spec = _db_to_linear(full_db, self.stype)
        else:
            print("Warning: non-clipped db values are not provided. The inversion will be lossy.")
            linear_mel_spec = _db_to_linear(features, self.stype)

        wav = super().inverse(linear_mel_spec, phase=phase)

        return wav