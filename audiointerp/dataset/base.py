from abc import ABC, abstractmethod
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F


class BaseAudioDataset(Dataset, ABC):
    """Abstract base class for audio datasets"""
    def __init__(self, root_dir, sr=None, duration=None, normalize=None, feature_extractor=None, time_augs=None, feature_augs=None):
        """
        root_dir: path to dataset directory
        sr: sampling rate (default - None == each file keeps it's original sr)
        duration: duration of audios in seconds (default - None == keep original duration)
        normalize: type of audio normalization (currently only peak norm is supported) (default - None)
        feature_extractor: a transform object which performs feature extraction 
            (default - None for e2e systems or when feature extraction is performed elsewhere)
        time_augs: a transofrm or a sequence of transforms applied to wavs during training (default - None)
        feature_augs: a transofrm or a sequence of transforms applied to features during training (default - None)
        """

        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        if sr is not None and duration is not None:
            self.num_frames = int(sr * duration)
        else:
            self.num_frames = None
        self.normalize = normalize
        self.feature_extractor = feature_extractor
        self.time_augs = time_augs
        self.feature_augs = feature_augs
        self.metadata = self.load_metadata()

    @abstractmethod
    def load_metadata(self):
        """Loads data from corresponding metafile and converts it to format path_to_audio, target"""
        pass

    def _fix_length(self, audio):
        if self.num_frames is not None:
            if audio.shape[1] < self.num_frames:
                pad_frames = self.num_frames - audio.shape[1]
                return torch.nn.functional.pad(audio, (0, pad_frames))
            elif audio.shape[1] > self.num_frames:
                return audio[..., :self.num_frames]
        return audio
            

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        audio_path = self.metadata.loc[idx, "path_to_audio"]
        target = torch.tensor(self.metadata.loc[idx, "target"], dtype=torch.long)

        audio, original_sr = torchaudio.load(audio_path)

        # convert to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # resample if necessary
        if original_sr != self.sr:
            audio = F.resample(audio, original_sr, self.sr)

        # normalization
        if self.normalize is not None:
            abs_max = audio.abs().max()
            if abs_max != 0.:
                audio /= abs_max

        if self.time_augs is not None:
            audio = self.time_augs(audio)

        audio = self._fix_length(audio)

        if self.feature_extractor is not None:
            features = self.feature_extractor(audio)
        else:
            features = audio

        if self.feature_augs is not None:
            features = self.feature_augs(features)

        return features, target