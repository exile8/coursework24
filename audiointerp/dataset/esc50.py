import os
import pandas as pd
from .base import BaseAudioDataset


class ESC50dataset(BaseAudioDataset):
    """Class to work with ESC-50 dataset"""
    def __init__(self, root_dir, folds=None, sr=44100, duration=5.0, normalize=None, feature_extractor=None, time_augs=None, feature_augs=None):
        """
        root_dir: path to dataset directory
        folds (int, list, tuple): folds that will be used in the current subset
        sr: sampling rate (default - None == each file keeps it's original sr)
        duration: duration of audios in seconds (default - None == keep original duration)
        normalize: type of audio normalization (currently only peak norm is supported) (default - None)
        feature_extractor: a transform object which performs feature extraction 
            (default - None for e2e systems or when feature extraction is performed elsewhere)
        augmentations: a transofrm or a sequence of transforms used during training (default - None)
        """
        self.folds = self.validate_folds(folds)

        super().__init__(root_dir, sr, duration, normalize, feature_extractor, time_augs, feature_augs)

    def validate_folds(self, folds):
        if folds is None:
            return [1, 2, 3, 4, 5]
        elif isinstance(folds, int):
             return [folds]
        elif isinstance(folds, tuple) or isinstance(folds, list):
                if set(folds).issubset({1, 2, 3, 4, 5}):
                     return list(folds)
                else:
                     raise ValueError(f"Invalid folds {folds}. ESC-50 has folds 1, 2, 3, 4, 5")
        else:
            raise TypeError(f"folds must be int, list or tuple")

    def load_metadata(self):
        meta_path = os.path.join(self.root_dir, "meta", "esc50.csv")
        meta = pd.read_csv(meta_path)

        # leave necessary folds
        meta = meta[meta["fold"].isin(self.folds)]
        meta = meta[["filename", "target"]]

        # turn filenames into paths
        meta.loc[:, "filename"] = meta["filename"].apply(lambda fn: os.path.join(self.root_dir, "audio", fn))

        # rename columns
        metadata = meta.rename(columns={"filename": "path_to_audio"})

        return metadata