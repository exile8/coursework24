# coursework24
Coursework 2024-2025 Master's Program (1st Year)

This repository contains code, notebooks and data for the course project **Constructing post-hoc interpretations for audio classification models** and the paper based on this project.

## 1. Layout
```
.
├── README.md
├── requirements.txt
├── audiointerp    # Python package for running interpretation experiments 
│   ├── dataset    # Dataset processing
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── esc50.py
│   └── processing    # Feature extraction and inversion
│       ├── __init__.py
│       ├── baseprocessor.py
│       └── spectrogram.py
│   ├── interpretation    # Interpretation methods
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gradcam.py
│   │   ├── lime.py
│   │   ├── saliency.py
│   │   └── shap.py
│   ├── model    # Model
│   │   ├── __init__.py
│   │   └── cnn14.py
│   ├── metrics.py    # Quality metrric
|   ├── fit.py    # Model traiing, validation and testing
│   ├── predict.py    # Running interpretation experiments and evaluating results
├── samples    # Suggested samples for trying out the setup
│   ├── car_horn.wav
│   ├── cat.wav
│   ├── clapping.wav
│   ├── crow.wav
│   ├── dog.wav
│   └── sea_waves.wav
├── noises    # Suggested samples for background contamination experiments
│   ├── 149024__foxen10__horse_whinny.wav
│   ├── 165058__theundecided__white-noise.wav
│   └── 203297__mzui__room-tone-office-industrial-ambience-01.wav
├── train_model_stft.ipynb    # Fine-tune Cnn14 on ESC-50 log-stft spectrograms
├── train_model_mel.ipynb    # Fine-tune Cnn14 on ESC-50 log-mel-stft spectrograms
├── experiments_stft.ipynb    # Run experiments on clean data for stft model
├── experiments_mel.ipynb    # Run experiments on clean data for mel model
├── experiments_mel_noise1.ipynb    # Run experiments on data contaminated with white noise for mel model
├── experiments_mel_noise2.ipynb    # Run experiments on data contaminated with industrial room ambience for mel model
├── experiments_mel_noise3.ipynb    # Run experiments on data contaminated with horse sounds for mel model
├── figure.png    # A figure illustrating interpretatios obtained with different methods
├── illustration.ipynb    # Obtain illustrations
├── predictions_illust    # Examples of visual and audio interpretations
│   ├── mel
│   └── stft
├── tables.ipynb    # Full set of tables containing experimental results for the project and the paper
├── tables.pdf    # The same set of tables in pdf format
```

### Correspondence between audio files in this repository and ESC-50 dataset

The `samples` directory contains 5 audio files sourced from the ESC-50 dataset [2]. The following table maps the filenames used in this repository to their corresponding filenames in the original dataset.

| File in the `samples` directory   | File in the ESC-50 dataset   |
| --------------------------------- | ---------------------------- |
|  car_horn.wav                     |  2-100648-A-43.wav           |
|  clapping.wav                     |  2-76408-D-22.wav            |
|  crow.wav                         |  1-56234-A-9.wav             |
|  dog.wav                          |  3-157695-A-0.wav            |
|  sea_waves.wav                    |  4-182613-A-11.wav           |
|  cat.wav                          |  5-177614-A-5.wav            |

Please refer to the [ESC-50 official repository](https://github.com/karolpiczak/ESC-50) for more details about the dataset.

The `noises` directory contains 5sec fragments of clips from Freesound.org ([4], [5], [6]).

## 2. Methodology - Mask and Reconstruct

Our workflow starts by running a model (Cnn14 [3]) on a spectrogram and computing a post-hoc attribution map with one of four generic explainers (Saliency, Grad-CAM, LIME or SHAP). The attribution scores are then converted into a binary or soft mask; we apply this mask to the spectrogram, invert the masked representation back into the time domain, and evaluate the result with a suite of fidelity (FF, AI, AD, AG, Fid-in) and complexity (SPS, COMP) metrics.

This “mask-and-reconstruct” loop lets us analyse each explanation both visually (masked spectrogram) and audibly (resynthesised waveform), revealing how much of the classifier’s confidence is carried by the retained time–frequency regions. The procedure is inspired by the LMAC framework of Paissan et al. (2024) [1]: we adopt their core idea—measuring explanation quality through spectrogram masking and audio reconstruction, but generalise it to standard attribution methods, several mask-generation strategies and noise scenarios.

## 3. Quick start

```bash
git clone https://github.com/exile8/coursework24.git
cd coursework24
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # torch, torchaudio, librosa, lime, shap …

# Run a notebook
jupyter lab experiments_mel.ipynb
```

## 4. Example

```Python
import torch, torchaudio
from audiointerp.interpretation.gradcam import GradCAMInterpreter
from audiointerp.processing.spectrogram import LogMelSTFTSpectrogram

# load audio
wav, sr = torchaudio.load("samples/dog.wav")

# feature extractor (Mel, dB)
fe = LogMelSTFTSpectrogram(return_phase=False, return_full_db=False)
spec_db = fe(wav.unsqueeze(0))[0]          # (1, F, T)

# pretrained model
model = torch.load("weights/cnn14_mel.pth")

# Grad-CAM heat-map
interp = GradCAMInterpreter(model, target_layers=["cnn.bn6"])
heatmap = interp.interpret(spec_db)        # (1, F, T)
```

## References
[1] F. Paissan, M. Ravanelli, C. Subakan, Listenable maps for audio classifiers, International Conference on Machine Learning (ICML), 2024. [DOI: ]

[2] K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015. [DOI: 10.1145/2733373.2806390]

[3] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894. [DOI: 10.1109/TASLP.2020.3030497]

[4] White noise by theundecided -- https://freesound.org/s/165058/ -- License: Creative Commons 0

[5] Room Tone Office Industrial Ambience 01 by mzui -- https://freesound.org/s/203297/ -- License: Creative Commons 0

[6] Horse_Whinny.wav by foxen10 -- https://freesound.org/s/149024/ -- License: Creative Commons 0