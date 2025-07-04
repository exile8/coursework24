import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from .metrics import Metrics
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd
import gc
from collections import defaultdict


def make_exp_dirs(root, model_type, method_name):
    csv_dir  = os.path.join(root, model_type, method_name, "csvs")
    attr_dir = os.path.join(root, model_type, method_name, "attributions")
    os.makedirs(csv_dir,  exist_ok=True)
    os.makedirs(attr_dir, exist_ok=True)
    return csv_dir, attr_dir


def apply_mask(inputs, mask, silence_val=None):
    if silence_val is None:
        base = inputs.amin(dim=(-3, -2, -1), keepdim=True)
    else:
        base = torch.tensor(silence_val,
                            dtype=inputs.dtype,
                            device=inputs.device).view(1, 1, 1, 1)

    masked = (inputs - base) * mask + base
    unmasked = (inputs - base) * (1 - mask) + base
    return masked, unmasked


def _set_freq_ticks(ax, n_freqs, feature_type, fmin, fmax):
    if "mel" in feature_type.lower():
        freqs = librosa.mel_frequencies(n_mels=n_freqs, fmin=fmin, fmax=fmax)
    else:
        freqs = np.linspace(fmin, fmax, n_freqs)
    idx = np.linspace(0, n_freqs - 1, 6, dtype=int)
    labels = [f"{round(freqs[i])}" for i in idx]
    ax.set_yticks(idx)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Frequency (Hz)")


def plot_mask_pair(original, masked, feature_type, fmin, fmax,
                   save_path, suptitle=""):
    if isinstance(original, torch.Tensor):
        original = original.squeeze().detach().cpu().numpy()
    if isinstance(masked, torch.Tensor):
        masked = masked.squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    all_min = min(original.min(), masked.min())
    all_max = max(original.max(), masked.max())

    for ax, mat, title in zip(axs, [original, masked], ["Original", "Masked"]):
        im = ax.imshow(mat, aspect='auto', origin='lower', vmin=all_min, vmax=all_max, cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time steps")
    fig.colorbar(im, ax=axs, fraction=0.046, label="Energy (dB)")

    _set_freq_ticks(axs[0], original.shape[0], feature_type, fmin, fmax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


class Predict:

    def __init__(self, model, feature_extractor, interp_method_cls, interp_method_kwargs, device):

        self.model = model.to(device).eval()
        self.feature_extractor = feature_extractor.to(device)
        self.interp_method_cls = interp_method_cls
        self.interp_method_kwargs = interp_method_kwargs
        self.interpretator = self.interp_method_cls(self.model, **self.interp_method_kwargs)
        self.device = device


    def predict(self, wav, wav_name, sr, feature_type,
                silence_val=None, fmin=50, fmax=14000,
                save_root="predictions", model_type="default_model"):
        
        csv_dir, attr_dir = make_exp_dirs(save_root, model_type, wav_name)

        inputs, *stages = self.feature_extractor(wav.unsqueeze(0).to(self.device))
        # inputs = inputs.to(self.device)

        phase = full_db = pre_db = None
        idx = 0
        if getattr(self.feature_extractor, "return_phase", False):
            phase = stages[idx]
            idx += 1
        if getattr(self.feature_extractor, "return_full_db", False):
            full_db = stages[idx]
            idx += 1
        if getattr(self.feature_extractor, "return_pre_db", False):
            pre_db = stages[idx]
            idx += 1

        _, masks = self.interpretator.interpret(inputs, ret_masks=True)

        orig_wav = self.feature_extractor.inverse(inputs, phase=phase, full_db=full_db, pre_db=pre_db).squeeze(0)
        torchaudio.save(os.path.join(csv_dir, f"{wav_name}_original.wav"),
                        orig_wav.cpu(), sr)

        for mask_type, m in masks.items():
            masked, unmasked = apply_mask(inputs, m, silence_val)
            if full_db is not None:
                full_db_masked, full_db_unmasked = apply_mask(full_db, m, silence_val)
            else:
                full_db_masked = full_db_unmasked = None

            if pre_db is not None:
                pre_db_masked, pre_db_unmasked = apply_mask(pre_db, m, 0.)
            else:
                pre_db_masked = pre_db_unmasked = None

            masked_wav = self.feature_extractor.inverse(
                masked,
                phase=phase,
                full_db=full_db_masked,
                pre_db=pre_db_masked
            ).squeeze(0)

            unmasked_wav = self.feature_extractor.inverse(
                unmasked,
                phase=phase,
                full_db=full_db_unmasked,
                pre_db=pre_db_unmasked
            ).squeeze(0)

            torchaudio.save(os.path.join(csv_dir, f"{wav_name}_{mask_type}_masked.wav"),
                            masked_wav.cpu(), sr)
            torchaudio.save(os.path.join(csv_dir, f"{wav_name}_{mask_type}_unmasked.wav"),
                            unmasked_wav.cpu(), sr)

            plot_mask_pair(inputs[0], masked[0],
                           feature_type, fmin, fmax,
                           save_path=os.path.join(csv_dir,
                                                  f"{wav_name}_{mask_type}_pair.png"),
                           suptitle=mask_type)

            with torch.no_grad():
                probs_orig = F.softmax(self.model(inputs), dim=1)
                probs_masked = F.softmax(self.model(masked), dim=1)
                probs_unmasked = F.softmax(self.model(unmasked), dim=1)

            results = dict(
                FF = Metrics.compute_FF(probs_orig, probs_unmasked).cpu(),
                AI = Metrics.compute_AI(probs_orig, probs_masked).cpu(),
                AD = Metrics.compute_AD(probs_orig, probs_masked).cpu(),
                AG = Metrics.compute_AG(probs_orig, probs_masked).cpu(),
                FidIn = Metrics.compute_FidIn(probs_orig, probs_masked).cpu(),
                SPS = torch.tensor(Metrics.compute_SPS(inputs, m,
                                                         probs_orig, self.device)),
                COMP = torch.tensor(Metrics.compute_COMP(inputs, m,
                                                          probs_orig, self.device))
            )

            pd.DataFrame({k: v.view(-1).numpy() for k, v in results.items()}).to_csv(
                os.path.join(csv_dir, f"{mask_type}.csv"), index=False)
            torch.save(m.cpu(),
                       os.path.join(attr_dir, f"{mask_type}.pt"))

        return results


    def compute_and_save_interpretations_set(self, dataloader, save_path, start_from_sample=1):
        os.makedirs(save_path, exist_ok=True)

        for iter_num, (wavs, y) in enumerate(dataloader, start=1):

            if iter_num < start_from_sample:
                continue

            if iter_num % 50 == 0:
                del self.interpretator
                gc.collect()
                torch.cuda.empty_cache()
                self.interpretator = self.interp_method_cls(self.model, **self.interp_method_kwargs)

            wavs = wavs.to(self.device)
            inputs, *_ = self.feature_extractor(wavs)
            attrs = self.interpretator.interpret(inputs, ret_masks=False)

            fn = os.path.join(save_path, f'iter_{iter_num}.pt')
            torch.save(attrs, fn)
        

    def predict_set(self, dataloader, results_csv_name, silence_val=None,
                    save_dir="results", model_type="default_model",
                    compute_first=True, start_from_sample=1):

        method_name, _ = os.path.splitext(results_csv_name)
        csv_dir, attr_dir = make_exp_dirs(save_dir, model_type, method_name)

        if compute_first:
            self.compute_and_save_interpretations_set(
                dataloader, attr_dir, start_from_sample)

        results = defaultdict(lambda: defaultdict(list))

        for iter_num, (wavs, y) in enumerate(dataloader, start=1):
            interp_file = os.path.join(attr_dir, f'iter_{iter_num}.pt')
            interpretations = torch.load(interp_file).to(self.device)

            wavs = wavs.to(self.device)
            inputs, *_ = self.feature_extractor(wavs)
            _, masks = self.interpretator.interpret(inputs,
                                                    interpretations=interpretations,
                                                    ret_masks=True)
            for mask_type, m in masks.items():
                masked_inputs, unmasked_inputs = apply_mask(inputs, m, silence_val)

                with torch.no_grad():
                    probs_original = F.softmax(self.model(inputs), dim=1)
                    probs_masked = F.softmax(self.model(masked_inputs), dim=1)
                    probs_unmasked = F.softmax(self.model(unmasked_inputs), dim=1)

                _, preds = torch.max(probs_original, dim=1)

                ff = Metrics.compute_FF(probs=probs_original, probs_out=probs_unmasked)
                ai = Metrics.compute_AI(probs=probs_original, probs_in=probs_masked)
                ad = Metrics.compute_AD(probs=probs_original, probs_in=probs_masked)
                ag = Metrics.compute_AG(probs=probs_original, probs_in=probs_masked)
                fidin = Metrics.compute_FidIn(probs=probs_original, probs_in=probs_masked)
                sps = Metrics.compute_SPS(inputs, m, probs_original, self.device)
                comp = Metrics.compute_COMP(inputs, m, probs_original, self.device)

                results[mask_type]["FF"].append(ff.cpu().view(-1))
                results[mask_type]["AI"].append(ai.cpu().view(-1))
                results[mask_type]["AD"].append(ad.cpu().view(-1))
                results[mask_type]["AG"].append(ag.cpu().view(-1))
                results[mask_type]["FidIn"].append(fidin.cpu().view(-1))
                results[mask_type]["SPS"].append(torch.tensor(sps).cpu().view(-1))
                results[mask_type]["COMP"].append(torch.tensor(comp).cpu().view(-1))
                results[mask_type]["is_correct"].append((preds.cpu() == y).view(-1))


        for mtype, metrics in results.items():
            df = pd.DataFrame(
                {metric: torch.cat(vals).numpy()
                 for metric, vals in metrics.items()})
            df.to_csv(os.path.join(csv_dir, f"{mtype}.csv"),
                      index_label="sample")

        print(f"Все CSV-файлы сохранены в {csv_dir}")
        return {mtype: list(mdict.keys()) for mtype, mdict in results.items()}