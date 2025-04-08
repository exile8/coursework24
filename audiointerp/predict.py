import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from .metrics import Metrics
import torch.nn.functional as F
import librosa
import numpy as np

class Predict:

    def __init__(self, model, feature_extractor, interp_method_cls, interp_method_kwargs, device):

        self.model = model.to(device).eval()
        self.feature_extractor = feature_extractor.to(device)
        self.interp_method_cls = interp_method_cls
        self.interp_method_kwargs = interp_method_kwargs
        self.interpretator = self.interp_method_cls(self.model, **self.interp_method_kwargs)
        self.device = device


    def predict(self, wav, wav_name, sr, save_dir="predictions"):
        os.makedirs(save_dir, exist_ok=True)

        inputs, *stages = self.feature_extractor(wav.unsqueeze(0))
        inputs = inputs.to(self.device)


        out_dict = self.interpretator.interpret(inputs)
        _, masks = out_dict.values()

        min_per_sample = inputs.amin(dim=(1, 2, 3), keepdim=True)
        unmasked_inputs = (inputs - min_per_sample) * (1 - masks) + min_per_sample
        masked_inputs = (inputs - min_per_sample) * masks + min_per_sample

        with torch.no_grad():
            probs_original = F.softmax(self.model(inputs), dim=1)
            probs_masked   = F.softmax(self.model(masked_inputs), dim=1)
            probs_unmasked = F.softmax(self.model(unmasked_inputs), dim=1)

        ff     = Metrics.compute_FF(probs=probs_original, probs_out=probs_unmasked)
        ai     = Metrics.compute_AI(probs=probs_original, probs_in=probs_masked)
        ad     = Metrics.compute_AD(probs=probs_original, probs_in=probs_masked)
        ag     = Metrics.compute_AG(probs=probs_original, probs_in=probs_masked)
        fidin  = Metrics.compute_FidIn(probs=probs_original, probs_in=probs_masked)
        sps    = Metrics.compute_SPS(inputs, masks, probs_original, self.device)
        comp   = Metrics.compute_COMP(inputs, masks, probs_original, self.device)

        results = {
            "FF":    ff.detach().cpu(),
            "AI":    ai.detach().cpu(),
            "AD":    ad.detach().cpu(),
            "AG":    ag.detach().cpu(),
            "FidIn": fidin.detach().cpu(),
            "SPS":   torch.tensor(sps),
            "COMP":  torch.tensor(comp)
        }

        if len(stages) == 2:
            stages_masked = (stages[0], stages[1] * masks)
            stages_unmasked = (stages[0], stages[1] * (1 - masks))
        else:
            stages_masked = stages
            stages_unmasked = stages

        masked_wav = self.feature_extractor.inverse(masked_inputs, *stages_masked).squeeze(0)
        unmasked_wav = self.feature_extractor.inverse(unmasked_inputs, *stages_unmasked).squeeze(0)
        orig_wav = self.feature_extractor.inverse(inputs, *stages).squeeze(0)

        torchaudio.save(os.path.join(save_dir, f"{wav_name}_original.wav"),
                        orig_wav.cpu(), sr)
        torchaudio.save(os.path.join(save_dir, f"{wav_name}_masked.wav"),
                        masked_wav.cpu(), sr)
        torchaudio.save(os.path.join(save_dir, f"{wav_name}_unmasked.wav"),
                        unmasked_wav.cpu(), sr)



        n_mels = inputs.shape[2]
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)

        plt.figure(figsize=(6, 4))
        original_spec = inputs[0].squeeze().detach().cpu().numpy()
        plt.imshow(original_spec, aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel("Time Steps")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Original Spectrogram for {wav_name}")

        yticks = np.linspace(0, n_mels - 1, 6)
        ylabels = [f"{int(mel_freqs[int(i)])} Hz" for i in yticks]
        plt.yticks(yticks, ylabels)

        plt.savefig(os.path.join(save_dir, f"{wav_name}_original_spec.png"))
        plt.close()

        plt.figure(figsize=(6, 4))
        mask_for_plot = masked_inputs[0].squeeze().detach().cpu().numpy()
        plt.imshow(mask_for_plot, aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel("Time Steps")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Mask for {wav_name}")

        plt.yticks(yticks, ylabels)

        plt.savefig(os.path.join(save_dir, f"{wav_name}_mask.png"))
        plt.close()

        return results
        

    def predict_set(self, dataloader):
        results = {
            "FF": [],
            "AI": [],
            "AD": [],
            "AG": [],
            "FidIn": [],
            "SPS": [],
            "COMP": []
        }

        for wavs, y in dataloader:
            wavs = wavs.to(self.device)
            inputs, *_ = self.feature_extractor(wavs)
            _, masks = self.interpretator.interpret(inputs).values()

            unmasked_inputs = (inputs - inputs.amin(dim=(1, 2, 3), keepdim=True)) * (1 - masks) + inputs.amin(dim=(1, 2, 3), keepdim=True)
            masked_inputs = (inputs - inputs.amin(dim=(1, 2, 3), keepdim=True)) * masks + inputs.amin(dim=(1, 2, 3), keepdim=True)

            with torch.no_grad():
                probs_original = F.softmax(self.model(inputs), dim=1)
                probs_masked = F.softmax(self.model(masked_inputs), dim=1)
                probs_unmasked = F.softmax(self.model(unmasked_inputs), dim=1)

            ff = Metrics.compute_FF(probs=probs_original, probs_out=probs_unmasked)
            ai = Metrics.compute_AI(probs=probs_original, probs_in=probs_masked)
            ad = Metrics.compute_AD(probs=probs_original, probs_in=probs_masked)
            ag = Metrics.compute_AG(probs=probs_original, probs_in=probs_masked)
            fidin = Metrics.compute_FidIn(probs=probs_original, probs_in=probs_masked)
            sps = Metrics.compute_SPS(inputs, masks, probs_original, self.device)
            comp = Metrics.compute_COMP(inputs, masks, probs_original, self.device)

            results["FF"].append(ff.cpu())
            results["AI"].append(ai.cpu())
            results["AD"].append(ad.cpu())
            results["AG"].append(ag.cpu())
            results["FidIn"].append(fidin.cpu())
            results["SPS"].append(torch.tensor(sps).cpu())
            results["COMP"].append(torch.tensor(comp).cpu())

    
        for m in results:
            results[m] = torch.cat(results[m])

        return results