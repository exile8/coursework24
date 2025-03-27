import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from .metrics import Metrics

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

        if not isinstance(wav, list):
            wavs = [wav]
        else:
            wavs = wav

        inputs, stages = self.feature_extractor(wavs)
        inputs = inputs.to(self.device)

        out_dict = self.interpretator.interpret(inputs)
        _, masks = out_dict.values()

        min_per_sample = inputs.amin(dim=(1, 2, 3), keepdim=True)
        unmasked_inputs = (inputs - min_per_sample) * (1 - masks) + min_per_sample
        masked_inputs = (inputs - min_per_sample) * masks + min_per_sample

        with torch.no_grad():
            logits_original = self.model(inputs)
            logits_masked   = self.model(masked_inputs)
            logits_unmasked = self.model(unmasked_inputs)

        ff     = Metrics.compute_FF(logits=logits_original, logits_out=logits_unmasked)
        ai     = Metrics.compute_AI(logits=logits_original, logits_in=logits_masked)
        ad     = Metrics.compute_AD(logits=logits_original, logits_in=logits_masked)
        ag     = Metrics.compute_AG(logits=logits_original, logits_in=logits_masked)
        fidin  = Metrics.compute_FidIn(logits=logits_original, logits_in=logits_masked)
        sps    = Metrics.compute_SPS(inputs, masks, logits_original, self.device)
        comp   = Metrics.compute_COMP(inputs, masks, logits_original, self.device)

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
            stages = (stages[0], stages[1] * masks)

        masked_wav   = self.feature_extractor.inverse(masked_inputs, *stages)
        unmasked_wav = self.feature_extractor.inverse(unmasked_inputs, *stages)
        orig_wav     = self.feature_extractor.inverse(inputs, *stages)

        torchaudio.save(os.path.join(save_dir, f"{wav_name}_original.wav"),
                        orig_wav.cpu(), sr)
        torchaudio.save(os.path.join(save_dir, f"{wav_name}_masked.wav"),
                        masked_wav.cpu(), sr)
        torchaudio.save(os.path.join(save_dir, f"{wav_name}_unmasked.wav"),
                        unmasked_wav.cpu(), sr)

        plt.figure(figsize=(6,4))
        mask_for_plot = masks[0].squeeze().detach().cpu().numpy()
        plt.imshow(mask_for_plot, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"Mask for {wav_name}")
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
                logits_original = self.model(inputs)
                logits_masked = self.model(masked_inputs)
                logits_unmasked = self.model(unmasked_inputs)

        ff = Metrics.compute_FF(logits=logits_original, logits_out=logits_unmasked)
        ai = Metrics.compute_AI(logits=logits_original, logits_in=logits_masked)
        ad = Metrics.compute_AD(logits=logits_original, logits_in=logits_masked)
        ag = Metrics.compute_AG(logits=logits_original, logits_in=logits_masked)
        fidin = Metrics.compute_FidIn(logits=logits_original, logits_in=logits_masked)
        sps = Metrics.compute_SPS(inputs, masks, logits_original, self.device)
        comp = Metrics.compute_COMP(inputs, masks, logits_original, self.device)

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