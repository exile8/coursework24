import torch
import quantus

class Metrics:

    @staticmethod
    def compute_FF(probs, probs_out):
        pred_cl = torch.argmax(probs, dim=1, keepdim=True)

        probs_cl = torch.gather(probs, dim=1, index=pred_cl)
        probs_out_cl = torch.gather(probs_out, dim=1, index=pred_cl)

        ff = (probs_cl - probs_out_cl).squeeze()

        return ff

    @staticmethod
    def compute_AI(probs, probs_in):
        pred_cl = torch.argmax(probs, dim=1, keepdim=True)

        probs_cl = torch.gather(probs, dim=1, index=pred_cl)
        probs_in_cl = torch.gather(probs_in, dim=1, index=pred_cl)

        ai = (probs_in_cl > probs_cl).float().squeeze() * 100

        return ai

    @staticmethod
    def compute_AD(probs, probs_in):
        pred_cl = torch.argmax(probs, dim=1, keepdim=True)

        probs_cl = torch.gather(probs, dim=1, index=pred_cl)
        probs_in_cl = torch.gather(probs_in, dim=1, index=pred_cl)

        ad = (torch.clamp((probs_cl - probs_in_cl), min=0) / (probs_cl + 1e-6)).squeeze() * 100

        return ad

    @staticmethod
    def compute_AG(probs, probs_in):
        pred_cl = torch.argmax(probs, dim=1, keepdim=True)

        probs_cl = torch.gather(probs, dim=1, index=pred_cl)
        probs_in_cl = torch.gather(probs_in, dim=1, index=pred_cl)

        ag = (torch.clamp((probs_in_cl - probs_cl), min=0) / (1 - probs_cl + 1e-6)).squeeze() * 100

        return ag

    @staticmethod
    def compute_FidIn(probs, probs_in):
        pred_cl = torch.argmax(probs, dim=1)
        pred_in_cl = torch.argmax(probs_in, dim=1)

        fidin = (pred_in_cl == pred_cl).float()

        return fidin

    @staticmethod
    def compute_SPS(samples, interpretations, probs, device):
        pred_cl = torch.argmax(probs, dim=1)

        sps_metric = quantus.Sparseness(normalise=True, abs=True)

        if torch.allclose(interpretations, torch.zeros_like(interpretations)):
            return [0.0]

        sps = sps_metric(model=None, x_batch=samples.clone().detach().cpu().numpy(),
                         y_batch=pred_cl.clone().detach().cpu().numpy(),
                         a_batch=interpretations.clone().detach().cpu().numpy(),
                         softmax=False, device=device)
        
        return sps
        
    @staticmethod
    def compute_COMP(samples, interpretations, probs, device):
        pred_cl = torch.argmax(probs, dim=1)

        comp_metric = quantus.Complexity(normalise=True, abs=True)

        if torch.allclose(interpretations, torch.zeros_like(interpretations)):
            return [0.0]
            
        comp = comp_metric(model=None, x_batch=samples.clone().detach().cpu().numpy(),
                           y_batch=pred_cl.clone().detach().cpu().numpy(),
                           a_batch=interpretations.clone().detach().cpu().numpy(),
                           softmax=False, device=device)
        
        return comp