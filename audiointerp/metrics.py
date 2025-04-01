import torch
import quantus
import torch.nn.functional as F

class Metrics:

    @staticmethod
    def compute_FF(logits, logits_out):
        pred_cl = torch.argmax(logits, dim=1, keepdim=True)

        logits_cl = torch.gather(logits, dim=1, index=pred_cl)
        logits_out_cl = torch.gather(logits_out, dim=1, index=pred_cl)

        ff = (logits_cl - logits_out_cl).squeeze()

        return ff

    @staticmethod
    def compute_AI(logits, logits_in):
        pred_cl = torch.argmax(logits, dim=1, keepdim=True)

        logits_cl = torch.gather(logits, dim=1, index=pred_cl)
        logits_in_cl = torch.gather(logits_in, dim=1, index=pred_cl)

        ai = (logits_in_cl > logits_cl).float().squeeze() * 100

        return ai

    @staticmethod
    def compute_AD(logits, logits_in):
        pred_cl = torch.argmax(logits, dim=1, keepdim=True)

        logits_cl = torch.gather(logits, dim=1, index=pred_cl)
        logits_in_cl = torch.gather(logits_in, dim=1, index=pred_cl)

        ad = (torch.clamp((logits_cl - logits_in_cl), min=0) / (logits_cl + 1e-6)).squeeze() * 100

        return ad

    @staticmethod
    def compute_AG(logits, logits_in):
        pred_cl = torch.argmax(logits, dim=1, keepdim=True)

        logits_cl = torch.gather(logits, dim=1, index=pred_cl)
        logits_in_cl = torch.gather(logits_in, dim=1, index=pred_cl)

        ag = (torch.clamp((logits_in_cl - logits_cl), min=0) / (1 - logits_cl + 1e-6)).squeeze() * 100

        return ag

    @staticmethod
    def compute_FidIn(logits, logits_in):
        pred_cl = torch.argmax(logits, dim=1)
        pred_in_cl = torch.argmax(logits_in, dim=1)

        fidin = (pred_in_cl == pred_cl).float()

        return fidin

    @staticmethod
    def compute_SPS(samples, interpretations, logits, device):
        pred_cl = torch.argmax(logits, dim=1)

        sps_metric = quantus.Sparseness()

        sps = sps_metric(model=None, x_batch=samples.clone().detach().cpu().numpy(),
                         y_batch=pred_cl.clone().detach().cpu().numpy(),
                         a_batch=interpretations.clone().detach().cpu().numpy(),
                         softmax=False, device=device)
        
        return sps
        
    @staticmethod
    def compute_COMP(samples, interpretations, logits, device):
        pred_cl = torch.argmax(logits, dim=1)

        comp_metric = quantus.Complexity()

        comp = comp_metric(model=None, x_batch=samples.clone().detach().cpu().numpy(),
                           y_batch=pred_cl.clone().detach().cpu().numpy(),
                           a_batch=interpretations.clone().detach().cpu().numpy(),
                           softmax=False, device=device)
        
        return comp