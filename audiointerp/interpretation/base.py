from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch


class BaseInterpreter(ABC):

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device


    @abstractmethod
    def compute_interpretation(self, inputs, **kwargs):
        """Apply interpretation method to given samples"""
        pass


    def create_masks(self, A):
        """Mask intepretation"""

        threshold_vals = [0.25, 0.5, 0.75]
        topK_percent_vals = [0.05, 0.3, 0.5]

        A_pos = torch.clamp(A, min=0.)
        
        A_min = A.amin(dim=(-3, -2, -1), keepdim=True)
        A_max = A.amax(dim=(-3, -2, -1), keepdim=True)

        m_minmax = (A - A_min) / (A_max - A_min + 1e-8)
        m_sigmoid = F.sigmoid(A)
        m_bin = (A > 0).float()
        m_topK = dict()
        for k in topK_percent_vals:
            thresh = torch.quantile(A.flatten(1), 1-k, dim=1, keepdim=True)
            thresh = thresh.view(A.size(0), *[1]*(A.ndim-1))
            m_topK[f"topK_{int(k*100)}"] = (A >= thresh).float()

        m_pos_minmax = A_pos / (A_max + 1e-8)
        m_pos_sigmoid = F.sigmoid(A_pos)
        m_pos_thresh = dict()
        for tau in threshold_vals:
            m_pos_thresh[f"pos_thresh_{int(tau * 100)}"] = (m_pos_minmax >= tau).float()
        m_pos_topK = dict()
        for k in topK_percent_vals:
            thresh = torch.quantile(A_pos.flatten(1), 1-k, dim=1, keepdim=True)
            thresh = thresh.view(A_pos.size(0), *[1]*(A.ndim-1))
            m_pos_topK[f"topK_{int(k*100)}_pos"] = (A_pos >= thresh).float()

        masks = {
            "minmax": m_minmax,
            "sigmoid": m_sigmoid,
            "bin": m_bin,
            **m_topK,
            "minmax_pos": m_pos_minmax,
            "sigmoid_pos": m_pos_sigmoid,
            **m_pos_thresh,
            **m_pos_topK
        }

        return masks
    

    def interpret(self, inputs, interpretations=None, ret_masks=True, **kwargs):
        """Interpret given samples
        Returns attributions and masks (optionally)
        """

        if interpretations is None:
            interpretations = self.compute_interpretation(inputs, **kwargs)

        if ret_masks:
            masks = self.create_masks(interpretations)
            return interpretations, masks

        return interpretations