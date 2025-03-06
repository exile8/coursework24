from abc import ABC, abstractmethod
import torch.nn.functional as F


class BaseInterpreter(ABC):

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device


    @abstractmethod
    def compute_interpretation(self, inputs, **kwargs):
        """Apply interpretation method to given samples"""
        pass


    def create_masks(self, attributions):
        """Mask intepretation"""
        masks = F.sigmoid(attributions)
        return masks
    

    def interpret(self, inputs, **kwargs):
        """Interpret given samples.
        Return interpretations with corresponding masks.
        Results are stored as a dict {"ints": ..., "masks": ...}
        """

        interpretations = self.compute_interpretation(inputs, **kwargs)
        masks = self.create_masks(interpretations)

        result = {
            "ints": interpretations,
            "masks": masks
        }

        return result