from .base import BaseInterpreter
from captum.attr import Saliency as Saliency_captum
import torch


class SaliencyInterpreter(BaseInterpreter):

    def __init__(self, model):
        super().__init__(model)
        
        self.saliency = Saliency_captum(self.model)

    def compute_interpretation(self, inputs):
        
        inputs = inputs.to(self.device).requires_grad_(True)
        logits = self.model(inputs)
        predicted_class = logits.argmax(dim=1)

        attributions = self.saliency.attribute(inputs, target=predicted_class, abs=False)

        return attributions