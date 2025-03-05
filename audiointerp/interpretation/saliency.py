from .base import BaseInterpretation
from captum.attr import Saliency as Saliency_captum


class Saliency(BaseInterpretation):

    def __init__(self, model):
        super().__init__(model)

        self.saliency = Saliency_captum(model)

    def interpret(self, inputs):
        
        inputs = inputs.to(self.device).requires_grad_(True)
        logits = self.model(inputs)
        predicted_class = logits.argmax(dim=1)

        attributions = self.saliency.attribute(inputs, target=predicted_class, abs=True)

        return attributions