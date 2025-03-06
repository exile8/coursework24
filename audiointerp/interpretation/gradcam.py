from .base import BaseInterpreter
import torch
from pytorch_grad_cam import GradCAM as GradCAM_official
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMInterpreter(BaseInterpreter):

    def __init__(self, model, target_layers):
        super().__init__(model)

        self.gradcam = GradCAM_official(model=model, target_layers=target_layers)

    def compute_interpretation(self, inputs):
        
        inputs = inputs.to(self.device).requires_grad_(True)
        logits = self.model(inputs)
        predicted_class = logits.argmax(dim=1)
        targets = [ClassifierOutputTarget(int(pred)) for pred in predicted_class]

        attributions = self.gradcam(input_tensor=inputs, targets=targets)
        attributions = torch.from_numpy(attributions).unsqueeze(1).to(self.device)

        return attributions