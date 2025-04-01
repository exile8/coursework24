from .base import BaseInterpreter
import torch
import shap

class SHAPInterpreter(BaseInterpreter):

    def __init__(self, model, background_data):
        super().__init__(model)
        self.explainer = shap.Explainer(self.model)

        background_data = background_data.to("cpu").detach().numpy()
        
        self.explainer = shap.KernelExplainer(self.model_forward, background_data)

    def model_forward(self, inputs):
        inputs = torch.tensor(inputs).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
        return logits.cpu().numpy()

    def compute_interpretation(self, inputs):
        inputs = inputs.to(self.device).requires_grad_(True)
        inputs_np = inputs.detach().cpu().numpy()
        
        logits = self.model(inputs)
        predicted_class = logits.argmax(dim=1).cpu().numpy()

        shap_values = self.explainer(inputs_np)
        attributions = torch.tensor(shap_values.values).to(self.device)

        attributions = attributions.unsqueeze(1)

        return attributions

