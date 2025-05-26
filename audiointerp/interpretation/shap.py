from .base import BaseInterpreter
import torch
import shap
import gc

class SHAPInterpreter(BaseInterpreter):

    def __init__(self, model, background_data):
        super().__init__(model)
        
        self.explainer = shap.DeepExplainer(self.model, background_data)

    def compute_interpretation(self, inputs):
        inputs = inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
        predicted_class = logits.argmax(dim=1).cpu().numpy()

        shap_values = self.explainer.shap_values(inputs, ranked_outputs=1, output_rank_order='max', check_additivity=False)
        print("Done extracting shap values")

        attributions = torch.from_numpy(shap_values[0][..., 0]).to(device=self.device, dtype=torch.float32)

        return attributions

