from .base import BaseInterpreter
from lime.lime_image import LimeImageExplainer
import torch
import numpy as np
import gc

class LIMEInterpreter(BaseInterpreter):

    def __init__(self, model, num_samples=1000):
        super().__init__(model)

        self.num_samples = num_samples
        self.lime = LimeImageExplainer(random_state=12345)


    def compute_interpretation_single(self, input_single):

        def transform_fn(tensor_spec):
            img = tensor_spec.detach().cpu().squeeze().numpy().astype(np.float32)
            spec_min = img.min()
            spec_max = img.max()
            img = (img - spec_min) / (spec_max - spec_min + 1e-8)
            return img, spec_max, spec_min

        input_img, spec_max, spec_min = transform_fn(input_single)

        def predict_fn(imgs):

            specs_gray = np.mean(imgs, axis=-1)
            specs_restored =  specs_gray * (spec_max - spec_min) + spec_min

            input_specs = torch.from_numpy(specs_restored).unsqueeze(1).to(self.device)

            with torch.no_grad():
                output = self.model(input_specs)
                probs = torch.softmax(output, dim=1)

            probs_np = probs.detach().cpu()
            
            return probs_np

        explanation = self.lime.explain_instance(
            input_img,
            classifier_fn=predict_fn,
            top_labels=1,
            num_samples=self.num_samples
        )
        
        segments = explanation.segments
        attributions = np.zeros_like(segments, dtype=np.float32)

        for seg_id, weight in list(explanation.local_exp.values())[0]:
            attributions[segments == seg_id] = weight

        attributions = torch.from_numpy(attributions).unsqueeze(0).to(self.device)

        return attributions

    def compute_interpretation(self, inputs):
        attributions = []
        for i in range(inputs.shape[0]):
            input_single = inputs[i:i+1]
            attr = self.compute_interpretation_single(input_single)
            attributions.append(attr)

        return torch.stack(attributions, dim=0) 