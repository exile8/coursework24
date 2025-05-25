from .base import BaseInterpreter
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def scale_cam_image_signed(cam, target_size, keep_dtype=False):
    import numpy as np
    import cv2

    dtype = cam.dtype
    cam = np.nan_to_num(cam)
    is_3d = cam.ndim == 3
    if is_3d:
        resized = []
        for z in cam:
            z_rz = cv2.resize(z, target_size, interpolation=cv2.INTER_LINEAR)
            resized.append(z_rz)
        cam = np.stack(resized, axis=0)
    else:
        cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)

    if keep_dtype:
        cam = cam.astype(dtype)
    return cam

class SignedGradCAM(GradCAM):
    def __init__(self, model, target_layers):
        super(SignedGradCAM, self).__init__(model=model, target_layers=target_layers)
        self.detach = True

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            scaled = scale_cam_image_signed(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

class GradCAMInterpreter(BaseInterpreter):

    def __init__(self, model, target_layers):
        super().__init__(model)

        self.gradcam = SignedGradCAM(model=model, target_layers=target_layers)

    def compute_interpretation(self, inputs):
        
        inputs = inputs.to(self.device).requires_grad_(True)
        logits = self.model(inputs)
        predicted_class = logits.argmax(dim=1)
        targets = [ClassifierOutputTarget(int(pred)) for pred in predicted_class]

        attributions = self.gradcam(input_tensor=inputs, targets=targets)
        attributions = torch.from_numpy(attributions).unsqueeze(1).to(self.device)

        return attributions