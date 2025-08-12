from scipy.ndimage import label, generate_binary_structure, binary_dilation
import numpy as np
import torch
from typing import Callable, List, Tuple
from .activations_and_gradients import ActivationsAndGradients
from .utils.svd_on_activations import get_3d_projection
from .utils.image import scale_cam_image
from .utils.model_targets import ClassifierOutputTarget


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        

        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None, None] * activations
        if eigen_smooth:
            cam = get_3d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                condition_tensor: torch.Tensor=None,
                intervention_tensor: torch.Tensor=None,
                targets: List[torch.nn.Module]=None,
                eigen_smooth: bool = False) -> np.ndarray:

        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor, condition_tensor, intervention_tensor)
        
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(outputs) for target in targets])
            loss.backward(retain_graph=True)
            
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        height, width, depth = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
        return height, width, depth

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            target_size = None if len(target_size) != len(cam.shape) - 1 else target_size
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        if len(cam_per_target_layer) == 1:
            result = np.mean(cam_per_target_layer[0], axis=1)
            return result
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def set_cam_values(self, cam: np.ndarray) -> np.ndarray:
        # cam (B, H, W, D)
        cam[:, 0, 0, :] = 0
        cam[:, 0, -1, :] = 0
        cam[:, -1, 0, :] = 0
        cam[:, -1, -1, :] = 0
        
        connectivity = generate_binary_structure(rank=3, connectivity=1)
        corner_points = [(0, 0, 0), (0, 0, -1), (0, -1, 0), (0, -1, -1), 
                        (-1, 0, 0), (-1, 0, -1), (-1, -1, 0), (-1, -1, -1)]
        
        batch_size, height, width, depth = cam.shape
        for i in range(batch_size):
            zero_mask = np.zeros_like(cam[i], dtype=bool)
            zero_mask[cam[i] <= 0] = True
            labeled_array, num_features = label(zero_mask, structure=connectivity)
            
            connected_from_corner = np.zeros_like(cam[i], dtype=bool)
            for corner in corner_points:
                corner_label = labeled_array[corner]
                if corner_label > 0:
                    connected_from_corner |= (labeled_array == corner_label)
            
            connected_from_corner_dilated = binary_dilation(connected_from_corner, structure=connectivity)
            mask_dilated = connected_from_corner_dilated & ~connected_from_corner
            cam_dilated_mean = np.mean(cam[i][mask_dilated])
            original_values = cam[i][connected_from_corner]
            cam[i][connected_from_corner] = cam_dilated_mean

            median_value = np.median(cam[i])
            if np.any(original_values > median_value):
                cam[i] = np.max(cam[i]) - cam[i]

        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 condition_tensor: torch.Tensor = None,
                 intervention_tensor: torch.Tensor = None,
                 targets: List[torch.nn.Module] = None,
                 eigen_smooth: bool = False) -> np.ndarray:

        return self.forward(input_tensor, condition_tensor, intervention_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
