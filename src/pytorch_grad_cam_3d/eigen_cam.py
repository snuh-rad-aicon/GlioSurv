import numpy as np
from .base_cam import BaseCAM
from .utils.svd_on_activations import get_3d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        activations = activations.clip(min=np.quantile(activations, 0.01), max=np.quantile(activations, 0.99))
        return get_3d_projection(activations)
