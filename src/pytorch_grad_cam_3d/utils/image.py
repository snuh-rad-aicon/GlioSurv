import numpy as np
from skimage.transform import resize


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-9 + np.max(img))
        if target_size is not None:
            img = resize(img, target_size)
        result.append(img)
    result = np.float32(result)
    return result
