import numpy as np
import matplotlib.pyplot as plt

NUM_ANCHORS_X = 4
NUM_ANCHORS_Y = 12
ASPECT_RATIOS = [2.0, 1.5]     # w/h
ANCHOR_SCALE = 1.5
NUM_SCALES = 1


def generate_boxes(imshape):
    """Generates multiscale anchor boxes. 
    Code is adapted from the repository of EfficientDet: https://arxiv.org/abs/1911.09070"""
    anchor_configs = []
    for scale_octave in range(NUM_SCALES):
        for aspect in ASPECT_RATIOS:
            stride = (imshape[0] // NUM_ANCHORS_Y, imshape[1] // NUM_ANCHORS_X)
            anchor_configs.append((stride, scale_octave / float(NUM_SCALES), aspect, ANCHOR_SCALE))

    all_boxes = []

    for config in anchor_configs:
        stride, octave_scale, aspect, scale = config
        base_anchor_size_x = scale * stride[1] * 2**octave_scale
        base_anchor_size_y = scale * stride[0] * 2**octave_scale

        aspect_x = np.sqrt(aspect)
        aspect_y = 1.0 / aspect_x

        anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0
        anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0

        x = np.arange(stride[1] / 2, imshape[1], stride[1])
        y = np.arange(stride[0] / 2, imshape[0], stride[0])
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)

        boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,      # [y1, x1, y2, x2]
                           yv + anchor_size_y_2, xv + anchor_size_x_2))
        boxes = np.swapaxes(boxes, 0, 1)
        all_boxes.append(boxes)
    all_boxes = np.vstack(all_boxes)

    lower_bounds = np.array([0, 0, 0, 0])
    upper_bounds = np.array([imshape[0], imshape[1], imshape[0], imshape[1]])

    all_boxes = np.clip(all_boxes, lower_bounds, upper_bounds)
    return all_boxes


if __name__ == "__main__":
    imshape = (1300, 1300)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.zeros(imshape))
    boxes = generate_boxes(imshape)
    for box in boxes:
        rect = plt.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0],
                             edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("Anchor boxes")
    plt.show()

     