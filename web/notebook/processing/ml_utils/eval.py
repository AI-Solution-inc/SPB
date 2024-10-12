import numpy as np
import cv2

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from model import get_model_deeplabv3_resnet50
from skimage.transform import resize


def normalize_tensor(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

def preprocess_img(img: np.ndarray):
    img_t = cv2.resize(img, (640, 640))
    img_t = torch.from_numpy(img_t).permute((2, 0, 1)).float()
    img_t = normalize_tensor(img_t)[None, :]
    return img_t


def initialize_model(ckpt_path="checkpoints/exp_2/best.pth"):
    model = get_model_deeplabv3_resnet50()

    # загрузим чекпойнт в модель для проверки
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = "cpu"
    model.to(device)
    return model


MODEL = initialize_model()
cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys','YlOrBr']
labels = ["scratch", "pixels", "keyboard", "lock", "screw", "chip", "other"]


def write_outputs2mask(outputs):
    pass

def run_on_image(image: np.ndarray):
    img_t = preprocess_img(image)
    with torch.no_grad():
        outputs = MODEL(img_t)
        outputs = outputs["out"]

        # use softmax to get probabilities
        outputs = outputs.softmax(dim=1)

    outputs = outputs.cpu().numpy()[0]
    outputs = resize(outputs.transpose((1, 2, 0)), image.shape[:2])

    plt.imshow(image)

    patches = []
    for i in range(7):
        masked_output = np.ma.masked_where(outputs[..., i] < 0.7, outputs[..., i])
        if not np.all(masked_output.mask):
            im = plt.imshow(masked_output, alpha=0.8, cmap=cmaps[i])
            color = im.cmap(np.max(masked_output))
            patches += [mpatches.Patch(color=color, label=labels[i])]

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.axis("off")
    plt.savefig("../myapp/media/processed.png", bbox_inches="tight")