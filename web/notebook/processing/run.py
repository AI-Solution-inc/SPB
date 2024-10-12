import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import use

from processing.ml_utils.model import get_model_deeplabv3_resnet50
from PIL import Image
from skimage.transform import resize

use('Agg') 

device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_tensor(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

def preprocess_img(img: np.ndarray):
    img_t = cv2.resize(img, (640, 640))
    img_t = torch.from_numpy(img_t).permute((2, 0, 1)).float()
    img_t = normalize_tensor(img_t)[None, :]
    return img_t


def initialize_model(ckpt_path="processing/ml_utils/checkpoints/best.pth"):
    model = get_model_deeplabv3_resnet50()

    # загрузим чекпойнт в модель для проверки
    checkpoint = torch.load(ckpt_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.to(device)
    return model


MODEL = initialize_model()
cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys','YlOrBr']
labels = ["scratch", "pixels", "keyboard", "lock", "screw", "chip", "other"]


def write_outputs2mask(serial_id, outputs):
    mask_img = 7 * np.ones(shape=outputs.shape[:2])
    for i in range(7):
        masked_output = np.ma.masked_where(outputs[..., i] > 0.7, outputs[..., i])
        mask_img[masked_output.mask] = i

    pil_mask = Image.fromarray(mask_img).convert('RGB')
    pil_mask.save(f"../notebook/media/mask_{serial_id}.png", "PNG")


def run_on_image(serial_id, image: np.ndarray):
    image_orig = Image.fromarray(image).convert('RGB')
    image_orig.save(f"../notebook/media/original_{serial_id}.png", "PNG")
    
    img_t = preprocess_img(image)
    with torch.no_grad():
        img_t = img_t.to(device)
        outputs = MODEL(img_t)
        outputs = outputs["out"]

        # use softmax to get probabilities
        outputs = outputs.softmax(dim=1)

    outputs = outputs.cpu().numpy()[0]
    outputs = resize(outputs.transpose((1, 2, 0)), image.shape[:2])
    write_outputs2mask(serial_id, outputs)

    plt.imshow(image)

    patches = []
    line2db = f"{serial_id}"
    for i in range(7):
        masked_output = np.ma.masked_where(outputs[..., i] < 0.7, outputs[..., i])
        if not np.all(masked_output.mask):
            im = plt.imshow(masked_output, alpha=0.8, cmap=cmaps[i])
            color = im.cmap(np.max(masked_output))
            patches += [mpatches.Patch(color=color, label=labels[i])]
            line2db += f" {i}"

    with open("database.txt", 'wa') as f:
        f.write(line2db)

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.axis("off")
    plt.savefig(f"../notebook/media/processed_{serial_id}.png", bbox_inches="tight")
    
    return f"../notebook/media/mask_{serial_id}.png", f"../notebook/media/processed_{serial_id}.png"