import urllib.request
import zipfile
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pycocotools.coco import COCO
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.utils import draw_segmentation_masks

# 1. DOWNLOAD WEIGHTS
urllib.request.urlretrieve(
    "https://github.com/milesberberich/Tree-MaskR-CNN/releases/download/v1.0/Tree_segmentation_model.zip",
    "my_model.zip")
with zipfile.ZipFile("my_model.zip", 'r') as z:
    z.extractall("model_weights")

urllib.request.urlretrieve("https://github.com/milesberberich/Tree-MaskR-CNN/raw/main/mini_test_set.zip",
                           "test_data.zip")
with zipfile.ZipFile("test_data.zip", 'r') as z:
    z.extractall("test_data")

# 2. PATHS
base_dir = "test_data/coco1024/test2023/Test-Set-2"
json_path = "test_data/coco1024/annotations/instances_tree_TestSet22023.json"

# 3. BUILD MODEL
model = MaskRCNN(resnet_fpn_backbone(backbone_name="resnet34", weights=None), num_classes=2)
model.load_state_dict(
    torch.load("model_weights/Tree_segmentation_model.pth", map_location='cpu', weights_only=True)['model_state_dict'])
model.eval()

# 4. LOAD COCO & SETUP PLOT
coco = COCO(json_path)
valid_files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]
n_imgs = len(valid_files)

fig, axs = plt.subplots(n_imgs, 3, figsize=(15, 5 * n_imgs), squeeze=False)

# 5. RUN MODEL & PLOT PREDICTIONS
for idx, filename in enumerate(valid_files):
    # Load test image examples
    img_path = f"{base_dir}/{filename}"
    img_arr = io.imread(img_path)[:, :, :3]
    img_tensor = torch.as_tensor(img_arr.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    img_uint8 = (img_tensor * 255).to(torch.uint8)

    with torch.no_grad():
        pred = model([img_tensor])[0]

    pred_masks = pred['masks'][pred['scores'] > 0.5] > 0.5
    pred_masks_flat = pred_masks.any(dim=0, keepdim=True).squeeze(1)
    pred_viz = draw_segmentation_masks(img_uint8, pred_masks_flat, alpha=0.5, colors="red")

    # Extract Ground Truth
    img_id = next(k for k, v in coco.imgs.items() if v['file_name'] == filename)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    gt_masks = [coco.annToMask(ann) for ann in coco.loadAnns(ann_ids)]

    # Flatten ground truth
    gt_tensor_flat = torch.as_tensor(np.array(gt_masks), dtype=torch.bool).any(dim=0, keepdim=True)
    gt_viz = draw_segmentation_masks(img_uint8, gt_tensor_flat, alpha=0.5, colors="blue")

    # Plot
    axs[idx, 0].imshow(img_arr)
    axs[idx, 0].set_title(f"Original: {filename}")
    axs[idx, 0].axis('off')

    axs[idx, 1].imshow(gt_viz.permute(1, 2, 0).numpy())
    axs[idx, 1].set_title("Ground Truth (Blue)")
    axs[idx, 1].axis('off')

    axs[idx, 2].imshow(pred_viz.permute(1, 2, 0).numpy())
    axs[idx, 2].set_title("Predictions (Red)")
    axs[idx, 2].axis('off')

plt.tight_layout()
plt.show()
