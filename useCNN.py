# DOWNLOAD LIBRARIES ############################################

import urllib.request
import zipfile
import os

# DOWNLOAD MODEL  ############################################

url_model = "https://github.com/milesberberich/TreeCrown-InstanceSegementation/releases/download/v1.0/Tree_segmentation_model.zip"

zip_path = "my_model.zip" # change the paths
extract_dir = "model_weights" # change the paths

urllib.request.urlretrieve(url_model, zip_path)
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# DOWNLOAD DATA  ############################################

url_data = "https://github.com/milesberberich/TreeCrown-InstanceSegementation/blob/main/mini_test_set.zip"

urllib.request.urlretrieve(url_data, "test_data.zip")

with zipfile.ZipFile("test_data.zip", 'r') as zip_ref:
    zip_ref.extractall("test_data")


import torch
import matplotlib.pyplot as plt
from skimage import io
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.utils import draw_segmentation_masks

# 1. Initialize the exact architecture used in training
backbone = resnet_fpn_backbone(backbone_name="resnet34", weights=None)[cite: 1]
model = MaskRCNN(backbone, num_classes=2)[cite: 1]

# Load weights (extracting 'model_state_dict' from your saved checkpoint)
ckpt = torch.load("model_weights/best_model.pth", map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])[cite: 1]
model.eval()

# 2. Load a TIFF test image
img_path = "tiff_data/sample_tree_image.tiff"
img_arr = io.imread(img_path)[:, :, :3]  # Extract first 3 bands for consistency[cite: 1]
img_tensor = torch.as_tensor(img_arr.transpose(2, 0, 1), dtype=torch.float32) / 255.0[cite: 1]

# 3. Run Inference
with torch.no_grad():
    prediction = model([img_tensor])[0]

# Filter masks by confidence score (e.g., > 50% certainty)
score_threshold = 0.5
pred_masks = prediction['masks'][prediction['scores'] > score_threshold] > 0.5

# 4. Plot Results
# Convert image to uint8 format required by torchvision's drawing utilities
img_uint8 = (img_tensor * 255).to(torch.uint8)
pred_viz = draw_segmentation_masks(img_uint8, pred_masks.squeeze(1), alpha=0.5, colors="red")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Original Image (Use this as your visual ground truth baseline if COCO JSON is unloaded)
axs[0].imshow(img_arr)
axs[0].set_title("Original TIFF")
axs[0].axis('off')

# Model Predictions
axs[1].imshow(pred_viz.permute(1, 2, 0).numpy())
axs[1].set_title("Mask R-CNN Predicted Crowns")
axs[1].axis('off')

plt.tight_layout()
plt.show()