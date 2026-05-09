"""
Author: Ceren Arkaç
Date: May 9, 2026

FG/BG segmentation using DINOv2's emergent PCA property.
PCA1 of patch features separates foreground from background;
sign correction uses prenorm feature norms.

"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import glob
import os
import argparse
from tqdm import tqdm

def extract_patch_embs(image_path, model, transform, device):
    img = Image.open(image_path).convert("RGB")
    img = transform(img) # (3,224,224)
    # pytorch models expect data in batches. add 1 more dimension
    inp = img.unsqueeze(0).to(device) # (1,3,224,224)
    with torch.no_grad():
        out = model.forward_features(inp) # out keys: ['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks']
        patches = out["x_norm_patchtokens"].squeeze(0).cpu().numpy() # (N, D)
        patches_prenorm = out["x_prenorm"].squeeze(0).cpu().numpy() # (N+1, D)
        patches_prenorm = patches_prenorm[-patches.shape[0]:, :] # (N, D)
        patch_norms = np.linalg.norm(patches_prenorm, axis=1) # (N,)

    return patches, patch_norms

def foreg_backg_detector(patch_embs, patch_norms):
    pca = PCA(n_components=1)
    pca.fit(patch_embs)
    # get the eigenvector that matches up with the biggest eigenvalue of the covariance matrix of the projection
    pc1 = pca.components_[0] # (D,)
    # subtract the mean calculated by sklearn to shift our vectors to the new origin
    centered_embs = patch_embs - pca.mean_ # (total_N, D)
    # project patches onto the first principal component
    projection = centered_embs @ pc1 # (total_N,)
    """
    For vector x, the projection is ||x||.||pc1||.cos(angle in between)
    Since the principal components are unit vectors, 
    the result gets simplified to ||x||.cos(angle in between)
    Actually, DINO patch features are also L2-normalized if x_norm_patchtokens are used.
    But, sklearn's PCA automatically mean-centers the data under the hood. 
    Because the origin shifts to this new center of mass, our shifted features (x - mean) 
    are no longer unit vectors.
    The sign of the projection depends entirely on the angle in between.
    I want the positive projected features (angle < 90) to be for the foreground object.
    But there is a problem: PC1's direction is arbitrary. Positive side could be FG or BG.
    To solve this problem, I will utilize
    one of the properties of DINO: foreground patches exhibit 
    higher prenorm feature norms than background patches.
    If projection correlates negatively with norms, positive side = BG ---> flip the projection.
    """
    correlation, _ = pearsonr(projection, patch_norms)
    if correlation < 0:
        projection = -projection
    mask = projection > 0.0

    return mask, projection

def visualize_foreground(image_paths, per_image_masks, k=5):
    # pick k random images (or fewer if dataset is smaller)
    k = min(k, len(image_paths))
    indices = np.random.choice(len(image_paths), k, replace=False)

    fig, axes = plt.subplots(k, 2, figsize=(8, 4 * k))
    if k == 1:
        axes = axes[np.newaxis, :]  # keep 2D indexing consistent

    patch_size = 14
    grid_size = 224 // patch_size  # 16

    for i, idx in enumerate(indices):
        img = Image.open(image_paths[idx]).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img)

        mask_2d = per_image_masks[idx].reshape(grid_size, grid_size)
        mask_pixel = np.repeat(np.repeat(mask_2d, patch_size, axis=0), patch_size, axis=1)

        result = img_np.copy()
        result[~mask_pixel] = 0

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(os.path.basename(image_paths[idx]))
        axes[i, 1].imshow(result)
        axes[i, 1].set_title("Foreground")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("foreground_vis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to foreground_vis.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="dinov2_vitb14")
    args = parser.parse_args()

    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/dinov2', args.model).to(device)
    model.eval()

    # Image preprocessing (DINOv2 uses ImageNet normalization)
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Process all images in a given directory
    # find all folders in the given directory and get the image paths inside of the each folder
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "**", "*.jpg"), recursive=True))
    print(f"Found {len(image_paths)} images")

    all_patches = []
    all_norms = []
    patches_per_image = []  # track how many patches each image contributes

    for path in tqdm(image_paths, desc="Extracting patches"):
        patches, norms = extract_patch_embs(path, model, transform, device)
        all_patches.append(patches)
        all_norms.append(norms)
        patches_per_image.append(patches.shape[0])

    all_patches = np.concatenate(all_patches, axis=0)    # (total_N, D)
    all_norms = np.concatenate(all_norms, axis=0)         # (total_N,)

    # PCA on ALL images combined. this captures dataset-wide FG/BG structure, not single-image noise
    mask, _ = foreg_backg_detector(all_patches, all_norms)

    # split the global mask back to per-image masks
    per_image_masks = np.split(mask, np.cumsum(patches_per_image)[:-1])

    visualize_foreground(image_paths, per_image_masks, args.num_vis)