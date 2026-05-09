# dinov2-foreground-discovery
Foreground/background segmentation using DINOv2's emergent PCA property. Learning project exploring how PCA1 of ViT patch features separates objects from background without any supervision.

## How it works

DINOv2's patch features have an emergent property: the first principal component of their covariance matrix separates foreground from background. This script exploits that property:

1. **Extract patch embeddings** from DINOv2 ViT-B/14 for all images in a dataset
2. **Fit PCA** on the combined patch features (dataset-wide, not per-image)
3. **Project** each patch onto PC1 — positive = one cluster, negative = the other
4. **Sign correction** using prenorm feature norms — foreground patches have higher norms, so if the correlation between projection and norms is negative, flip the sign
5. **Visualize** — randomly selected images shown side-by-side with their foreground masks, saved to `foreground_vis.png`

## Why dataset-wide PCA?

Running PCA on a single image is noisy — PC1 may capture lighting or color gradients instead of FG/BG. When PCA is fit on thousands of images, the FG/BG separation becomes the dominant axis of variance because foreground objects (birds, cars) are systematically different from backgrounds (sky, trees, pavement) across the dataset.

## Usage

```bash
python main.py --image_dir /path/to/images
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | required | Path to image directory (searches recursively for .jpg files) |
| `--num_vis` | 5 | Number of images to visualize |
| `--model` | `dinov2_vitb14` | DINOv2 model variant (e.g. `dinov2_vitb14_reg`, `dinov2_vitl14`) |
| `--seed` | 42 | Random seed for reproducible visualization |

## Requirements

```
torch
torchvision
numpy
scikit-learn
scipy
matplotlib
Pillow
tqdm
```

## References

- [DINOv2 (Oquab et al., 2024)](https://arxiv.org/abs/2304.07193) — self-supervised ViT features with emergent FG/BG separation in PCA

## License

MIT
