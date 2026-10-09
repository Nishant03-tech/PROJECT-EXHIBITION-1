"""Lung-Masked Grad-CAM Visualizer for Chest X-Ray TB Detection.

Reproduces clinical paper visualizations:
- Side-by-side Dual Column: "Original" and "GradCAM"
- Multi-row labeled formatting: (A), (B), (C), (D)
- Anatomical Lung Field Masking: Restricts Grad-CAM heatmap strictly inside the lungs
- Dimmed anatomical background with purple wash (eliminates border & text artifacts)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import IMG_SIZE


def extract_lung_mask(rgb_img: np.ndarray) -> np.ndarray:
    """Extracts an anatomical lung field mask using adaptive thresholding,

    morphological filtering, and bilateral thoracic component extraction.
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. CLAHE enhancement to maximize lung vs bone/soft-tissue contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gaussian blur to suppress fine noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 3. Otsu thresholding (lung cavities are dark, thoracic cage is light)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Remove image border artifacts (clear outer 6% margin to avoid border padding/labels)
    margin_y, margin_x = int(0.06 * h), int(0.06 * w)
    inner_mask = np.zeros_like(thresh)
    inner_mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255
    thresh = cv2.bitwise_and(thresh, inner_mask)

    # 5. Morphological closing to seal internal vascular markings and rib shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. Extract bilateral thoracic lung cavities
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    lung_mask = np.zeros_like(gray, dtype=np.uint8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1] + 1
        total_pixels = h * w

        for idx in sorted_indices[:4]:
            area = stats[idx, cv2.CC_STAT_AREA]
            # Plausible lung cavity size is between 4% and 42% of total image area
            if 0.04 * total_pixels <= area <= 0.42 * total_pixels:
                lung_mask[labels == idx] = 255

    # 7. Fallback to soft thoracic ribcage envelope if segmentation area is too small
    if np.sum(lung_mask > 0) < 0.08 * (h * w):
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.48, w * 0.50
        # Two bilateral lobes
        left_lobe = (((x - (w * 0.32)) ** 2 / (0.18 * w) ** 2 +
                      (y - center_y) ** 2 / (0.35 * h) ** 2) <= 1)
        right_lobe = (((x - (w * 0.68)) ** 2 / (0.18 * w) ** 2 +
                       (y - center_y) ** 2 / (0.35 * h) ** 2) <= 1)
        lung_mask = ((left_lobe | right_lobe).astype(np.uint8)) * 255

    # Smooth mask contours for natural anatomical blending
    lung_mask = cv2.GaussianBlur(lung_mask, (11, 11), 0)
    return (lung_mask / 255.0).astype(np.float32)


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
) -> tuple[np.ndarray, float]:
    """Computes Grad-CAM using numerically stable pre-sigmoid gradients.
    
    100% compatible with Keras 3 & loaded models (avoids Functional submodel reconstruction).
    """
    backbone = model.get_layer("densenet121")
    layer_names = [l.name for l in model.layers]

    with tf.GradientTape() as tape:
        conv_outputs = backbone(input_tensor, training=False)
        tape.watch(conv_outputs)

        x = conv_outputs
        if "cbam_attention" in layer_names:
            x = model.get_layer("cbam_attention")(x, training=False)

        if "global_avg_pool" in layer_names and "global_max_pool" in layer_names:
            gap = model.get_layer("global_avg_pool")(x)
            gmp = model.get_layer("global_max_pool")(x)
            x = model.get_layer("hybrid_pooling")([gap, gmp])
        elif "global_average_pooling2d" in layer_names:
            x = model.get_layer("global_average_pooling2d")(x)

        # Forward through dense layers
        for name in ["head_dense_1", "head_bn_1", "head_dropout_1",
                     "head_dense_2", "head_bn_2", "head_dropout_2",
                     "dense", "dropout", "dense_1", "dropout_1"]:
            if name in layer_names:
                try:
                    x = model.get_layer(name)(x, training=False)
                except Exception:
                    x = model.get_layer(name)(x)

        if "tb_probability" in layer_names:
            predictions = model.get_layer("tb_probability")(x)
        else:
            predictions = model.layers[-1](x)

        score_val = float(predictions[0, 0].numpy())
        score_clipped = tf.clip_by_value(predictions[:, 0], 1e-7, 1.0 - 1e-7)
        logit = tf.math.log(score_clipped / (1.0 - score_clipped))

    grads = tape.gradient(logit, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()

    cam = np.maximum(cam, 0)
    max_cam = np.max(cam)
    if max_cam > 0:
        cam = cam / max_cam

    cam_resized = cv2.resize(cam, IMG_SIZE)
    return cam_resized, score_val


def render_clinical_gradcam(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    lung_mask: np.ndarray,
) -> np.ndarray:
    """Combines original image, masked heatmap, and dimmed purple surround.
    
    Replicates the exact clinical appearance in published radiology papers.
    """
    # 1. Confine the heatmap strictly inside the lung mask
    masked_cam = cam * lung_mask
    if np.max(masked_cam) > 0:
        masked_cam = masked_cam / np.max(masked_cam)

    # 2. Color mapping (JET: Blue=Normal/Cool, Green/Yellow=Moderate, Red=High Disease Activation)
    heat_color = cv2.applyColorMap(np.uint8(255 * masked_cam), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB) / 255.0

    # 3. Base image in float [0, 1]
    base_img = original_rgb.astype(np.float32) / 255.0

    # 4. Soft anatomical purple wash for background suppression
    purple_wash = np.array([0.22, 0.08, 0.52], dtype=np.float32)
    dimmed_background = base_img * 0.45 + purple_wash * 0.55

    # 5. Blend lung interior: 52% X-ray anatomical structure + 48% heat activation
    lung_interior = 0.52 * base_img + 0.48 * heat_color

    # 6. Smooth composite
    mask_3d = np.repeat(lung_mask[:, :, np.newaxis], 3, axis=2)
    composite = lung_interior * mask_3d + dimmed_background * (1.0 - mask_3d)

    return np.clip(composite, 0.0, 1.0)


def generate_paper_gradcam_figure(
    model: tf.keras.Model,
    image_paths: list[Path | str],
    output_path: Path | str = "paper_gradcam_figure.png",
) -> None:
    """Generates the multi-row, dual-column figure matching the user's reference:

    Columns: 'Original'  |  'GradCAM'
    Rows: (A), (B), (C), (D)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)

    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.6 * n), squeeze=False)
    letters = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]

    for i, path in enumerate(image_paths):
        # 1. Load clean image for visualization
        raw = cv2.imread(str(path))
        if raw is None:
            raise ValueError(f"Cannot read image: {path}")
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw_resized = cv2.resize(raw, IMG_SIZE)

        # 2. Preprocess copy for model prediction
        x_pre = tf.keras.applications.densenet.preprocess_input(raw_resized.copy().astype(np.float32))
        x_tensor = tf.expand_dims(x_pre, axis=0)

        # 3. Lung mask, Grad-CAM, and clinical overlay
        mask = extract_lung_mask(raw_resized)
        cam, score = compute_gradcam_heatmap(model, x_tensor)
        overlay = render_clinical_gradcam(raw_resized, cam, mask)

        # Left Column: Original
        axes[i, 0].imshow(raw_resized)
        axes[i, 0].axis("off")
        label = letters[i] if i < len(letters) else f"({i+1})"
        axes[i, 0].text(
            -0.12, 0.95, label,
            transform=axes[i, 0].transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="right",
        )

        # Right Column: GradCAM
        axes[i, 1].imshow(overlay)
        axes[i, 1].axis("off")

    # Column titles
    axes[0, 0].set_title("Original", fontsize=15, fontweight="bold", pad=12)
    axes[0, 1].set_title("GradCAM", fontsize=15, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Paper-style GradCAM figure generated: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clinical Lung-Masked Grad-CAM.")
    parser.add_argument("--model-path", type=Path, default=Path("best_tb_densenet_model.keras"))
    parser.add_argument("--images", nargs="+", type=Path, help="Paths to chest X-ray images.")
    parser.add_argument("--output", type=Path, default=Path("paper_gradcam_figure.png"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.images:
        # Default test images if none specified
        sample_pool = list(Path("new_test_images").glob("*.*"))
        if not sample_pool:
            sample_pool = list(Path("dataset/test").rglob("*.*"))
        args.images = sample_pool[:4]

    model = tf.keras.models.load_model(args.model_path)
    generate_paper_gradcam_figure(model, args.images, args.output)
