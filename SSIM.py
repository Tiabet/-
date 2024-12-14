from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def calculate_ssim_and_grouped_plot(original_image_path, comparison_image_paths):
    """
    Calculate SSIM scores and display them as a grouped bar chart based on attack method and epsilon.
    """
    # Load the original image and convert it to grayscale
    original_image = Image.open(original_image_path).convert('L')
    original_np = np.array(original_image)

    ssim_scores = []
    labels = []

    # Compare with each image
    for comp_image_path in comparison_image_paths:
        comparison_image = Image.open(comp_image_path).convert('L')
        comparison_np = np.array(comparison_image)

        # Resize comparison image to match the original image dimensions
        if comparison_np.shape != original_np.shape:
            comparison_image = comparison_image.resize(original_image.size, Image.Resampling.LANCZOS)
            comparison_np = np.array(comparison_image)

        # Calculate SSIM score
        score, _ = ssim(original_np, comparison_np, full=True)
        ssim_scores.append(score)

        # Extract meaningful label (method + epsilon)
        parts = comp_image_path.split('/')[-2:]  # Extract folder and filename
        method = parts[0]  # e.g., FGSM_DSFD
        epsilon = parts[1].split('_')[-1].replace('.png', '')  # e.g., eps0.01
        labels.append(f"{method}\n{epsilon}")

    # Grouped Bar Chart
    plt.figure(figsize=(12, 8))
    plt.bar(labels, ssim_scores, color='skyblue')
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("SSIM Score")
    # plt.title("SSIM Scores by Attack Method and Epsilon Value")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # Print SSIM scores
    for label, score in zip(labels, ssim_scores):
        print(f"{label}: {score:.4f}")


# File paths
original_image_path = "image/DONGWOOK/DONGWOOK1.png"
comparison_image_paths = [
    "image/DONGWOOK/FGSM_DSFD/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/FGSM_DSFD/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/FGSM_DSFD/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/FGSM_S3FD/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/FGSM_S3FD/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/FGSM_S3FD/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/FGSM_MTCNN/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/FGSM_MTCNN/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/FGSM_MTCNN/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/NI-FGSM_DSFD/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/NI-FGSM_DSFD/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/NI-FGSM_DSFD/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/NI-FGSM_S3FD/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/NI-FGSM_S3FD/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/NI-FGSM_S3FD/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/NI-FGSM_MTCNN/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/NI-FGSM_MTCNN/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/NI-FGSM_MTCNN/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/MI-FGSM_DSFD/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/MI-FGSM_DSFD/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/MI-FGSM_DSFD/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/MI-FGSM_S3FD/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/MI-FGSM_S3FD/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/MI-FGSM_S3FD/DONGWOOK1_eps0.05.png",
    "image/DONGWOOK/MI-FGSM_MTCNN/DONGWOOK1_eps0.01.png",
    "image/DONGWOOK/MI-FGSM_MTCNN/DONGWOOK1_eps0.03.png",
    "image/DONGWOOK/MI-FGSM_MTCNN/DONGWOOK1_eps0.05.png",
]

# Calculate SSIM and display grouped chart
calculate_ssim_and_grouped_plot(original_image_path, comparison_image_paths)
