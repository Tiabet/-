from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_average_ssim(original_folder, attack_folders):
    """
    Calculate average SSIM scores for each attack method and epsilon.
    """
    # Initialize storage for SSIM scores
    ssim_scores = defaultdict(lambda: defaultdict(list))
    epsilons = set()

    # List all original images (KANG1.png ~ KANG20.png)
    original_images = [f"KANG{i}.png" for i in range(1, 21)]

    for original_image_name in original_images:
        # Load original image
        original_image_path = os.path.join(original_folder, original_image_name)
        original_image = Image.open(original_image_path).convert('L')
        original_np = np.array(original_image)

        # Loop through attack folders
        for attack_folder in attack_folders:
            attack_method = os.path.basename(attack_folder)  # e.g., NI-FGSM_S3FD

            for attack_image_name in os.listdir(attack_folder):
                # Match corresponding attack images (e.g., KANG1_eps0.01.png)
                if original_image_name.split('.')[0] in attack_image_name:
                    epsilon = float(attack_image_name.split('_')[-1].replace('.png', '').replace('eps', ''))
                    epsilons.add(epsilon)

                    # Load attack image
                    attack_image_path = os.path.join(attack_folder, attack_image_name)
                    attack_image = Image.open(attack_image_path).convert('L')
                    attack_np = np.array(attack_image)

                    # Resize if dimensions do not match
                    if attack_np.shape != original_np.shape:
                        attack_image = attack_image.resize(original_image.size, Image.Resampling.LANCZOS)
                        attack_np = np.array(attack_image)

                    # Compute SSIM score
                    score, _ = ssim(original_np, attack_np, full=True)
                    ssim_scores[attack_method][epsilon].append(score)

    # Calculate average SSIM scores
    avg_ssim_scores = {method: {eps: np.mean(scores) for eps, scores in eps_dict.items()}
                       for method, eps_dict in ssim_scores.items()}
    return avg_ssim_scores, sorted(epsilons)


def plot_average_ssim(avg_ssim_scores, epsilons):
    """
    Plot average SSIM scores for each attack method across epsilon values.
    """
    plt.figure(figsize=(10, 6))
    for method, scores in avg_ssim_scores.items():
        y_vals = [scores[eps] for eps in epsilons]
        plt.plot(epsilons, y_vals, marker='o', label=method)

    plt.xlabel("Epsilon")
    plt.ylabel("Average SSIM Score")
    plt.title("Average SSIM Scores by Attack Method and Epsilon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Define paths
original_folder = "original/output"
attack_folders = [
    "NI-FGSM_S3FD/output",
    "MI-FGSM_MTCNN/output",
    "FGSM_DSFD/output",
    "FGSM_S3FD/output",
    "FGSM_MTCNN/output",
    "NI-FGSM_DSFD/output",
    "MI-FGSM_DSFD/output",
    "MI-FGSM_S3FD/output",
    "NI-FGSM_MTCNN/output"
]

# Calculate average SSIM scores
avg_ssim_scores, epsilons = calculate_average_ssim(original_folder, attack_folders)

# Plot average SSIM scores
plot_average_ssim(avg_ssim_scores, epsilons)
