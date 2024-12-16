from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def visualize_perturbation(original_image, adv_image):
    """
    Visualize the perturbation applied to the original image.
    """
    # Ensure images are the same size
    adv_image = adv_image.resize(original_image.size)

    original_np = np.array(original_image).astype(np.float32) / 255.0
    adv_np = np.array(adv_image).astype(np.float32) / 255.0
    perturbation = adv_np - original_np

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    plt.imshow(original_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # plt.title("Adversarial Image")
    plt.imshow(adv_np)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # plt.title("Perturbation")
    plt.imshow((perturbation + 0.5))  # Amplify perturbation for visibility
    plt.axis('off')

    plt.show()

# Load images
original_image = Image.open("image/SONG/SONG3.png")
adv_image = Image.open("image/SONG/NI-FGSM_S3FD/SONG3_eps0.05.png")

# Call the function
visualize_perturbation(original_image, adv_image)
