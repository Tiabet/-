import matplotlib.pyplot as plt

def visualize_perturbation(original_image, adv_image):
    """
    Visualize the perturbation applied to the original image.
    """
    original_np = np.array(original_image).astype(np.float32) / 255.0
    adv_np = np.array(adv_image).astype(np.float32) / 255.0
    perturbation = adv_np - original_np

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Adversarial Image")
    plt.imshow(adv_np)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Perturbation")
    # Amplify perturbation for visibility
    plt.imshow((perturbation + 0.5))
    plt.axis('off')

    plt.show()
