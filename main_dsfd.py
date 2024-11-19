import torch
import cv2
import numpy as np
from dsfd import dsfd
# from dsfd.utils import draw_predict

def fgsm_attack(image_tensor, epsilon, data_grad):
    """
    Perform the FGSM attack by adding a small perturbation to the input image.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    # Clamp the image to be in valid range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def detect_and_attack(image_path, epsilon=0.05, confidence_threshold=0.5, nms_threshold=0.3):
    # Load and preprocess the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Set device to CPU
    device = torch.device("cpu")

    # Convert image to tensor
    image_tensor = torch.from_numpy(original_image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad = True

    print(f"[DEBUG] Initial image_tensor grad_fn: {image_tensor.grad_fn}")

    # Perform face detection and get raw detections
    detections = dsfd.detect_faces(image_tensor, confidence_threshold, nms_threshold)

    # Check if detections are made
    if detections is None or len(detections) == 0:
        print("No faces detected in the original image.")
        return

    print(f"[DEBUG] detections grad_fn: {detections.grad_fn if isinstance(detections, torch.Tensor) else 'None'}")

    # Extract confidence scores directly from the detection tensor
    confidences = detections[:, 0]  # Assuming confidences are in the first column

    # Calculate a loss based on the detection confidence scores
    loss = -torch.sum(confidences)
    loss.backward()

    # Check if gradients are computed
    print(f"[DEBUG] image_tensor.grad: {image_tensor.grad}")
    if image_tensor.grad is None:
        print("Error: Gradients not computed.")
        return

    # Get the gradients and generate the adversarial example
    data_grad = image_tensor.grad.data
    perturbed_image_tensor = fgsm_attack(image_tensor, epsilon, data_grad)

    # Convert the perturbed image tensor back to a NumPy array
    perturbed_image = perturbed_image_tensor.squeeze().permute(1, 2, 0).detach().numpy() * 255
    perturbed_image = perturbed_image.astype(np.uint8)

    # Save the perturbed image
    output_path = image_path.replace(".jpg", "_perturbed.jpg")
    output_path += f"_eps_{epsilon}.jpg"
    cv2.imwrite(output_path, perturbed_image)
    print(f"Perturbed image saved at {output_path}")

    # Display the perturbed image
    cv2.imshow("Perturbed Image", perturbed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = 'dsfd/unnamed.jpg'
    for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
        detect_and_attack(image_path, epsilon=epsilon)
