import os
import glob
import cv2
import torch
from torchvision import transforms
from S3FD.data.config import cfg
from S3FD.s3fd_model import build_s3fd
from PIL import Image

# Define device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_s3fd_model(weights_path, device):
    """
    Load the S3FD model.
    """
    net = build_s3fd('attack', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.to(device)
    net.eval()
    return net


def fgsm_attack(image_tensor, epsilon, data_grad):
    """
    Perform the FGSM attack by adding a small perturbation to the input image.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def mi_fgsm_attack(image_tensor, epsilon, alpha, momentum, num_iter, data_grad_func):
    """
    Perform the MI-FGSM attack using iterative perturbations with momentum.
    """
    perturbed_image = image_tensor.clone().detach().to(device)
    g = torch.zeros_like(image_tensor).to(device)  # Momentum accumulator

    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        loss = data_grad_func(perturbed_image)
        loss.backward()
        data_grad = perturbed_image.grad.data

        # Update the gradient accumulator with momentum
        g = momentum * g + data_grad / (torch.norm(data_grad, p=1) + 1e-8)

        # Update the perturbed image using the sign of accumulated gradients
        perturbed_image = perturbed_image + alpha * g.sign()

        # Clamp the image to ensure perturbation remains within epsilon
        perturbed_image = torch.max(torch.min(perturbed_image, image_tensor + epsilon), image_tensor - epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

        # Zero out gradients for the next iteration
        if perturbed_image.grad is not None:
            perturbed_image.grad.zero_()

    return perturbed_image


def ni_fgsm_attack(image_tensor, epsilon, alpha, momentum, num_iter, data_grad_func):
    """
    Perform the NI-FGSM attack using Nesterov momentum and iterative perturbations.
    """
    perturbed_image = image_tensor.clone().detach().to(device)
    g = torch.zeros_like(image_tensor).to(device)  # Momentum accumulator

    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        loss = data_grad_func(perturbed_image)
        loss.backward()
        data_grad = perturbed_image.grad.data

        grad_norm = torch.mean(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        grad = data_grad / (grad_norm + 1e-8)  # Avoid division by zero

        # Update the momentum accumulator
        g = momentum * g + grad

        # Update the perturbed image using the sign of the momentum
        perturbed_image = perturbed_image + alpha * torch.sign(g)

        # Clamp the perturbation to be within the epsilon-ball
        perturbed_image = torch.max(torch.min(perturbed_image, image_tensor + epsilon), image_tensor - epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

        # Zero out gradients for the next iteration
        if perturbed_image.grad is not None:
            perturbed_image.grad.zero_()

    return perturbed_image


def compute_loss(tensor, s3fd_model, target_class=1):
    """
    Compute the loss based on S3FD's confidence scores for the target class.
    The loss is defined to minimize the confidence scores of detected faces.
    """
    decoded_boxes, conf_preds = s3fd_model(tensor)
    if conf_preds.size(0) == 0:
        return torch.tensor(0.0, device=device)
    # Extract confidence scores for the target class
    target_conf = conf_preds[:, target_class, :].max()
    return -target_conf


def attack_image(model, image_tensor, epsilon, attack_type, alpha=0.01, momentum=0.9, num_iter=10):
    """
    Perform the specified adversarial attack on a single image using S3FD.

    Args:
        model: The S3FD model.
        image_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].
        epsilon (float): Maximum perturbation.
        attack_type (str): Type of attack ("FGSM", "MI-FGSM", "NI-FGSM").
        alpha (float): Step size for iterative attacks.
        momentum (float): Momentum factor for iterative attacks.
        num_iter (int): Number of iterations for iterative attacks.

    Returns:
        torch.Tensor: Perturbed image tensor.
    """
    image_tensor.requires_grad = True

    # Forward pass
    decoded_boxes, conf_preds = model(image_tensor)

    # Check if any faces are detected
    if conf_preds.size(0) == 0:
        return image_tensor  # No faces detected; return original image

    # Compute loss
    loss = compute_loss(image_tensor, model)

    # Backward pass
    loss.backward()

    # Select and perform the attack
    if attack_type.upper() == "FGSM":
        if image_tensor.grad is not None:
            data_grad = image_tensor.grad.data
            perturbed_tensor = fgsm_attack(image_tensor, epsilon, data_grad)
        else:
            perturbed_tensor = image_tensor  # No gradients; return original image

    elif attack_type.upper() == "MI-FGSM":
        # Bind the s3fd_model to compute_loss using a lambda function
        perturbed_tensor = mi_fgsm_attack(
            image_tensor, epsilon, alpha, momentum, num_iter,
            lambda x: compute_loss(x, model)
        )

    elif attack_type.upper() == "NI-FGSM":
        # Bind the s3fd_model to compute_loss using a lambda function
        perturbed_tensor = ni_fgsm_attack(
            image_tensor, epsilon, alpha, momentum, num_iter,
            lambda x: compute_loss(x, model)
        )

    else:
        perturbed_tensor = image_tensor  # Unknown attack type; return original image

    return perturbed_tensor


def save_image(image_tensor, output_path):
    """
    Save the perturbed image tensor to a file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
    image_np = image_np.astype('uint8')
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_np)
    print(f"Saved perturbed image to: {output_path}")


def process_images(input_dir, attack_type, model_name, model, epsilon, alpha=0.01, momentum=0.9, num_iter=10):
    """
    Process all images in the input directory, perform attack, and save results.

    Args:
        input_dir (str): Path to the input image directory.
        attack_type (str): Type of attack ("FGSM", "MI-FGSM", "NI-FGSM").
        model_name (str): Name of the model (e.g., "s3fd").
        model: The S3FD model.
        epsilon (float): Maximum perturbation.
        alpha (float): Step size for iterative attacks.
        momentum (float): Momentum factor for iterative attacks.
        num_iter (int): Number of iterations for iterative attacks.
    """
    # Define output directory based on attack type and model name
    output_dir = os.path.join(input_dir, f"{attack_type.upper()}_{model_name.upper()}")
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    supported_extensions = ("*.jpg", "*.jpeg", "*.png")

    # Gather all image files with full paths
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    if not image_files:
        print(f"No images found in directory {input_dir}. Supported formats: {supported_extensions}")
        return

    for input_path in image_files:
        if not os.path.isfile(input_path):
            continue

        try:
            # Read and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Unable to read file: {input_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (300, 300))  # S3FD typically uses 300x300 images

            # Convert to tensor
            transform = transforms.ToTensor()
            image_tensor = transform(image_resized).unsqueeze(0).to(device)

            # Perform attack
            perturbed_tensor = attack_image(model, image_tensor, epsilon, attack_type, alpha, momentum, num_iter)

            # Generate output filename: original_filename_eps<epsilon>.<ext>
            original_name = os.path.splitext(os.path.basename(input_path))[0]
            ext = os.path.splitext(os.path.basename(input_path))[1]
            epsilon_str = f"{epsilon:.2f}".rstrip('0').rstrip('.') if '.' in f"{epsilon:.2f}" else f"{epsilon:.2f}"
            adv_image_filename = f"{original_name}_eps{epsilon_str}{ext}"
            output_path = os.path.join(output_dir, adv_image_filename)

            # Save adversarial image
            save_image(perturbed_tensor, output_path)

        except Exception as e:
            print(f"Error processing file {input_path}: {e}")


if __name__ == "__main__":
    # Define parameters and paths
    weights_path = "./S3FD/weights/sfd_face.pth"  # Path to S3FD weights
    input_directory = "image/JOO"  # Directory containing input images
    model_name = "s3fd"  # Name of the model
    epsilon_list = [0.01,0.03,0.05]
    alpha = 0.01  # Step size for iterative attacks
    momentum = 0.9  # Momentum factor for iterative attacks
    num_iter = 10  # Number of iterations for iterative attacks
    # attack_type = "NI-FGSM"  # Attack type: "FGSM", "MI-FGSM", or "NI-FGSM"

    # Load the S3FD model
    print(f"Using device: {device}")
    s3fd_model = load_s3fd_model(weights_path, device=device)
    for eps in epsilon_list :
        process_images(
            input_dir=input_directory,
            attack_type="FGSM",
            model_name=model_name,
            model=s3fd_model,
            epsilon=eps,
            alpha=alpha,
            momentum=momentum,
            num_iter=num_iter
        ),
        process_images(
            input_dir=input_directory,
            attack_type="MI-FGSM",
            model_name=model_name,
            model=s3fd_model,
            epsilon=eps,
            alpha=alpha,
            momentum=momentum,
            num_iter=num_iter
        ),
        process_images(
            input_dir=input_directory,
            attack_type="NI-FGSM",
            model_name=model_name,
            model=s3fd_model,
            epsilon=eps,
            alpha=alpha,
            momentum=momentum,
            num_iter=num_iter
        )
