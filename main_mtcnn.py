import os
import glob
import torch
from torchvision import transforms
from MTCNN.mtcnn import MTCNN
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# FGSM 공격 함수
def fgsm_attack(image_tensor, epsilon, data_grad):
    """
    Perform the FGSM attack by adding a small perturbation to the input image.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# MI-FGSM 공격 함수
def mi_fgsm_attack(image_tensor, epsilon, alpha, momentum, num_iter, data_grad_func):
    """
    Perform the MI-FGSM attack using iterative perturbations with momentum.
    """
    perturbed_image = image_tensor.clone().detach().to(device)
    g = torch.zeros_like(image_tensor).to(device)  # Momentum accumulator

    perturbed_image.requires_grad = True
    # perturbed_image.retain_grad()


    for i in range(num_iter):
        print(i)
        # Compute the loss and gradients at the current image
        loss = data_grad_func(perturbed_image)
        loss.backward()
        data_grad = perturbed_image.grad.data

        # Update the gradient accumulator with momentum
        g = momentum * g + data_grad / torch.norm(data_grad, p=1)

        # Update the perturbed image using the sign of accumulated gradients
        perturbed_image.data = perturbed_image.data + alpha * g.sign()
        # Clamp the image to ensure perturbation remains in valid range
        # perturbed_image = torch.clamp(perturbed_image, image_tensor - epsilon, image_tensor + epsilon)
        perturbed_image.data.clamp_(image_tensor - epsilon, image_tensor + epsilon)

        perturbed_image.data.clamp_(0, 1)

        # Zero out gradients for the next iteration
        perturbed_image.grad.zero_()

    return perturbed_image


# NI-FGSM 공격 함수
def ni_fgsm_attack(image_tensor, epsilon, alpha, momentum, num_iter, data_grad_func):
    """
    Perform the NI-FGSM attack using Nesterov momentum and iterative perturbations.
    """
    perturbed_image = image_tensor.clone().detach().to(device)
    g = torch.zeros_like(image_tensor).to(device)  # Momentum accumulator

    perturbed_image.requires_grad = True

    for i in range(num_iter):
        print(i)
        # Compute the loss and gradients at the look-ahead point
        loss = data_grad_func(perturbed_image)
        loss.backward()
        # print(perturbed_image)
        data_grad = perturbed_image.grad.data
        print(data_grad)

        grad_norm = torch.mean(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        grad = data_grad / grad_norm
        # 모멘텀 업데이트
        momentum = g * momentum + grad

        # 이미지 업데이트 (in-place 연산 사용)
        perturbed_image.data = perturbed_image.data + alpha * torch.sign(momentum)
        perturbed_image.data.clamp_(image_tensor - epsilon, image_tensor + epsilon)
        perturbed_image.data.clamp_(0, 1)  # 클램핑을 in-place로 수행

        # Zero out gradients for the next iteration
        perturbed_image.grad.zero_()

    return perturbed_image

def detect_and_attack_all_images(image_path, epsilon=0.05, alpha=0.01, momentum=0.9, num_iter=3,
                                 attack_type="FGSM"):

    model_name = "MTCNN"
    output_dir = os.path.join(image_path, f"{attack_type}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running on device: {device}")
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=True,
        select_largest=True, keep_all=False, device=device
    )

    mtcnn.eval()

    # Load all images (supports common image formats)
    image_files = glob.glob(os.path.join(image_path, "*.[jp][pn]g"))  # Matches .jpg, .jpeg, .png
    if not image_files:
        return print(f"No images found in directory {image_path}")

    for idx, image_file in enumerate(image_files):
            # Load and preprocess image
            original_image = Image.open(image_file).convert("RGB")
            transform = transforms.ToTensor()
            image_tensor = transform(original_image).unsqueeze(0).to(device).requires_grad_(True)

            # Detect faces and probabilities
            img_input = image_tensor.permute(0, 2, 3, 1) * 255  # Convert to [batch, height, width, channels]
            faces, probs = mtcnn(img_input, return_prob=True)

            if faces is not None and len(faces) > 0:
                print(len(faces))
                face = faces[0]  # Using the first detected face
                prob = probs[0]
                print(f"Image {idx + 1}: Face detected with probability {prob}")

                # Define a closure for computing the loss
                def compute_loss(tensor):
                    faces, probs = mtcnn(tensor.permute(0, 2, 3, 1) * 255, return_prob=True)
                    if faces is None or len(faces) == 0:
                        return torch.tensor(0.0, device=device)
                    return -probs[0]

                # Select and perform the attack
                if attack_type == "FGSM":
                    loss = compute_loss(image_tensor)
                    loss.backward()
                    if image_tensor.grad is not None:
                        data_grad = image_tensor.grad.data
                        perturbed_tensor = fgsm_attack(image_tensor, epsilon, data_grad)
                    else:
                        print(f"Gradient not computed for image {image_file}. Skipping attack.")
                        continue

                elif attack_type == "MI-FGSM":
                    perturbed_tensor = mi_fgsm_attack(image_tensor, epsilon, alpha, momentum, num_iter, compute_loss)

                elif attack_type == "NI-FGSM":
                    perturbed_tensor = ni_fgsm_attack(image_tensor, epsilon, alpha, momentum, num_iter, compute_loss)

                else:
                    print(f"Unknown attack type: {attack_type}. Skipping image {image_file}.")
                    continue

                # Save the adversarial image
                adv_image = transforms.ToPILImage()(perturbed_tensor.squeeze(0).cpu())
                image_basename = os.path.basename(image_file)
                base, ext = os.path.splitext(image_basename)
                # Format epsilon to two decimal places
                epsilon_str = f"{epsilon:.2f}".rstrip('0').rstrip('.') if '.' in f"{epsilon:.2f}" else f"{epsilon:.2f}"
                adv_image_filename = f"{base}_eps{epsilon_str}{ext}"
                adv_image_path = os.path.join(output_dir, adv_image_filename)
                adv_image.save(adv_image_path)

            else:
                print(f"Image {idx + 1}: No face detected. Skipping attack.")

            # Clear gradients
            if image_tensor.grad is not None:
                image_tensor.grad.zero_()



if __name__ == "__main__":
    # Define the path to your image directory
    image_path = "image/JOO"

    # Define attack parameters
    epsilon_list = [0.01,0.03,0.05]
    alpha = 0.01
    momentum = 0.9
    num_iter = 10

    # Define the attack type: "FGSM", "MI-FGSM", or "NI-FGSM"
    # attack_type = "NI-FGSM"  # Change as needed

    for eps in epsilon_list:
        detect_and_attack_all_images(
            image_path=image_path,
            epsilon=eps,
            alpha=alpha,
            momentum=momentum,
            num_iter=num_iter,
            attack_type="NI-FGSM"
        ),
    detect_and_attack_all_images(
        image_path=image_path,
        epsilon=eps,
        alpha=alpha,
        momentum=momentum,
        num_iter=num_iter,
        attack_type="MI-FGSM"
    ),
    detect_and_attack_all_images(
        image_path=image_path,
        epsilon=eps,
        alpha=alpha,
        momentum=momentum,
        num_iter=num_iter,
        attack_type="FGSM")