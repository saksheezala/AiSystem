# pgd_tiny.py
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils_tiny import load_tiny_imagenet
from train_tiny import load_model

def pgd_attack(model, image, label, epsilon, alpha, num_iter):
    """
    Perform a PGD attack on the input image.
    
    Parameters:
        model: The classification model.
        image: The original input image tensor.
        label: The target label for the attack.
        epsilon: Maximum perturbation (in normalized space).
        alpha: Step size for each iteration.
        num_iter: Number of iterations.
        
    Returns:
        The perturbed (adversarial) image.
    """
    # Clone the original image to start the attack
    perturbed = image.clone().detach()
    perturbed.requires_grad = True

    for i in range(num_iter):
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, label)
        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data
        
        # Take a step in the direction of the gradient
        perturbed = perturbed + alpha * grad.sign()
        
        # Project the perturbation onto the epsilon-ball around the original image
        perturbation = torch.clamp(perturbed - image, min=-epsilon, max=epsilon)
        perturbed = image + perturbation
        
        # Clamp the values to ensure they are within the valid range (normalized space)
        perturbed = torch.clamp(perturbed, -1, 1)
        
        # Detach and re-enable gradients for the next iteration
        perturbed = perturbed.detach()
        perturbed.requires_grad = True

    return perturbed

def generate_pgd_examples(model, device, data_loader, epsilon=0.3, num_iter=40, alpha=0.01, num_batches=1):
    """
    Generate PGD adversarial examples for a few batches of data.
    
    Parameters:
        model: The classification model.
        device: Device (CPU or CUDA).
        data_loader: DataLoader to load test images.
        epsilon: Maximum perturbation.
        num_iter: Number of PGD iterations.
        alpha: Step size per iteration.
        num_batches: Number of batches to process.
        
    Returns:
        normal_images: Original images (tensor).
        adv_images: PGD adversarial images (tensor).
    """
    model.eval()
    normal_images_list = []
    adv_images_list = []
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # For attack target, we can use the model's prediction (or the true label)
        outputs = model(inputs)
        pred_label = outputs.argmax(dim=1)
        
        # Generate adversarial examples using PGD
        adv = pgd_attack(model, inputs, pred_label, epsilon, alpha, num_iter)
        
        normal_images_list.append(inputs.detach().cpu())
        adv_images_list.append(adv.detach().cpu())
    
    normal_images = torch.cat(normal_images_list, dim=0)
    adv_images = torch.cat(adv_images_list, dim=0)
    
    return normal_images, adv_images

if __name__ == "__main__":
    # Testing the PGD attack generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = load_tiny_imagenet(batch_size=128, train=True)
    model = load_model().to(device)
    
    # Generate PGD adversarial examples with chosen hyperparameters
    normal_imgs, adv_imgs = generate_pgd_examples(
        model, device, test_loader, epsilon=0.3, num_iter=40, alpha=0.01, num_batches=1
    )
    
    print(f"Generated {normal_imgs.size(0)} normal and {adv_imgs.size(0)} PGD adversarial images.")
    
    # Optionally, save one pair for visual inspection
    os.makedirs("generated_images", exist_ok=True)
    from torchvision.transforms.functional import normalize, to_pil_image
    
    # For visualization, you may need to unnormalize the images.
    # Here we assume your normalization parameters are the same as before:
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.2770, 0.2691, 0.2821])
    
    # Unnormalize helper
    def unnormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    # Unnormalize and clamp the first image for display
    orig = normal_imgs[0].clone()
    adv = adv_imgs[0].clone()
    orig = unnormalize(orig, mean, std)
    adv = unnormalize(adv, mean, std)
    orig = torch.clamp(orig, 0, 1)
    adv = torch.clamp(adv, 0, 1)
    
    save_image(orig, os.path.join("generated_images", "pgd_original.png"))
    save_image(adv, os.path.join("generated_images", "pgd_adversarial.png"))
    print("Saved sample PGD images in 'generated_images' folder.")
