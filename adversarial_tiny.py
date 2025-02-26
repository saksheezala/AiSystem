# adversarial_tiny.py
import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    diff = torch.abs(perturbed_image - image)
    print("Max perturbation difference:", diff.max().item())
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def generate_fgsm_examples(model, device, data_loader, epsilon=0.5, num_batches=1):
    model.eval()
    normal_images_list = []
    adv_images_list = []
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        
        # Debug: Print gradient norm for first image in batch
        grad_norm = data_grad[0].norm().item()
        print("Gradient norm for first image:", grad_norm)
        
        perturbed = fgsm_attack(inputs, epsilon, data_grad)
        normal_images_list.append(inputs.detach().cpu())
        adv_images_list.append(perturbed.detach().cpu())
    normal_images = torch.cat(normal_images_list, dim=0)
    adv_images = torch.cat(adv_images_list, dim=0)
    return normal_images, adv_images

if __name__ == "__main__":
    from utils_tiny import load_tiny_imagenet
    from train_tiny import load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # To this:
    test_loader = load_tiny_imagenet(batch_size=128, train=True)
    model = load_model().to(device)
    normal_imgs, adv_imgs = generate_fgsm_examples(model, device, test_loader, epsilon=0.5, num_batches=1)
    print(f"Generated {normal_imgs.size(0)} normal and {adv_imgs.size(0)} adversarial images.")
