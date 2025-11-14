import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

def get_train_transforms(image_size=(224,224)):
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1), shear=5),
        T.GaussianBlur(3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def get_val_test_transforms(image_size=(224,224)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# -- Demo visualization on a sample image --
if __name__ == "__main__":
    img_path = "Data/boys/pre-peak/101/u.jpg"  # Or any image path from your dataset
    img = Image.open(img_path).convert("RGB")
    transform = get_train_transforms()
    plt.figure(figsize=(12,6))
    for i in range(6):
        plt.subplot(2,3,i+1)
        aug_img = transform(img)
        # Undo normalization for viewing (multiply by std, add mean, then convert to [0,1] range)
        vis_img = aug_img.permute(1,2,0).detach().cpu().numpy()
        vis_img = vis_img * [0.229, 0.224, 0.225] + [0.485,0.456,0.406]  # Unnormalize
        vis_img = vis_img.clip(0,1)
        plt.imshow(vis_img)
        plt.axis("off")
    plt.suptitle("Sample Augmented Images")
    plt.show()
