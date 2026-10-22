import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.vision_transformer import get_vit_model

CLASSES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
    'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'
]

def predict(image_path, model_path="best_vit_model.pth", device="cpu"):
    device = torch.device(device)
    model = get_vit_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()

    print(f"Predicted Disease: {CLASSES[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]*100:.2f}%")
    return CLASSES[pred_idx], probs[pred_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Tomato Leaf Disease using ViT-B/16")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--model", type=str, default="best_vit_model.pth", help="Checkpoint path")
    args = parser.parse_args()
    predict(args.image, args.model)
