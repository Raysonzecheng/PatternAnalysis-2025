import torch
from torchvision import transforms
from PIL import Image
from modules import SiameseNetwork
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(img1_path, img2_path):
    img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
    out1, out2 = model(img1, img2)
    dist = F.pairwise_distance(out1, out2).item()
    print(f"Distance: {dist:.4f}")
    return "Same class" if dist < 0.5 else "Different class"

# 示例
print(predict("test1.jpg", "test2.jpg"))
