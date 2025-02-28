from ultralytics import YOLO
import torch

# Check for CUDA device and set it
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = YOLO("yolo11n.pt").to(device)
    model.train(data="data.yaml", batch=2)
