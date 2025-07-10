from ultralytics import YOLO
import torch


class CrackDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def train_detection_model(self, config):
        """Train YOLOv8 detection model"""
        model = YOLO(config['model_name']).to(self.device)

        train_args = {
            'data': config['data_yaml'],
            'epochs': config['epochs'],
            'imgsz': config['imgsz'],
            'batch': config['batch'],
            'name': config['exp_name'],
            'device': self.device,
            'augment': True,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 45,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.5
        }

        results = model.train(**train_args)
        return model


if __name__ == "__main__":
    detector = CrackDetector()

    CONFIG = {
        'model_name': 'yolov8n.pt',  # Detection model
        'data_yaml': 'crack_detection.yaml',
        'epochs': 100,
        'imgsz': 256,
        'batch': 16,
        'exp_name': 'crack_detection_v1'
    }

    print("Training detection model...")
    detector.train_detection_model(CONFIG)
    print(f"Training complete! Model saved in runs/detect/{CONFIG['exp_name']}")