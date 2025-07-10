import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from pathlib import Path


class ConcreteCrackAnalyzer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # Detection parameters
        self.detection_conf = 0.99  # Confidence threshold (0-1)
        self.min_box_area = 10  # Minimum detection box area (pixels)
        self.merge_iou_thresh = 0.4  # Threshold for merging nearby boxes

        # Crack validation parameters
        self.px_to_mm = 0.5  # Calibrate this! (0.1-0.5 for walls)
        self.min_crack_length = 25  # pixels (5-20 for walls)
        self.min_crack_width = 3  # pixels (1-3 for walls)
        self.min_crack_area = 75  # pixels (10-30 for walls)
        self.min_solidity = 0.001  # Shape consistency (0-1)

    def detect_and_merge_cracks(self, image_path):
        """Detect and merge cracks, returning the largest one"""
        # Initial detection
        results = self.model.predict(image_path, conf=self.detection_conf)
        if not results[0].boxes:
            return None, None, None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Filter small detections
        valid_indices = [
            i for i, box in enumerate(boxes)
            if (box[2] - box[0]) * (box[3] - box[1]) >= self.min_box_area
        ]
        boxes, confidences = boxes[valid_indices], confidences[valid_indices]

        # Merge nearby boxes
        if len(boxes) > 1:
            boxes, confidences = self.merge_close_boxes(boxes, confidences)

        # Select largest box
        largest_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        main_box = boxes[largest_idx]
        main_conf = confidences[largest_idx]

        return main_box, main_conf, results[0].orig_img

    def merge_close_boxes(self, boxes, confidences):
        """Merge nearby boxes that likely belong to the same crack"""
        iou_matrix = np.zeros((len(boxes), len(boxes)))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou_matrix[i, j] = self.calculate_iou(boxes[i], boxes[j])

        merged_boxes, merged_confs = [], []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue

            to_merge = [i] + [j for j in range(i + 1, len(boxes))
                              if iou_matrix[i, j] > self.merge_iou_thresh]

            merged_box = np.average(boxes[to_merge], axis=0, weights=confidences[to_merge])
            merged_conf = np.mean(confidences[to_merge])

            merged_boxes.append(merged_box)
            merged_confs.append(merged_conf)
            used.update(to_merge)

        return np.array(merged_boxes), np.array(merged_confs)

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-7)

    def extract_crack_region(self, image, box):
        """Extract and pad the crack region with 10% margin"""
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]

        # Calculate padding (10% of box dimensions)
        pad_x = int(0.1 * (x2 - x1))
        pad_y = int(0.1 * (y2 - y1))

        # Apply padding with boundary checks
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image[y1:y2, x1:x2], (x1, y1, x2, y2)

    def semantic_segmentation(self, cropped_image):
        """Perform semantic segmentation on the cropped crack region"""
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Clean up small artifacts
        cleaned = remove_small_objects(
            thresh.astype(bool),
            min_size=self.min_crack_area
        ).astype(np.uint8) * 255

        # Find the largest connected component
        labeled = label(cleaned)
        regions = regionprops(labeled)
        if not regions:
            return None

        largest_region = max(regions, key=lambda x: x.area)
        mask = np.zeros_like(cleaned)
        mask[labeled == largest_region.label] = 255

        return mask

    def measure_crack(self, mask, px_to_mm):
        """Calculate crack dimensions from segmentation mask"""
        skeleton = skeletonize(mask // 255)
        length_px = np.sum(skeleton)

        if length_px < self.min_crack_length:
            return None

        area_px = np.sum(mask) // 255
        width_px = area_px / length_px

        if width_px < self.min_crack_width:
            return None

        return {
            'length_mm': length_px * px_to_mm,
            'width_mm': width_px * px_to_mm,
            'area_px': area_px
        }

    def visualize_analysis(self, image, main_box, cropped_region, segmentation_mask, measurements):
        """Visualize all analysis steps"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original image with bounding box
        display = image.copy()
        x1, y1, x2, y2 = map(int, main_box)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        axes[0, 0].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Detected Crack Region')
        axes[0, 0].axis('off')

        # Cropped region
        axes[0, 1].imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Extracted Crack ROI')
        axes[0, 1].axis('off')

        # Segmentation mask
        axes[1, 0].imshow(segmentation_mask, cmap='gray')
        axes[1, 0].set_title('Segmentation Result')
        axes[1, 0].axis('off')

        # Measurements overlay
        overlay = cropped_region.copy()
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        if measurements:
            text = f"Length: {measurements['length_mm']:.1f}mm\nWidth: {measurements['width_mm']:.1f}mm"
            cv2.putText(overlay, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Measurements')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def analyze(self, image_path):
        """Complete analysis pipeline"""
        print(f"\nAnalyzing {Path(image_path).name}")

        # Step 1: Detect and merge cracks, get largest one
        main_box, confidence, image = self.detect_and_merge_cracks(image_path)
        if main_box is None:
            print("No cracks detected")
            return None

        # Step 2: Extract crack region with padding
        cropped_region, _ = self.extract_crack_region(image, main_box)

        # Step 3: Semantic segmentation
        segmentation_mask = self.semantic_segmentation(cropped_region)
        if segmentation_mask is None:
            print("Segmentation failed")
            return None

        # Step 4: Measurement
        measurements = self.measure_crack(segmentation_mask, self.px_to_mm)
        if not measurements:
            print("Measurement failed")
            return None

        print(f"\nCrack Measurements:")
        print(f"Length: {measurements['length_mm']:.1f} mm")
        print(f"Width: {measurements['width_mm']:.1f} mm")

        # Step 5: Visualization
        self.visualize_analysis(image, main_box, cropped_region, segmentation_mask, measurements)

        return measurements


if __name__ == "__main__":
    analyzer = ConcreteCrackAnalyzer(
        model_path="runs/detect/crack_detection_v1/weights/best.pt"
    )

    # Adjust parameters as needed
    analyzer.detection_conf = 0.9
    analyzer.px_to_mm = 0.2  # Calibrate for your camera setup

    # Run analysis
    result = analyzer.analyze("test_images/2.jpg")