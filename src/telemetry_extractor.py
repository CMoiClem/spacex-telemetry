import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import yaml
import numpy as np
import cv2
import pytesseract
from natsort import natsorted
from datetime import datetime, timedelta
import csv

# Configure logging
logging.basicConfig(
    filename='telemetry_extractor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TelemetryExtractor:
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """Initialize with configuration."""
        self.previous_values = [[], [], [], [], []]
        self.config = self._load_config(config_path)
        
        # Convert ROI dict to list in correct order
        self.rois = [
            self.config['rois']['booster_speed'],
            self.config['rois']['booster_altitude'],
            self.config['rois']['timestamp'],
            self.config['rois']['ship_speed'],
            self.config['rois']['ship_altitude']
        ]
        
        # Load validation parameters
        self.validation_config = self.config['validation']
        self.confidence_threshold = self.config['confidence_threshold']
        pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_path']

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess region of interest for OCR."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def process_roi(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        """Process ROI and extract text with confidence."""
        processed = self.preprocess_roi(roi)
        data = pytesseract.image_to_data(processed, config='--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789:', output_type=pytesseract.Output.DICT)
        
        confidences = [float(conf) for conf in data['conf'] if conf != '-1']
        texts = [text for text, conf in zip(data['text'], data['conf']) if conf != '-1']
        
        if not confidences or not texts:
            return None, 0
        
        max_conf_idx = np.argmax(confidences)
        return texts[max_conf_idx].strip(), confidences[max_conf_idx]

    def validate_value(self, current_value: Optional[str], index: int, min_samples: int = 5) -> Optional[int]:
        """Validate extracted value based on context and previous values."""
        try:
            if current_value is None:
                if index == 1:
                    logger.info("  REJECTED: Value is None")
                return None
                
            current_value = int(current_value)
            valid_values = [v for v in self.previous_values[index] if v is not None]
            
            if len(self.previous_values[index]) == 0:
                if index == 1:
                    logger.info("  ACCEPTED: First value in sequence")
                self.previous_values[index].append(current_value)
                return current_value

            last_value = valid_values[-1]

            if index in [1, 4]:  # Altitude indices
                if index == 1:
                    logger.info("\n  VALIDATION START:")
                    logger.info(f"  Current value: {current_value}")
                    logger.info(f"  Last value: {last_value}")
                    logger.info(f"  Current digits: {len(str(current_value))}")
                    logger.info(f"  Last digits: {len(str(last_value))}")
                
                # Check 1: Reject single digits after double digits unless it's a natural descent
                if len(str(current_value)) == 1 and len(str(last_value)) == 2:
                    if current_value != last_value - 1:
                        if index == 1:
                            logger.info("  REJECTED: Single digit after double digit")
                        return None
                
                # Check 2: Allow larger decreases during descent
                if current_value < last_value:
                    max_drop = self.validation_config['altitude']['max_drop']
                    if abs(current_value - last_value) > max_drop:
                        if index == 1:
                            logger.info(f"  REJECTED: Drop too large")
                        return None
                else:
                    max_increase = self.validation_config['altitude']['max_increase']
                    if current_value - last_value > max_increase:
                        if index == 1:
                            logger.info(f"  REJECTED: Increase too large")
                        return None

            else:  # Speed validation
                max_range = self.validation_config['speed']['max_range']
                max_deviation = self.validation_config['speed']['max_deviation']
                
                if not 0 <= current_value <= max_range:
                    return None
                    
                if len(valid_values) >= min_samples:
                    recent_values = valid_values[-min_samples:]
                    mean = np.mean(recent_values)
                    std = np.std(recent_values)
                    if abs(current_value - mean) > max_deviation:
                        return None
                else:
                    if abs(current_value - last_value) > max_deviation:
                        return None

            if index == 1:
                logger.info("  ACCEPTED: Value passed all validation checks")
                
            self.previous_values[index].append(current_value)
            return current_value
                
        except (ValueError, TypeError):
            if index == 1:
                logger.info("  REJECTED: Value conversion error")
            return None

    def process_frames(self, frames_dir: Path, output_file: Path) -> None:
        """Process all frames in directory and save results."""
        frame_files = natsorted([f for f in frames_dir.glob('*.png')])
        initial_timestamp = datetime.strptime("00:00:00", '%H:%M:%S')

        # Create new CSV file with headers
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Stage 1 Speed", "Stage 1 Altitude", "Timestamp", "Stage 2 Speed", "Stage 2 Altitude"])

        # Process frames and append data
        for idx, frame_path in enumerate(frame_files, start=1):
            self.process_frame(frame_path, idx, initial_timestamp, output_file)

        logger.info(f"Processed {len(frame_files)} frames")

    def process_frame(self, frame_path: Path, frame_number: int, initial_timestamp: datetime, output_csv: Path) -> None:
        """Process a single frame and extract telemetry data."""
        image = cv2.imread(str(frame_path))
        if image is None:
            return

        row = []
        
        for i, (x, y, w, h) in enumerate(self.rois):
            if i == 2:  # Skip timestamp ROI index
                continue
                
            text, confidence = self.process_roi(image[y:y+h, x:x+w])
            
            if i == 1:
                frame_timestamp = (initial_timestamp + timedelta(seconds=frame_number - 1)).strftime('%H:%M:%S')
                logger.info(f"Frame {frame_number} ({frame_timestamp}) - OCR Text: '{text}', Confidence: {confidence:.2f}%")
            
            if confidence >= self.confidence_threshold:
                cleaned = ''.join(c for c in text if c.isdigit())
                value = self.validate_value(cleaned, i)
            else:
                if i == 1:
                    logger.info(f"  REJECTED: Confidence {confidence:.2f}% below threshold {self.confidence_threshold}%")
                value = None
            row.append(value)
            
            if i == 1:
                timestamp = (initial_timestamp + timedelta(seconds=frame_number - 1)).strftime('%H:%M:%S')
                row.append(timestamp)

        if any(v is not None for v in row):
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract telemetry data from Starship test flight frames')
    parser.add_argument('--frames-dir', type=str, required=True, help='Directory containing frame images')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_file = Path(args.output)
    config_file = Path(args.config)

    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        extractor = TelemetryExtractor(str(config_file))
        extractor.process_frames(frames_dir, output_file)
        logger.info(f"Processing completed. Output saved to {output_file}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
