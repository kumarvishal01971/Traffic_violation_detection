"""
Fixed Traffic Violation Detection System
- ONE violation per frame per type (no duplicates on single image)
- All 4 models active with correct class detection:
  * two-wheeler-aj3hv/2: Detects bikes/motorcycles
  * two-person-ride/11: Detects "No_Helmet" class
  * more-than-two-passengers/1: Detects "people" class (= 3+ passengers)
  * traffic-lights-ydbwt/1: Detects "0"=GREEN, "1"=RED
- Direct trust in model predictions
"""

import os
import re
import cv2
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# OCR Libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    EASYOCR_AVAILABLE = False

# Inference client
try:
    from inference_sdk import InferenceHTTPClient
except Exception:
    InferenceHTTPClient = None


# Constants
EVIDENCE_FOLDER = 'violation_evidence'
TEMP_FRAME = 'temp_frame.jpg'


class SimpleViolationTracker:
    """Track violations to avoid duplicates"""

    def __init__(self, cooldown_seconds: float = 5.0):
        self.cooldown = cooldown_seconds
        self.violations = {}  # key -> {'timestamp', 'evidence_file'}

    def should_save(self, plate_number: str, violation_type: str, current_time: float) -> bool:
        """Check if we should save this violation (cooldown check)"""
        key = f"{plate_number}_{violation_type}"
        
        if key not in self.violations:
            return True
        
        last_time = self.violations[key]['timestamp']
        return (current_time - last_time) >= self.cooldown

    def mark_saved(self, plate_number: str, violation_type: str, evidence_file: str, timestamp: float):
        """Mark a violation as saved"""
        key = f"{plate_number}_{violation_type}"
        self.violations[key] = {
            'timestamp': timestamp,
            'evidence_file': evidence_file
        }

    def get_summary(self) -> Dict:
        """Get violation count by type"""
        summary = defaultdict(int)
        for key in self.violations:
            violation_type = key.split('_', 1)[1] if '_' in key else key
            summary[violation_type] += 1
        return dict(summary)


class SimplifiedTrafficViolationDetector:
    """Fixed detector - ONE violation per frame, all models active"""

    def __init__(self, api_key: str = "BJLvln2FsWwIwjDBpoMI"):
        self.api_key = api_key
        self.api_url = "https://serverless.roboflow.com"
        self.client = None
        if InferenceHTTPClient is not None:
            try:
                self.client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
            except Exception:
                self.client = None

        self.models = {
            'two_wheeler': 'two-wheeler-aj3hv/2',
            'riders': 'two-person-ride/11',
            'triple_riding': 'more-than-two-passengers/1',
            'traffic_light': 'traffic-lights-ydbwt/1'
        }
        
        print("📋 Using models:")
        print(f"   • Two Wheeler: {self.models['two_wheeler']}")
        print(f"   • Riders: {self.models['riders']}")
        print(f"   • Triple Riding: {self.models['triple_riding']}")
        print(f"   • Traffic Light: {self.models['traffic_light']}")

        # Simple confidence thresholds
        self.confidence_thresholds = {
            'bike': 0.3,
            'no_helmet': 0.3,
            'triple': 0.3,
            'light': 0.3
        }

        # Plate cascade
        cascade_path = 'haarcascade_russian_plate_number.xml'
        if os.path.exists(cascade_path):
            self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.plate_cascade = None

        os.makedirs(EVIDENCE_FOLDER, exist_ok=True)
        self.evidence_folder = EVIDENCE_FOLDER
        self.tracker = SimpleViolationTracker(cooldown_seconds=5.0)

        print("✅ Simplified Detector Initialized")
        print(f"🔌 Tesseract: {'Available' if TESSERACT_AVAILABLE else 'Not Available'}")
        print(f"🔌 EasyOCR: {'Available' if EASYOCR_AVAILABLE else 'Not Available'}")

    def apply_ocr(self, image: np.ndarray) -> Optional[str]:
        """Apply OCR to extract plate number"""
        try:
            if image is None or image.size == 0:
                return None
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if TESSERACT_AVAILABLE:
                try:
                    text = pytesseract.image_to_string(
                        binary,
                        config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )
                    text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(text) >= 4:
                        return self.format_plate_text(text)
                except Exception:
                    pass

            if EASYOCR_AVAILABLE:
                try:
                    results = reader.readtext(binary, detail=0, paragraph=False)
                    if results:
                        text = ''.join(results).upper()
                        text = re.sub(r'[^A-Z0-9]', '', text)
                        if len(text) >= 4:
                            return self.format_plate_text(text)
                except Exception:
                    pass

            return None
        except Exception:
            return None

    def format_plate_text(self, text: str) -> Optional[str]:
        """Format plate text"""
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(text) < 4 or len(text) > 12:
            return None
        return f"{text[:2]} {text[2:]}"

    def extract_license_plate(self, frame: np.ndarray, bike_box: Dict) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extract license plate from bike region"""
        try:
            x, y, w, h = int(bike_box['x']), int(bike_box['y']), int(bike_box['width']), int(bike_box['height'])
            margin = max(20, int(min(w, h) * 0.4))
            x1 = max(0, x - w // 2 - margin)
            y1 = max(0, y - h // 2 - margin)
            x2 = min(frame.shape[1], x + w // 2 + margin)
            y2 = min(frame.shape[0], y + h // 2 + margin)
            bike_roi = frame[y1:y2, x1:x2]
            if bike_roi.size == 0:
                return None, None

            # Try plate cascade first
            if self.plate_cascade is not None:
                try:
                    gray_roi = cv2.cvtColor(bike_roi, cv2.COLOR_BGR2GRAY)
                    plates = self.plate_cascade.detectMultiScale(gray_roi, 1.1, 3)
                    if len(plates) > 0:
                        px, py, pw, ph = plates[0]
                        plate_img = bike_roi[py:py + ph, px:px + pw]
                        plate_text = self.apply_ocr(plate_img)
                        if plate_text:
                            return plate_img, plate_text
                except Exception:
                    pass

            # Try bottom region
            h_roi = bike_roi.shape[0]
            bottom_roi = bike_roi[int(h_roi * 0.6):, :]
            if bottom_roi.size > 0:
                plate_text = self.apply_ocr(bottom_roi)
                if plate_text:
                    return bottom_roi, plate_text

            return None, None
        except Exception:
            return None, None

    def analyze_frame_simple(self, predictions: List[Dict], frame: np.ndarray, frame_num: int) -> List[Dict]:
        """
        Simple violation detection based on model predictions:
        - If model says "No_Helmet" -> ONE violation per frame
        - If model says "people" (more-than-two-passengers model) -> ONE violation per frame  
        - If class="1" (red light from traffic light model) -> ONE violation per frame
        """
        violations = []
        
        # Extract different prediction types
        bikes = [p for p in predictions if p.get('confidence', 0) > self.confidence_thresholds['bike']]
        no_helmet_detections = [p for p in predictions if 'no' in p.get('class', '').lower() and 'helmet' in p.get('class', '').lower()]
        
        # Triple riding: "people" class from more-than-two-passengers model
        triple_detections = [p for p in predictions if p.get('class', '').lower() == 'people' and p.get('confidence', 0) > self.confidence_thresholds['triple']]
        
        # Red light: class="1" from traffic light model
        red_light_detections = [p for p in predictions if p.get('class', '') == '1' and p.get('confidence', 0) > self.confidence_thresholds['light']]
        
        print(f"\n🔍 Frame {frame_num}:")
        print(f"   Bikes detected: {len(bikes)}")
        print(f"   No_Helmet detections: {len(no_helmet_detections)}")
        print(f"   Triple riding (people) detections: {len(triple_detections)}")
        print(f"   Red light (class=1) detections: {len(red_light_detections)}")
        
        # Get ONE plate number for this frame (prefer actual plate over generated ID)
        plate_img, plate_number = None, None
        if bikes:
            plate_img, plate_number = self.extract_license_plate(frame, bikes[0])
        if not plate_number:
            plate_number = f"BIKE_{frame_num:04d}"
        
        # Check for NO HELMET violation - ONE per frame
        if no_helmet_detections:
            best_no_helmet = max(no_helmet_detections, key=lambda x: x.get('confidence', 0))
            
            violation = {
                'type': 'NO_HELMET',
                'frame': frame_num,
                'plate_number': plate_number,
                'confidence': best_no_helmet.get('confidence', 0),
                'details': f"Model detected: {best_no_helmet.get('class', 'No_Helmet')}",
                'bike_box': bikes[0] if bikes else None,
                'plate_img': plate_img,
                'detection': best_no_helmet
            }
            violations.append(violation)
            print(f"   🚨 NO_HELMET violation: {plate_number} (conf: {best_no_helmet.get('confidence', 0):.2f})")
        
        # Check for TRIPLE RIDING - ONE per frame
        if triple_detections:
            best_triple = max(triple_detections, key=lambda x: x.get('confidence', 0))
            
            violation = {
                'type': 'TRIPLE_RIDING',
                'frame': frame_num,
                'plate_number': plate_number,
                'confidence': best_triple.get('confidence', 0),
                'details': f"Model detected: {best_triple.get('class', 'people')} (More than 2 passengers)",
                'bike_box': bikes[0] if bikes else None,
                'plate_img': plate_img,
                'detection': best_triple
            }
            violations.append(violation)
            print(f"   🚨 TRIPLE_RIDING violation: {plate_number} (conf: {best_triple.get('confidence', 0):.2f})")
        
        # Check for RED LIGHT violation - ONE per frame
        if red_light_detections and bikes:
            best_red_light = max(red_light_detections, key=lambda x: x.get('confidence', 0))
            
            violation = {
                'type': 'RED_LIGHT_VIOLATION',
                'frame': frame_num,
                'plate_number': plate_number,
                'confidence': best_red_light.get('confidence', 0),
                'details': f"Red light violation (class={best_red_light.get('class', '1')})",
                'bike_box': bikes[0],
                'plate_img': plate_img,
                'detection': best_red_light
            }
            violations.append(violation)
            print(f"   🚨 RED_LIGHT violation: {plate_number} (conf: {best_red_light.get('confidence', 0):.2f})")
        
        return violations

    def save_evidence(self, frame: np.ndarray, violation: Dict) -> str:
        """Save evidence image"""
        plate_num = violation.get('plate_number', 'UNKNOWN')
        v_type = violation.get('type', 'VIOLATION')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        evidence = frame.copy()
        
        # Draw violation overlay
        if v_type == 'NO_HELMET':
            color = (0, 0, 255)
            label = "NO HELMET"
        elif v_type == 'TRIPLE_RIDING':
            color = (255, 0, 255)
            label = "TRIPLE RIDING"
        elif v_type == 'RED_LIGHT_VIOLATION':
            color = (0, 165, 255)
            label = "RED LIGHT"
        else:
            color = (0, 255, 0)
            label = v_type

        label_text = f"{label} - {plate_num}"
        cv2.rectangle(evidence, (5, 5), (5 + 10 + len(label_text) * 12, 45), color, -1)
        cv2.putText(evidence, label_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if 'details' in violation:
            cv2.putText(evidence, violation['details'], (10, evidence.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        conf = violation.get('confidence', 0.0)
        cv2.putText(evidence, f"Conf: {conf:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw bike box
        if 'bike_box' in violation and violation['bike_box']:
            bike_box = violation['bike_box']
            x, y, w, h = int(bike_box['x']), int(bike_box['y']), int(bike_box['width']), int(bike_box['height'])
            cv2.rectangle(evidence, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 3)

        # Draw detection box if available
        if 'detection' in violation:
            det = violation['detection']
            dx, dy, dw, dh = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
            cv2.rectangle(evidence, (dx - dw // 2, dy - dh // 2), (dx + dw // 2, dy + dh // 2), (0, 255, 255), 2)
            cv2.putText(evidence, det.get('class', ''), (dx - dw // 2, dy - dh // 2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        safe_plate = plate_num.replace(' ', '_')
        filename = f"{v_type}_{safe_plate}_{timestamp}.jpg"
        filepath = os.path.join(self.evidence_folder, filename)
        cv2.imwrite(filepath, evidence)

        # Save plate image separately if available
        if 'plate_img' in violation and violation['plate_img'] is not None:
            plate_img = violation['plate_img']
            if plate_img.size > 0:
                plate_filename = f"PLATE_{safe_plate}_{timestamp}.jpg"
                plate_filepath = os.path.join(self.evidence_folder, plate_filename)
                cv2.imwrite(plate_filepath, plate_img)

        return filename

    def detect_on_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> Tuple[np.ndarray, List[Dict]]:
        """Run detection on a single frame"""
        try:
            cv2.imwrite(TEMP_FRAME, frame)
            if self.client is None:
                return frame.copy(), []
                
            # Get all model predictions
            wheeler_result = self.client.infer(TEMP_FRAME, model_id=self.models['two_wheeler'])
            rider_result = self.client.infer(TEMP_FRAME, model_id=self.models['riders'])
            triple_result = self.client.infer(TEMP_FRAME, model_id=self.models['triple_riding'])
            light_result = self.client.infer(TEMP_FRAME, model_id=self.models['traffic_light'])
        except Exception as e:
            if os.path.exists(TEMP_FRAME):
                os.remove(TEMP_FRAME)
            print(f"⚠️ API Error: {e}")
            return frame.copy(), []

        # Combine all predictions
        def _predictions(res):
            if isinstance(res, dict):
                return res.get('predictions', []) or []
            if isinstance(res, list):
                return res
            return []

        all_predictions = []
        all_predictions.extend(_predictions(wheeler_result))
        all_predictions.extend(_predictions(rider_result))
        all_predictions.extend(_predictions(triple_result))
        all_predictions.extend(_predictions(light_result))

        # Analyze for violations
        violations = self.analyze_frame_simple(all_predictions, frame, frame_num)

        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw all predictions for visualization
        for pred in all_predictions:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            conf = pred.get('confidence', 0)
            cls = pred.get('class', 'unknown')
            cls_lower = cls.lower()
            
            # Color based on class
            if 'no' in cls_lower and 'helmet' in cls_lower:
                color = (0, 0, 255)  # Red for no helmet
                display_label = f"No Helmet: {conf:.2f}"
            elif 'helmet' in cls_lower:
                color = (0, 255, 0)  # Green for helmet
                display_label = f"{cls}: {conf:.2f}"
            elif cls_lower == 'people':  # Triple riding detection
                color = (255, 0, 255)  # Magenta for more than 2 passengers
                display_label = f"3+ Passengers: {conf:.2f}"
            elif cls == '1':  # Red light
                color = (0, 0, 255)  # Red for red light
                display_label = f"Red Light: {conf:.2f}"
            elif cls == '0':  # Green light
                color = (0, 255, 0)  # Green for green light
                display_label = f"Green Light: {conf:.2f}"
            elif 'bike' in cls_lower or 'motorcycle' in cls_lower or 'two' in cls_lower:
                color = (255, 255, 0)  # Yellow for bikes
                display_label = f"{cls}: {conf:.2f}"
            else:
                color = (128, 128, 128)  # Gray for others
                display_label = f"{cls}: {conf:.2f}"
            
            cv2.rectangle(annotated_frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
            cv2.putText(annotated_frame, display_label, (x - w // 2, y - h // 2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add violations to annotated frame with timestamp
        for v in violations:
            v['timestamp'] = timestamp

        if os.path.exists(TEMP_FRAME):
            os.remove(TEMP_FRAME)

        return annotated_frame, violations

    def process_image(self, image_path: str, output_path: str = 'output_image.jpg') -> Dict:
        """Process a single image"""
        print(f"\n📷 Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Error: Could not read image!")
            return {'error': 'Could not read image'}

        self.tracker = SimpleViolationTracker()
        annotated_frame, violations = self.detect_on_frame(frame, 0, 0.0)

        saved_violations = []
        for v in violations:
            plate = v.get('plate_number', 'UNKNOWN')
            vtype = v.get('type', 'VIOLATION')
            
            if self.tracker.should_save(plate, vtype, time.time()):
                evidence_file = self.save_evidence(frame, v)
                v['evidence_file'] = evidence_file
                self.tracker.mark_saved(plate, vtype, evidence_file, time.time())
                saved_violations.append(v)

        cv2.imwrite(output_path, annotated_frame)
        
        report = {
            'input': image_path,
            'output': output_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_violations': len(saved_violations),
            'violations': saved_violations,
            'status': 'success'
        }
        
        print(f"✅ Output saved: {output_path}")
        print(f"✅ Total violations: {len(saved_violations)}")
        return report

    def process_video(self, video_path: str, output_path: str = 'output_video.mp4') -> Dict:
        """Process video"""
        print(f"\n🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video!")
            return {'error': 'Could not open video', 'status': 'failed'}

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / original_fps if original_fps > 0 else 0.0

        # Process every 3rd frame
        frame_skip = 3

        print(f"📊 Video info: {total_frames} frames, {original_fps:.1f} FPS, {duration:.1f}s")
        print(f"⚙️ Processing every {frame_skip} frames")

        # VideoWriter setup
        vwf = getattr(cv2, 'VideoWriter_fourcc', None)
        fourcc = 0
        if vwf is not None:
            try:
                fourcc = vwf(*'mp4v')
            except Exception:
                try:
                    fourcc = vwf(*'XVID')
                except Exception:
                    fourcc = 0
        
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return {'error': 'Could not create video writer', 'status': 'failed'}

        self.tracker = SimpleViolationTracker(cooldown_seconds=5.0)
        frame_num = 0
        saved_violations = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every Nth frame
                if frame_num % frame_skip == 0:
                    timestamp = frame_num / original_fps if original_fps > 0 else 0.0
                    annotated_frame, violations = self.detect_on_frame(frame, frame_num, timestamp)
                    out.write(annotated_frame)

                    # Save violations
                    for v in violations:
                        plate = v.get('plate_number', 'UNKNOWN')
                        vtype = v.get('type', 'VIOLATION')
                        
                        if self.tracker.should_save(plate, vtype, timestamp):
                            evidence_file = self.save_evidence(frame, v)
                            v['evidence_file'] = evidence_file
                            self.tracker.mark_saved(plate, vtype, evidence_file, timestamp)
                            saved_violations.append(v)

                    if frame_num % 100 == 0:
                        progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
                        print(f"⏳ Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)")
                else:
                    out.write(frame)

                frame_num += 1

            summary = self.tracker.get_summary()
            
            report = {
                'input': video_path,
                'output': output_path,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_frames': total_frames,
                'total_violations': len(saved_violations),
                'violations': saved_violations,
                'summary': summary,
                'status': 'success'
            }
            
            print("\n" + "="*70)
            print("✅ VIDEO PROCESSING COMPLETED")
            print("="*70)
            print(f"📁 Output: {output_path}")
            print(f"📊 Total Violations: {len(saved_violations)}")
            
            if summary:
                print("\n📈 Violation Summary:")
                for vtype, count in summary.items():
                    print(f"   • {vtype}: {count}")
            
            print("="*70 + "\n")
            
            return report

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'status': 'failed',
                'violations': saved_violations
            }
        
        finally:
            print("🧹 Cleaning up resources...")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("✅ Resources released")


# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fixed Traffic Violation Detector")
    parser.add_argument('--video', help='Path to input video')
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--out', help='Output path', default=None)
    args = parser.parse_args()

    detector = SimplifiedTrafficViolationDetector()
    
    if args.image:
        outp = args.out or 'output_image.jpg'
        report = detector.process_image(args.image, outp)
        print("\n" + json.dumps(report, indent=2))
    elif args.video:
        outp = args.out or 'output_video.mp4'
        report = detector.process_video(args.video, outp)
        print("\n" + json.dumps(report, indent=2))
    else:
        print("Usage: python app.py --video path/to/video.mp4")
        print("   or: python app.py --image path/to/image.jpg")