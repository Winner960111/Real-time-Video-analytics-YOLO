import cv2
import os
import re
import torch
from ultralytics import YOLO
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
# Path to the video file
video_path = "video.mp4"
output_folder = "frames"

# Path to the video file
video_path = "video.mp4"
output_folder = "frames"

# Load the trained YOLO model
yolo_model = YOLO("best2.pt")  # Ensure the path is correct

# Load TrOCR processor and model once
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1", use_fast=True)
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

def getScore(frame):
    try:
         # Load image correctly
        image = frame
        if image is None:
            raise ValueError("Error loading image.")

        # Run detection
        results = yolo_model(image)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                # conf = box.conf[0].item()  # Confidence score
                # cls = int(box.cls[0].item())  # Class index

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cropped_scoreboard = image[y1:y2, x1:x2]

                                # Convert to PIL format for OCR
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_scoreboard, cv2.COLOR_BGR2RGB))

                # Run OCR
                pixel_values = trocr_processor(images=cropped_pil, return_tensors="pt").pixel_values
                generated_ids = trocr_model.generate(pixel_values)
                text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Cleanup text
                text = text.replace(" O ", " 0 ").replace(" o ", " 0 ")
                text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

                # print("Detected Score:", text)

                # Extract team scores using regex
                matches = re.findall(r'([A-Z]+)\s+(\d+)', text)
                team_scores = []

                if len(matches) == 2:
                    for team, score in matches:
                        if len(score) > 2:
                            return
                        team_scores.append({team: int(score)})
                print("Extracted Scores:", team_scores)
                return team_scores  # Return the first valid scoreboard's results
            
    except Exception as e:
        print("error--->", e)
        return
# Load the video
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break if no more frames

    frame_count += 1

    if frame_count % 150 == 0:
        score = getScore(frame)
        print("Final Score:", score)

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Extracted {frame_count} frames and saved in '{output_folder}'")