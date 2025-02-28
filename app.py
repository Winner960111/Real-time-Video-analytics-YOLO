
import cv2, re
import torch
from ultralytics import YOLO
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the trained YOLO model
model = YOLO("best2.pt")  # Ensure the path is correct

# Load an image for testing
image_path = "./frames/frame_75750.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Process results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        # conf = box.conf[0].item()  # Confidence score
        # cls = int(box.cls[0].item())  # Class index

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropped_scoreboard = image[y1:y2, x1:x2]

# # Convert to grayscale
# gray = cv2.cvtColor(cropped_scoreboard, cv2.COLOR_BGR2GRAY)

# # Apply thresholding (adjust values if needed)
# _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)


# Save or show preprocessed image
# cv2.imwrite("processed_scoreboard.jpg", thresh)

# Load TrOCR model
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1", use_fast = True)
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# # Load the cropped scoreboard image
# # image = Image.open(thresh).convert("RGB")

# # Run OCR
# pixel_values = processor(images=cropped_scoreboard, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)
# text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# text = text.replace(" O ", " 0 ").replace(" o ", " 0 ")
# text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
# print("Detected Score:", text)

# # Regular expression to find team names followed by a number
# matches = re.findall(r'([A-Z]+)\s+(\d+)', text)

# # Convert matches into a dictionary
# team_scores = {team:int(score) for team, score in matches}

# # Print the extracted team scores
# print(team_scores)

cv2.imshow("Processed Scoreboard", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



