import cv2
import torch
from transformers import pipeline
from ultralytics import YOLO

# Initialize YOLOv8 model from Hugging Face
def load_model():
    model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with a larger model if Jetson Nano's GPU can handle it
    return model

# Display processed frames
def display_frame(window_name, frame):
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
        return False
    return True

# Process frames with YOLOv8
def process_frame(model, frame):
    # Convert BGR to RGB for YOLO processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame)
    
    # Visualize the results
    annotated_frame = results[0].plot()
    return annotated_frame

# Main function to handle RTSP connection and YOLO inference
def main():
    rtsp_url = "rtsp://<Your_IP_Address>:8554/stream"  # Replace with your RTSP URL

    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = load_model()

    # Open RTSP stream
    print(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream.")
        return

    print("Processing video stream...")
    window_name = "YOLOv8 Object Detection"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame. Exiting...")
            break

        # Process the frame
        processed_frame = process_frame(model, frame)

        # Display the processed frame
        if not display_frame(window_name, processed_frame):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()