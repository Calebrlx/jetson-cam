import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
def load_model():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Use "yolov8n.pt" for the smallest model (better for Jetson Nano)
    return model

# Process a single frame with YOLOv8
def process_frame(model, frame):
    # Convert the frame to RGB format (YOLO requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model.predict(rgb_frame, imgsz=640)  # Adjust image size if necessary

    # Draw results on the frame
    annotated_frame = results[0].plot()

    return annotated_frame

# Main loop to handle RTSP stream and YOLO processing
def main():
    # Replace with your RTSP stream URL
    rtsp_url = "rtsp://10.0.0.80:8554/stream"

    print(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream.")
        return

    # Load the YOLO model
    model = load_model()

    # Create a window for displaying the output
    window_name = "YOLOv8 Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Processing video stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame. Exiting...")
            break

        # Process the frame
        processed_frame = process_frame(model, frame)

        # Display the processed frame
        cv2.imshow(window_name, processed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("RTSP stream processing ended.")

if __name__ == "__main__":
    main()