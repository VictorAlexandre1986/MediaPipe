import cv2
import mediapipe as mp

def detect_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize the MediaPipe face detection model
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    results = face_detection.process(image_rgb)

    # Draw bounding boxes around the detected faces
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the image with the bounding boxes
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image you want to detect faces in
image_path = "/path/to/your/image.jpg"

# Call the detect_faces function
detect_faces(image_path)