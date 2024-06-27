import cv2
import mediapipe as mp

# Inicializar Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar OpenCV para capturar vídeo
cap = cv2.VideoCapture(0)

# Função para calcular a relação entre distâncias
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

# Função para detectar sorriso
def detect_smile(landmarks):
    # Pontos ao redor da boca
    points = [61, 291, 78, 308, 13, 14, 82, 312]

    # Distâncias horizontais
    mouth_width_top = calculate_distance(landmarks[78], landmarks[308])
    mouth_width_bottom = calculate_distance(landmarks[82], landmarks[312])
    
    # Distâncias verticais
    lip_height_left = calculate_distance(landmarks[61], landmarks[13])
    lip_height_right = calculate_distance(landmarks[291], landmarks[14])
    
    # Média das distâncias
    mouth_width = (mouth_width_top + mouth_width_bottom) / 2
    lip_height = (lip_height_left + lip_height_right) / 2

    smile_ratio = mouth_width / lip_height
    print(smile_ratio)
    return smile_ratio < 1.06  # Definir um limiar para considerar um sorriso

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Converter a imagem para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processar a imagem e detectar landmarks
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Desenhar landmarks ao redor da boca
            for idx in [61, 291, 78, 308, 13, 14, 82, 312]:
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Detectar sorriso
            if detect_smile(face_landmarks.landmark):
                cv2.putText(image, 'Sorrindo', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Nao sorrindo', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar a imagem com a detecção
    cv2.imshow('Smile Detector', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
