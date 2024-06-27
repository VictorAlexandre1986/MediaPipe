import cv2
import mediapipe as mp
import numpy as np

# import cv2
# import mediapipe as mp

# # Inicializar Mediapipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Inicializar OpenCV para capturar vídeo
# cap = cv2.VideoCapture(0)

# # Função para detectar máscara
# def detect_mask(landmarks):
#     # Pontos ao redor do nariz e boca
#     nose_top = landmarks[1]  # ponto no topo do nariz
#     nose_bottom = landmarks[2]  # ponto na base do nariz
#     chin = landmarks[152]  # ponto no queixo
#     left_cheek = landmarks[234]  # ponto na bochecha esquerda
#     right_cheek = landmarks[454]  # ponto na bochecha direita
    
#     # Definir limites da máscara
#     mask_limit = (nose_bottom.y + nose_top.y) / 2

#     # Verificar se a máscara está cobrindo a região
#     if chin.y > mask_limit and left_cheek.y > mask_limit and right_cheek.y > mask_limit:
#         return True
#     return False

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         break

#     # Converter a imagem para RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Processar a imagem e detectar landmarks
#     results = face_mesh.process(image_rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Desenhar landmarks na face
#             mp_drawing.draw_landmarks(
#                 image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

#             # Detectar máscara
#             if detect_mask(face_landmarks.landmark):
#                 cv2.putText(image, 'Mask On', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(image, 'No Mask', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#     # Mostrar a imagem com a detecção
#     cv2.imshow('Mask Detector', image)

#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np

# # Inicializar Mediapipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Inicializar OpenCV para capturar vídeo
# cap = cv2.VideoCapture(0)

# # Função para detectar máscara
# def detect_mask(image, landmarks):
#     # Pontos ao redor do nariz e boca
#     points = [1, 2, 164, 0, 17, 78, 308, 13, 14, 87, 317, 82, 312]
#     mask_region = np.array([(landmarks[idx].x * image.shape[1], landmarks[idx].y * image.shape[0]) for idx in points], np.int32)
    
#     # Criar uma máscara para a região do nariz e boca
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [mask_region], 255)

#     # Extrair a região de interesse
#     roi = cv2.bitwise_and(image, image, mask=mask)
#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
#     # Calcular a média de intensidade na região de interesse
#     mean_intensity = cv2.mean(roi_gray, mask=mask)[0]

#     # Verificar se a intensidade média está abaixo de um certo limite
#     return mean_intensity < 100

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         break

#     # Converter a imagem para RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Processar a imagem e detectar landmarks
#     results = face_mesh.process(image_rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Desenhar landmarks na face
#             mp_drawing.draw_landmarks(
#                 image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

#             # Detectar máscara
#             if detect_mask(image, face_landmarks.landmark):
#                 cv2.putText(image, 'Mask On', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(image, 'No Mask', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#     # Mostrar a imagem com a detecção
#     cv2.imshow('Mask Detector', image)

#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# Inicializar Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar OpenCV para capturar vídeo
cap = cv2.VideoCapture(0)

# Função para detectar máscara
def detect_mask(image, landmarks):
    # Pontos ao redor do nariz e boca
    points = [1, 2, 164, 0, 17, 78, 308, 13, 14, 87, 317, 82, 312]
    mask_region = np.array([(landmarks[idx].x * image.shape[1], landmarks[idx].y * image.shape[0]) for idx in points], np.int32)
    
    # Criar uma máscara para a região do nariz e boca
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [mask_region], 255)

    # Extrair a região de interesse
    roi = cv2.bitwise_and(image, image, mask=mask)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Calcular a média de intensidade na região de interesse
    mean_intensity = cv2.mean(roi_gray, mask=mask)[0]

    # Verificar se a intensidade média está abaixo de um certo limite
    return mean_intensity < 100

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
            # Desenhar landmarks na face
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Detectar máscara
            if detect_mask(image, face_landmarks.landmark):
                cv2.putText(image, 'Mask On', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'No Mask', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar a imagem com a detecção
    cv2.imshow('Mask Detector', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()