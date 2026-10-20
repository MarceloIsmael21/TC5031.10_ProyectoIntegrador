import cv2
import time
from pathlib import Path
from datetime import datetime

# Configuración
CAMERA_INDEX = 0
COOLDOWN_SECONDS = 5.0
OUTPUT_DIR = Path("DataCapture/raw")

# Por precaución, no guardar una captura si el detector no encuentra
# ningún rostro. Esto no garantiza que haya detectado todos los rostros.
SAVE_ONLY_IF_FACE_DETECTED = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("No se pudo cargar el detector de rostros de OpenCV.")

cap = cv2.VideoCapture(CAMERA_INDEX)
# Configurar resolución deseada
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Alto

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la webcam.")

last_attempt = time.monotonic() - COOLDOWN_SECONDS
status = "Esperando captura..."

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un fotograma de la webcam.")
            break

        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Crear la imagen anonimizada; no se guarda el fotograma original.
        anonymized = frame.copy()

        for (x, y, w, h) in faces:
            # Ampliar la máscara alrededor del rostro detectado.
            margin_x = int(0.25 * w)
            margin_y = int(0.25 * h)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)

            # Rectángulo negro opaco.
            cv2.rectangle(
                anonymized,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                thickness=-1
            )

        now = time.monotonic()

        # Intentar guardar una nueva captura al cumplirse el intervalo.
        if now - last_attempt >= COOLDOWN_SECONDS:
            last_attempt = now

            if len(faces) > 0 or not SAVE_ONLY_IF_FACE_DETECTED:
                filename = datetime.now().strftime(
                    "captura_%Y%m%d_%H%M%S_%f.jpg"
                )
                output_path = OUTPUT_DIR / filename

                saved = cv2.imwrite(
                    str(output_path),
                    anonymized,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                status = (
                    f"Guardada: {filename}"
                    if saved
                    else "Error al guardar la captura"
                )
            else:
                status = "No guardada: no se detectó ningún rostro"

        # Mostrar una copia con el estado; el texto no se añade al archivo.
        preview = anonymized.copy()
        cv2.putText(
            preview,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )
        cv2.imshow("Vista anonimizada — pulsa q para salir", preview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
