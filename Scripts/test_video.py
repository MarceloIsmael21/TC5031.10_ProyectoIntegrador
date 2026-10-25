from pathlib import Path
import os

current_dir = os.getcwd()
model_path = os.path.join(current_dir, "Models", "best_fit")
#model_path = os.path.abspath(model_path)  # Normalize to absolute path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint_file = input("Escribe la ubicación del checkpoint .pth: ").strip()
#checkpoint_file = "../Models/best_fit"
checkpoint_file=model_path
print(checkpoint_file)

try:
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True
    )
except TypeError:
    # Compatibilidad con versiones anteriores de PyTorch.
    checkpoint = torch.load(checkpoint_file, map_location=device)

# Acepta el checkpoint con metadatos o un state_dict guardado directamente.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Reconstruye la misma arquitectura: Faster R-CNN, accesorio + fondo.
model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Modelo cargado en:", device)


CAMERA_INDEX = 0          # Prueba 1 si la cámara principal no es la correcta.
SCORE_THRESHOLD = 0.50    # Usa el umbral que seleccionaste en validación.
MAX_AREA_FRACTION = 0.35  # Usa tu filtro elegido; pon None para desactivarlo.

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "No se pudo abrir la webcam. Revisa CAMERA_INDEX y los permisos de cámara."
    )

frame_number = 0

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("No se pudo leer un fotograma de la webcam.")
            break

        frame_number += 1
        height, width = frame_bgr.shape[:2]

        # OpenCV entrega BGR; el modelo recibe un tensor RGB con valores [0, 1].
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(device)
        )

        with torch.no_grad():
            prediction = model([image_tensor])[0]

        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()

        # Se llena con las detecciones conservadas de este fotograma,
        # en formato [x, y, w, h].
        detections_frame = []

        for box, score, label in zip(boxes, scores, labels):
            if score < SCORE_THRESHOLD or label != 1:
                continue

            x1, y1, x2, y2 = box
            x1 = int(np.clip(round(x1), 0, width - 1))
            y1 = int(np.clip(round(y1), 0, height - 1))
            x2 = int(np.clip(round(x2), 0, width - 1))
            y2 = int(np.clip(round(y2), 0, height - 1))

            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue

            area_fraction = (box_width * box_height) / (width * height)

            # Filtro opcional para omitir cajas demasiado grandes.
            if (
                MAX_AREA_FRACTION is not None
                and area_fraction > MAX_AREA_FRACTION
            ):
                continue

            detections_frame.append({
                "x": x1,
                "y": y1,
                "w": box_width,
                "h": box_height,
                "confianza": float(score),
            })

            cv2.rectangle(
                frame_bgr,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            texto = f"accesorio {score:.2f}"
            cv2.putText(
                frame_bgr,
                texto,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame_bgr,
            "Presiona q para salir",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Detector en webcam", frame_bgr)

        # detections_frame contiene las cajas [x, y, w, h]
        # del fotograma actual, por si quieres inspeccionarlas:
        # print(frame_number, detections_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
