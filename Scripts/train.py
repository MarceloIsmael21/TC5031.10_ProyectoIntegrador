from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
# 1. Configuración: ingresa las ubicaciones al ejecutar la celda
# ------------------------------------------------------------------
current = Path.cwd()
#IMAGE_DIR= Path("../DataCapture/raw")
#ANNOTATIONS_CSV= Path("../DataCapture/anotaciones.csv")
#REVIEWED_CSV= Path("../DataCapture/revisado.csv")
IMAGE_DIR= current / "DataCapture/raw"
ANNOTATIONS_CSV= current / "DataCapture/anotaciones.csv"
REVIEWED_CSV= current / "DataCapture/revisado.csv"

#IMAGE_DIR = Path(input("Carpeta de imágenes originales completas: ").strip())
#ANNOTATIONS_CSV = Path(input("CSV con imagen, x, y, w, h: ").strip())
#REVIEWED_CSV = Path(input("CSV con una fila por imagen revisada: ").strip())

# Si tus columnas tienen otros nombres, cámbialos aquí.
IMAGE_COL = "imagen"
X_COL, Y_COL, W_COL, H_COL = "x", "y", "w", "h"

BATCH_SIZE = 2
EPOCHS = 10
RANDOM_STATE = 42


# ------------------------------------------------------------------
# 2. Cargar y validar anotaciones e imágenes revisadas
# ------------------------------------------------------------------

annotations = pd.read_csv(ANNOTATIONS_CSV)
reviewed = pd.read_csv(REVIEWED_CSV)

required_ann_cols = {IMAGE_COL, X_COL, Y_COL, W_COL, H_COL}
missing = required_ann_cols - set(annotations.columns)
if missing:
    raise ValueError(f"Faltan columnas en el CSV de anotaciones: {missing}")

if IMAGE_COL not in reviewed.columns:
    raise ValueError(f"El CSV de imágenes revisadas debe incluir '{IMAGE_COL}'.")

# En este ejemplo se espera que el CSV use nombres de archivo.
annotations[IMAGE_COL] = annotations[IMAGE_COL].astype(str).map(
    lambda name: Path(name).name
)
reviewed[IMAGE_COL] = reviewed[IMAGE_COL].astype(str).map(
    lambda name: Path(name).name
)

# Una fila por imagen en el manifiesto de imágenes revisadas.
reviewed = reviewed.drop_duplicates(subset=[IMAGE_COL]).reset_index(drop=True)

# Solo se admiten anotaciones de imágenes que estén en el manifiesto revisado.
unknown_images = set(annotations[IMAGE_COL]) - set(reviewed[IMAGE_COL])
if unknown_images:
    raise ValueError(
        "Hay anotaciones de imágenes que no aparecen en el CSV de revisadas. "
        f"Ejemplos: {list(unknown_images)[:5]}"
    )

# ------------------------------------------------------------------
# 3. Separación por imagen o por sesión para evitar fuga de datos
# ------------------------------------------------------------------

# Si el manifiesto tiene una columna 'session', las imágenes de una misma
# sesión permanecen juntas. Si no, se agrupa por imagen.
if "session" in reviewed.columns:
    groups = reviewed["session"].fillna(reviewed[IMAGE_COL]).astype(str)
else:
    groups = reviewed[IMAGE_COL].astype(str)

# 70 % entrenamiento, 15 % validación y 15 % prueba, aproximadamente.
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=RANDOM_STATE
)
trainval_idx, test_idx = next(
    split1.split(reviewed, groups=groups)
)

trainval = reviewed.iloc[trainval_idx].reset_index(drop=True)
test_df = reviewed.iloc[test_idx].reset_index(drop=True)

if "session" in trainval.columns:
    trainval_groups = (
        trainval["session"].fillna(trainval[IMAGE_COL]).astype(str)
    )
else:
    trainval_groups = trainval[IMAGE_COL].astype(str)

# 15 % del total equivale a ~17.6 % del bloque train+validation.
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=RANDOM_STATE
)
train_idx, val_idx = next(
    split2.split(trainval, groups=trainval_groups)
)

train_df = trainval.iloc[train_idx].reset_index(drop=True)
val_df = trainval.iloc[val_idx].reset_index(drop=True)

print(
    f"Imágenes: entrenamiento={len(train_df)}, "
    f"validación={len(val_df)}, prueba={len(test_df)}"
)


# ------------------------------------------------------------------
# 4. Dataset: cada imagen devuelve píxeles y cajas reales
# ------------------------------------------------------------------
class AccessoryDataset(Dataset):
    def __init__(self, manifest, annotations, image_dir):
        self.image_names = manifest[IMAGE_COL].tolist()
        self.image_dir = image_dir
        self.annotations_by_image = {
            name: group
            for name, group in annotations.groupby(IMAGE_COL)
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        rows = self.annotations_by_image.get(image_name)

        boxes = []
        if rows is not None:
            for row in rows.itertuples(index=False):
                x = float(getattr(row, X_COL))
                y = float(getattr(row, Y_COL))
                w = float(getattr(row, W_COL))
                h = float(getattr(row, H_COL))

                # CSV [x, y, w, h] -> Faster R-CNN [x1, y1, x2, y2].
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = min(float(image_width), x + w)
                y2 = min(float(image_height), y + h)

                if x2 <= x1 or y2 <= y1:
                    raise ValueError(
                        f"Caja inválida en {image_name}: "
                        f"{x}, {y}, {w}, {h}"
                    )

                boxes.append([x1, y1, x2, y2])

        # Una imagen revisada sin objetos tiene una matriz de cajas vacía.
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # 0 se reserva para fondo; 1 es la clase 'accesorio'.
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }

        # Convierte la imagen RGB a tensor float en [0, 1].
        image_tensor = TF.to_tensor(image)

        return image_tensor, target


def collate_fn(batch):
    # El detector admite imágenes de diferentes tamaños.
    return tuple(zip(*batch))


train_dataset = AccessoryDataset(train_df, annotations, IMAGE_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)


# ------------------------------------------------------------------
# 5. Crear Faster R-CNN preentrenado y reemplazar su clasificador
# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# Los pesos preentrenados pueden descargarse la primera vez.
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Dos clases contando el fondo: fondo=0, accesorio=1.
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)

model.to(device)


# ------------------------------------------------------------------
# 6. Entrenamiento
# ------------------------------------------------------------------
train_losses = []
val_losses = []
#MODELS_DIR = Path("../Models")
MODELS_DIR = current / "Models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# DataLoader de validación
val_dataset = AccessoryDataset(val_df, annotations, IMAGE_DIR)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

best_val_loss = float("inf")
best_model_state = None

EPOCH3_PATH = MODELS_DIR / "best_fit.pth"
BEST_PATH = MODELS_DIR / "fasterrcnn_accesorio_best_val.pth"

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

for epoch in range(EPOCHS):
    # -------------------------
    # Entrenamiento
    # -------------------------
    model.train()
    train_total = 0.0
    train_batches = 0

    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total += loss.detach().item()
        train_batches += 1

    mean_train_loss = train_total / max(train_batches, 1)
    train_losses.append(mean_train_loss)

    # -------------------------
    # Validación: calcular pérdidas, sin actualizar pesos
    # -------------------------
    # Torchvision devuelve las pérdidas cuando el detector está en modo train.
    # No se calculan gradientes ni se actualiza el optimizador.
    model.train()
    val_total = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            val_loss_dict = model(images, targets)
            val_loss = sum(val_loss_dict.values())

            val_total += val_loss.item()
            val_batches += 1

    mean_val_loss = val_total / max(val_batches, 1)
    val_losses.append(mean_val_loss)

    epoch_num = epoch + 1  # Python cuenta desde 0; la época visible empieza en 1.

    if epoch_num == 3:
        torch.save(
            {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
            },
            EPOCH3_PATH
        )
        print(f"Checkpoint de época 3 guardado: {EPOCH3_PATH.name}")

    scheduler.step()

    print(
        f"Época {epoch + 1:02d}/{EPOCHS} | "
        f"train loss: {mean_train_loss:.4f} | "
        f"val loss: {mean_val_loss:.4f}"
    )

    # Conserva el modelo de la época con menor pérdida de validación.
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        best_model_state = copy.deepcopy(model.state_dict())

# Restaurar el mejor punto según validación, no necesariamente la última época.
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# ------------------------------------------------------------------
# 7. Guardar el modelo entrenado
# ------------------------------------------------------------------
checkpoint_name="train.pth"

if not checkpoint_name:
    checkpoint_name = "fasterrcnn_accesorio.pth"

torch.save(model.state_dict(), checkpoint_name)
print("Modelo guardado.")


# ------------------------------------------------------------------
# 8. Probar una imagen y convertir las cajas a [x, y, w, h]
# ------------------------------------------------------------------

model.eval()

test_name = test_df.iloc[0][IMAGE_COL]
test_image = Image.open(IMAGE_DIR / test_name).convert("RGB")
test_tensor = TF.to_tensor(test_image).to(device)

with torch.no_grad():
    prediction = model([test_tensor])[0]

# Umbral inicial de confianza; después ajústalo usando validación.
SCORE_THRESHOLD = 0.50
keep = prediction["scores"] >= SCORE_THRESHOLD

pred_boxes = prediction["boxes"][keep].cpu()
pred_scores = prediction["scores"][keep].cpu()
pred_labels = prediction["labels"][keep].cpu()

detections = []
for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
    x1, y1, x2, y2 = box.tolist()

    detections.append({
        "clase": "accesorio",
        "x": x1,
        "y": y1,
        "w": x2 - x1,
        "h": y2 - y1,
        "confianza": float(score),
    })

detections_df = pd.DataFrame(detections)