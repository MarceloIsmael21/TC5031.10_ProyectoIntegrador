import csv
import os
from pathlib import Path

import cv2

actual_dir = Path(__file__).resolve().parent
#actual_dir = actual_dir.parent / "DataCapture"
IMAGE_DIR = actual_dir / "raw/"
CSV_OUT = actual_dir / "anotaciones.csv"
Revisado_OUT = actual_dir / "revisado.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Anotador de objetos"

# True: empieza en la primera imagen que aún no aparece en el CSV.
# False: comienza desde la primera imagen para permitir revisar todo.
RESUME_AT_FIRST_UNREVIEWED = True


class Annotator:
    def __init__(self, image_paths, csv_path, resume=True):
        self.image_paths = image_paths
        self.csv_path = csv_path
        self.columns = [
            "imagen", "x", "y", "w", "h", "ImageWidth", "ImageHeight"
        ]

        self.index = 0
        self.image = None
        self.image_name = None
        self.current_boxes = []
        self.first_point = None

        # Filas ya existentes y nombres de imágenes con algún registro.
        self.rows = []
        self.recorded_images = set()

        # Cajas recuperadas del CSV, por nombre de imagen.
        self.boxes_by_image = {}

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_csv()

        # Reanudar en la primera imagen sin registro.
        if resume:
            self.index = next(
                (
                    i for i, path in enumerate(self.image_paths)
                    if path.name not in self.recorded_images
                ),
                0,  # Si todas están registradas, permite recorrer desde el inicio.
            )

        self.load_current_image()

    def load_csv(self):
        """Carga los registros y reconstruye las cajas existentes."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.rows = []
            return

        with self.csv_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != self.columns:
                raise ValueError(
                    "Las columnas del CSV no coinciden con las esperadas.\n"
                    f"Esperadas: {self.columns}\n"
                    f"Encontradas: {reader.fieldnames}"
                )

            for raw_row in reader:
                row = {
                    col: (raw_row.get(col) or "").strip()
                    for col in self.columns
                }

                image_name = row["imagen"]
                if not image_name:
                    continue

                self.rows.append(row)
                self.recorded_images.add(image_name)

                # Una fila con x, y, w, h numéricos representa una caja.
                try:
                    x = int(float(row["x"]))
                    y = int(float(row["y"]))
                    w = int(float(row["w"]))
                    h = int(float(row["h"]))
                except (ValueError, TypeError):
                    # Por ejemplo: fila vacía que marca imagen sin detecciones.
                    continue

                if w <= 0 or h <= 0:
                    continue

                # El anotador original guardó w/h incluyendo el píxel final.
                x2 = x + w - 1
                y2 = y + h - 1

                self.boxes_by_image.setdefault(image_name, []).append(
                    (x, y, x2, y2)
                )

    def save_csv(self):
        """Reescribe el CSV con los registros actuales."""
        temp_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")

        with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

        # Reemplazo para evitar dejar un CSV a medio escribir.
        os.replace(temp_path, self.csv_path)

    def load_current_image(self):
        """Carga la imagen indicada por el índice y sus cajas previas."""
        self.image = None
        self.image_name = None
        self.current_boxes = []
        self.first_point = None

        while self.index < len(self.image_paths):
            path = self.image_paths[self.index]
            image = cv2.imread(str(path))

            if image is not None:
                self.image = image
                self.image_name = path.name
                self.current_boxes = list(
                    self.boxes_by_image.get(self.image_name, [])
                )
                return

            print(f"No se pudo leer; se omitirá: {path.name}")
            self.index += 1

    def on_mouse(self, event, x, y, flags, param):
        """Dos clics izquierdos definen las esquinas de una caja."""
        if event != cv2.EVENT_LBUTTONDOWN or self.image is None:
            return

        height, width = self.image.shape[:2]
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))

        if self.first_point is None:
            self.first_point = (x, y)
            return

        x0, y0 = self.first_point
        x1, y1 = min(x0, x), min(y0, y)
        x2, y2 = max(x0, x), max(y0, y)

        if x2 > x1 and y2 > y1:
            box_width = x2 - x1 + 1
            box_height = y2 - y1 + 1

            # Si la imagen tenía una fila vacía de "sin detecciones",
            # se elimina: ahora sí tiene una detección.
            self.rows = [
                row for row in self.rows
                if not (
                    row["imagen"] == self.image_name
                    and not row["x"] and not row["y"]
                    and not row["w"] and not row["h"]
                )
            ]

            self.current_boxes.append((x1, y1, x2, y2))

            row = {
                "imagen": self.image_name,
                "x": x1,
                "y": y1,
                "w": box_width,
                "h": box_height,
                "ImageWidth": width,
                "ImageHeight": height,
            }
            self.rows.append({col: str(row[col]) for col in self.columns})
            self.recorded_images.add(self.image_name)
            self.boxes_by_image[self.image_name] = list(self.current_boxes)

            # Guardado inmediato de cada caja completa.
            self.save_csv()

        self.first_point = None

    def render(self):
        """Dibuja las cajas existentes y nuevas sobre la imagen."""
        display = self.image.copy()

        for x1, y1, x2, y2 in self.current_boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.first_point is not None:
            cv2.drawMarker(
                display,
                self.first_point,
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )

        height = display.shape[0]

        cv2.putText(
            display,
            f"Imagen {self.index + 1}/{len(self.image_paths)}: "
            f"{self.image_name} | Cajas: {len(self.current_boxes)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "Dos clics: caja | N: siguiente | P: anterior | Q: salir",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        return display

    def next_image(self):
        """Avanza y registra una fila vacía solo si nunca se registró."""
        if self.image is None:
            return

        # Solo registra una fila vacía si no hay cajas y no existe ya
        # ningún registro para esta imagen.
        if (
            not self.current_boxes
            and self.image_name not in self.recorded_images
        ):
            height, width = self.image.shape[:2]
            row = {
                "imagen": self.image_name,
                "x": "",
                "y": "",
                "w": "",
                "h": "",
                "ImageWidth": width,
                "ImageHeight": height,
            }
            self.rows.append(row)
            self.recorded_images.add(self.image_name)
            self.save_csv()

        self.index += 1
        self.load_current_image()

    def previous_image(self):
        """Retrocede sin crear registros ni modificar las anotaciones."""
        if self.index > 0:
            self.index -= 1
            self.load_current_image()
    
def generar_revisado(csv_anotaciones, csv_revisado):
#"""Crea revisado.csv con una fila por imagen y su etiqueta de objeto."""
    orden_imagenes = []
    imagenes_vistas = set()
    imagenes_con_objeto = set()

    with csv_anotaciones.open(
        "r", newline="", encoding="utf-8-sig"
    ) as archivo:
        reader = csv.DictReader(archivo)

        columnas_requeridas = {"imagen", "x", "y", "w", "h"}
        faltantes = columnas_requeridas - set(reader.fieldnames or [])
        if faltantes:
            raise ValueError(
                f"Faltan columnas en {csv_anotaciones.name}: {sorted(faltantes)}"
            )

        for numero_fila, row in enumerate(reader, start=2):
            nombre = (row.get("imagen") or "").strip()
            if not nombre:
                continue

            if nombre not in imagenes_vistas:
                imagenes_vistas.add(nombre)
                orden_imagenes.append(nombre)

            valores = [
                (row.get(columna) or "").strip()
                for columna in ("x", "y", "w", "h")
            ]

            # Los cuatro campos vacíos representan una imagen sin objeto.
            if all(valor == "" for valor in valores):
                continue

            # Una caja debe tener sus cuatro valores.
            if any(valor == "" for valor in valores):
                raise ValueError(
                    f"Caja incompleta en fila {numero_fila}, imagen {nombre}"
                )

            try:
                x, y, ancho, alto = map(float, valores)
            except ValueError as error:
                raise ValueError(
                    f"Coordenadas no numéricas en fila {numero_fila}, "
                    f"imagen {nombre}"
                ) from error

            if ancho <= 0 or alto <= 0:
                raise ValueError(
                    f"Caja con ancho o alto no positivo en fila {numero_fila}, "
                    f"imagen {nombre}"
                )

            imagenes_con_objeto.add(nombre)

    csv_revisado.parent.mkdir(parents=True, exist_ok=True)

    with csv_revisado.open(
        "w", newline="", encoding="utf-8-sig"
    ) as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["imagen", "tiene_objeto"])
        writer.writerows(
            (nombre, int(nombre in imagenes_con_objeto))
            for nombre in orden_imagenes
        )

    return len(orden_imagenes)



def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta de imágenes: {IMAGE_DIR}"
        )

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            "No se encontraron imágenes compatibles en la carpeta."
        )

    annotator = Annotator(
        image_paths,
        CSV_OUT,
        resume=RESUME_AT_FIRST_UNREVIEWED,
    )

    if annotator.image is None:
        raise RuntimeError("No se pudo cargar ninguna imagen.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, annotator.on_mouse)

    try:
        while annotator.image is not None:
            cv2.imshow(WINDOW_NAME, annotator.render())
            key = cv2.waitKey(20) & 0xFF

            if key == ord("n"):
                annotator.first_point = None
                annotator.next_image()

            elif key == ord("p"):
                annotator.first_point = None
                annotator.previous_image()

            elif key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()

    print(f"Registros guardados en el CSV: {len(annotator.rows)}")
    cantidad = generar_revisado(CSV_OUT, Revisado_OUT)
    print(f"Archivo revisado guardado: {Revisado_OUT} ({cantidad} imágenes)")


if __name__ == "__main__":
    main()
