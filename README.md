# TC5031.10_ProyectoIntegrador
Visión artificial para confirmación de estándar de vestimenta

## Notas
El repositorio incluye imagenes de trabajadores.  Es de vital importancia mantener las caras ocultas para la captura de las mismas y con ello evitar PII

## Captura de video sin guardar rostros

El script `DataCapture/capture_wo_pii.py` abre una cámara o stream, detecta rostros con clasificadores Haar de OpenCV (frontal y perfil) y cubre las detecciones con rectángulos negros antes de mostrar el video. No guarda video; al presionar `s` guarda únicamente el fotograma ya anonimizado. Presiona `q` o Esc para salir.

Instala las dependencias y ejecuta desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python DataCapture/capture_wo_pii.py
```

Se fija `opencv-python==4.11.0.86` porque la instalación OpenCV 5.0.0.93 observada en este entorno no incluye los XML Haar que necesita el detector. Si ya instalaste ambas distribuciones (`opencv-python` y `opencv-contrib-python`), desinstala las dos del entorno virtual activo y luego vuelve a instalar `requirements.txt`.

Para elegir otra cámara, un archivo de video o un stream compatible con OpenCV, usa `--source`. Por ejemplo, `--source 1` selecciona la segunda cámara. Puedes ampliar el área negra con `--padding 0.5`; las imágenes se guardan en `captures_wo_pii` por defecto y se puede cambiar el directorio con `--output-dir`.

**Importante:** Haar es ligero, pero puede no detectar caras de perfil, parcialmente ocultas, pequeñas o con iluminación difícil. Un rectángulo solo cubre las detecciones; esto no garantiza anonimización. Valida el sistema con las condiciones reales de captura y revisa los fotogramas antes de usarlos. No uses imágenes o video sin máscara como respaldo.

## Para usar el repositorio
1. Clonar el repositorio
git init
git clone https://github.com/MarceloIsmael21/TC5031.10_ProyectoIntegrador

2. Crear y activar ambiente virtual (usar terminal powershell)
python -m venv .venv
.venv/Scripts/Activate.ps1

3. Instalar python (si ya cuentas con él en tu entorno, no es necesario)
Instala python desde la web
Instala el plugin de python en vs code
Habilita el ipykernel
pip install ipykernel -U -force-reinstall

4. Instalar requerimientos
pip install -r requirements.txt
