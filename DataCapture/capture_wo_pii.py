"""Live video capture with detected faces covered by black rectangles.

Only the anonymized preview is displayed, and screenshots saved with ``s`` are
written after masking. No raw frames are saved by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_source(value: str) -> int | str:
    """Interpret numeric sources as camera indexes; keep other sources as URLs/paths."""
    return int(value) if value.isdecimal() else value


def detect_faces(frame, frontal_cascade, profile_cascade):
    """Detect frontal and profile faces, checking mirrored profiles as well."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    options = {
        "scaleFactor": 1.1,
        "minNeighbors": 5,
        "minSize": (30, 30),
        "flags": cv2.CASCADE_SCALE_IMAGE,
    }

    detections = []
    detections.extend(frontal_cascade.detectMultiScale(gray, **options))
    detections.extend(profile_cascade.detectMultiScale(gray, **options))

    # The profile cascade is directional; mirror the image to catch the other side.
    mirrored_gray = cv2.flip(gray, 1)
    mirrored_faces = profile_cascade.detectMultiScale(mirrored_gray, **options)
    frame_width = frame.shape[1]
    detections.extend(
        (frame_width - x - width, y, width, height)
        for x, y, width, height in mirrored_faces
    )
    return detections


def cover_faces(frame, faces, padding: float):
    """Return a copy of the frame with padded face detections painted black."""
    anonymized = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    for x, y, width, height in faces:
        pad_x = round(width * padding)
        pad_y = round(height * padding)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(frame_width, x + width + pad_x)
        bottom = min(frame_height, y + height + pad_y)
        cv2.rectangle(anonymized, (left, top), (right, bottom), (0, 0, 0), thickness=-1)

    return anonymized


def load_cascades():
    """Load Haar XML assets and explain how to repair an incomplete OpenCV install."""
    cascade_dir = Path(cv2.data.haarcascades)
    cascade_paths = {
        "frontal": cascade_dir / "haarcascade_frontalface_default.xml",
        "profile": cascade_dir / "haarcascade_profileface.xml",
    }
    missing = [str(path) for path in cascade_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "OpenCV is installed without the required Haar XML files. Missing: "
            f"{', '.join(missing)}. This project requires opencv-python==4.11.0.86; "
            "uninstall both opencv-python and opencv-contrib-python, then reinstall "
            "requirements.txt in the active environment."
        )

    frontal = cv2.CascadeClassifier(str(cascade_paths["frontal"]))
    profile = cv2.CascadeClassifier(str(cascade_paths["profile"]))
    if frontal.empty() or profile.empty():
        raise RuntimeError(f"OpenCV could not parse the Haar XML files in {cascade_dir}.")
    return frontal, profile


def capture_video(
    source: int | str = 0,
    padding: float = 0.30,
    output_dir: Path = Path("captures_wo_pii"),
) -> None:
    """Show masked camera/video frames; save only masked stills when 's' is pressed."""
    if padding < 0:
        raise ValueError("padding must be non-negative")

    frontal_cascade, profile_cascade = load_cascades()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    print("Showing anonymized video. Press 's' to save a masked image; 'q' or Esc to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("No frame received; stopping the stream.")
                break

            faces = detect_faces(frame, frontal_cascade, profile_cascade)
            anonymized = cover_faces(frame, faces, padding)
            cv2.imshow("Video anonymized (Haar cascade)", anonymized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg")
                output_path = output_dir / filename
                if cv2.imwrite(str(output_path), anonymized):
                    print(f"Saved anonymized image: {output_path}")
                else:
                    print(f"Could not save image: {output_path}")
            elif key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream video with detected faces covered by black rectangles."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video-file path, or stream URL (default: 0).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.30,
        help="Extra mask size as a fraction of the detected face box (default: 0.30).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures_wo_pii"),
        help="Directory for anonymized screenshots saved with 's'.",
    )
    args = parser.parse_args()

    if args.padding < 0:
        parser.error("--padding must be non-negative")

    capture_video(
        source=parse_source(args.source),
        padding=args.padding,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
