import os
from typing import Tuple
import cv2
from google.cloud import vision    # バージョン: 3.5.0


def extract_text(img) -> Tuple[str, str, int]:
    """Detect text and bounding boxes using Google Cloud Vision API.

    Parameters
    ----------
    img : numpy.ndarray
        Image in BGR format.

    Returns
    -------
    Tuple[str, str, int]
        Detected full text, bounding boxes in Tesseract style and
        character count.
    """
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", "./gcp-signature.json")

    client = vision.ImageAnnotatorClient()

    success, buf = cv2.imencode('.png', img)
    if not success:
        raise RuntimeError("Failed to encode image for Vision API")

    image = vision.Image(content=buf.tobytes())

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors")

    annotation = response.full_text_annotation
    result_text = annotation.text if annotation.text else ""

    height, width = img.shape[:2]
    boxes_list = []
    char_count = 0

    for page in annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        ch = symbol.text
                        if not ch or ch.isspace():
                            continue
                        char_count += 1
                        vertices = symbol.bounding_box.vertices
                        if len(vertices) >= 4:
                            x1 = int(vertices[0].x)
                            y1 = int(vertices[0].y)
                            x2 = int(vertices[2].x)
                            y2 = int(vertices[2].y)
                            # Convert to Tesseract box format (origin bottom-left)
                            boxes_list.append(
                                f"{ch} {x1} {height - y2} {x2} {height - y1}")

    boxes = "\n".join(boxes_list)

    return result_text, boxes, char_count
