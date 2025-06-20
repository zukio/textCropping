import os
from typing import Tuple
import cv2
from google.cloud import vision    # バージョン: 3.5.0
from google.auth.exceptions import DefaultCredentialsError


def detect_text(content):
    """Detects text in the file."""
    from google.cloud import vision
    try:
        client = vision.ImageAnnotatorClient()
    except DefaultCredentialsError as e:
        raise RuntimeError(
            "Google credentials not configured. Set GOOGLE_APPLICATION_CREDENTIALS or specify 'gcp_credentials' in config.json"
        ) from e

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message)
        )

    return response


def detect_text_localpath(path):
    with open(path, "rb") as image_file:
        content = image_file.read()

    return detect_text(content)


def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message)
        )

    return response


def extract_text(img) -> Tuple[str, str, int]:
    """Detect text and bounding boxes using Google Cloud Vision API.

    Parameters
    ----------
    img : numpy.ndarray
        Image in BGR format.

    Returns
    -------
    Tuple[str, str, int]
        Detected fuconfigll text, bounding boxes in Tesseract style and
        character count.
    """
    # os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "./gcp-signature.json")

    success, buf = cv2.imencode('.png', img)
    if not success:
        raise RuntimeError("Failed to encode image for Vision API")

    response = detect_text(buf.tobytes())

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
