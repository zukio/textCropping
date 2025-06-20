import os
from google.cloud import vision    # バージョン: 3.5.0


def extract_text(image_file):
    """Detects text in the file."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcp-signature.json"

    client = vision.ImageAnnotatorClient()

    with open(image_file, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    result = texts[0].description if texts else None

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message)
        )

    print(f'Text: {result}')
