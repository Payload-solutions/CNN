
import numpy as np
import requests
from PIL import Image
from io import BytesIO


def conv(image, img_filter):

    heigh = image.shape[0]
    width = image.shape[1]

    imc = np.zeros((height - len(img_filter) + 1,
    width - len(img_filter)+1))
    print(f"""
    The value of height: {heigh}
    The value of width: {width}
    the value of len(img_filter): {len(img_filter)}
    """)
    print(imc)


def main():

    url = """https://api.ferrarinetwork.ferrari.com/v2/network-content/medias//resize/6094000a8c09a35ca689fba0-ferrari-magazine-S3OmQ-vnzt.jpg?apikey=9QscUiwr5n0NhOuQb463QEKghPrVlpaF"""

    resp = requests.get(url=url)
    
    image_rgb = np.asarray(Image.open(BytesIO(resp.content)).convert("RGB"))
    print(image_rgb)
    image_gray = np.mean(image_rgb, axis=2, dtype=np.uint)
    print(image_gray)

if __name__ == "__main__":
    main()
