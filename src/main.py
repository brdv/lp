import cv2
import pytesseract
import os

# Load license plate image

folder_path = "/Users/brdv/repos/personal/license_processor/src/angle"

file_lp = {}


def img_to_lp_to_text(imgpath: str):
    # Convert to grayscale
    img = cv2.imread(imgpath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess image to improve contrast and remove noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours in image
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find contour with largest area
    max_contour = max(contours, key=cv2.contourArea)

    # Draw bounding rectangle around contour
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract license plate region from image
    plate = gray[y : y + h, x : x + w]

    # Perform OCR on license plate region
    config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    text = pytesseract.image_to_string(plate, lang="eng", config=config)

    # Display output
    print("License Plate:", text.strip())
    cv2.imshow("License Plate", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return text


for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    lp = img_to_lp_to_text(file_path)
    file_lp[filename] = lp

for key in file_lp:
    print(f"{key} returns {file_lp[key]}")
