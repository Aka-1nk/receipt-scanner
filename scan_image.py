import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('receipt.jpg')
if image is None:
    print("Error: Image not found.")
    exit()

orig = image.copy()
ratio = image.shape[0] / 800.0 # Increased size for better detection
small_image = cv2.resize(image, (int(image.shape[1] / ratio), 800))

# 1. Stronger Blur to ignore carpet texture
gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0) # Increased blur
edged = cv2.Canny(gray, 30, 150) # Lower thresholds to catch softer edges

# 2. Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

# 3. The Fallback Logic
if screenCnt is None:
    print("Could not find a clear rectangle. Using the whole image as a fallback...")
    warped = orig # Just use the original image
else:
    print("Found a rectangle! Warping perspective...")
    pts = screenCnt.reshape(4, 2) * ratio
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# 4. Smart Orientation using Tesseract's built-in OSD (Orientation and Script Detection)
print("Detecting text orientation...")
try:
    osd = pytesseract.image_to_osd(warped)
    # Extract the rotation angle from the OSD output
    angle = int([line for line in osd.split('\n') if "Rotate" in line][0].split(":")[1].strip())
    print(f"Detected rotation: {angle} degrees")
    
    if angle == 90:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        warped = cv2.rotate(warped, cv2.ROTATE_180)
    elif angle == 270:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
except Exception as e:
    print("OSD Failed. Assuming text is upright.")

# 5. Save the flattened/straightened image
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_bw = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite("flattened_receipt.jpg", warped_bw)
print("Saved 'flattened_receipt.jpg'.")

# 6. OCR
print("\nExtracting text...")
text = pytesseract.image_to_string(warped_bw)
print("\n--- Extracted Text ---")
print(text)
print("----------------------")