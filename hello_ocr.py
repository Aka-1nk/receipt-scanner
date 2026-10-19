import cv2
import pytesseract

# 1. Point to the Tesseract engine we just installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Load the image
print("Loading image...")
image = cv2.imread('receipt.jpg')

if image is None:
    print("ERROR: Could not find 'receipt.jpg'. Make sure the image is in the same folder as this script.")
else:
    # 3. Convert to grayscale (makes text easier for the AI to read)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. Extract text
    print("Extracting text...")
    text = pytesseract.image_to_string(gray)
    
    # 5. Print the result
    print("\n--- Extracted Text ---")
    print(text)
    print("----------------------")