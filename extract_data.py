import cv2
import pytesseract
import re
import json

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('flattened_receipt.jpg')
if image is None:
    print("Error: Run scan_image.py first.")
    exit()

print("Running OCR...")
text = pytesseract.image_to_string(image)

extracted_data = {
    "vendor": None,
    "date": None,
    "total": None
}

# Clean up the lines
lines = [line.strip() for line in text.split('\n') if line.strip()]

# --- 1. Find Vendor ---
# The vendor is almost always the very first line with actual letters.
for line in lines:
    clean_line = re.sub(r'[^a-zA-Z0-9\s]', '', line).strip()
    if len(clean_line) > 3: # Skip tiny garbage lines
        extracted_data["vendor"] = clean_line
        break

# --- 2. Find Date and Total ---
for i, line in enumerate(lines):
    clean_line = re.sub(r'[^a-zA-Z0-9\$\.\,\:\/\-]', ' ', line).strip()
    lower_line = clean_line.lower()

    # Date Logic (Catches 10/18/24 and June 29, 2022)
    date_match = re.search(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b', clean_line)
    word_date_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}', clean_line, re.IGNORECASE)
    if (date_match or word_date_match) and not extracted_data["date"]:
        extracted_data["date"] = (date_match.group(1) if date_match else word_date_match.group(0))

# --- NEW TOTAL LOGIC ---
# Find ALL prices in the entire text (e.g., $103.68, 45.00)
all_prices = re.findall(r'\$?(\d+\.\d{2})', text)
if all_prices:
    # Convert strings to floats and find the maximum value
    max_price = max([float(price) for price in all_prices])
    # Format it back to 2 decimal places
    extracted_data["total"] = f"{max_price:.2f}"

print("\n--- Extracted JSON ---")
print(json.dumps(extracted_data, indent=4))
print("----------------------")