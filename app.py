import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
import json
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Receipt Scanner AI", layout="wide")
st.title("🧾 Document Intelligence Pipeline")
st.write("Upload a receipt photo. The pipeline will flatten the image, run OCR, and extract structured JSON.")

uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original Upload", use_container_width=True)

    with st.spinner("Processing pipeline..."):
        # --- 1. FLATTENING ---
        ratio = orig.shape[0] / 500.0
        small_image = cv2.resize(orig, (int(orig.shape[1] / ratio), 500))
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 5000:
                screenCnt = approx
                break

        if screenCnt is None:
            warped = orig 
        else:
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

        # --- 2. SAFETY CROP & PREPROCESSING ---
        h, w = warped.shape[:2]
        crop_margin_y = int(h * 0.03)
        crop_margin_x = int(w * 0.03)
        warped_cropped = warped[crop_margin_y:h-crop_margin_y, crop_margin_x:w-crop_margin_x]

        warped_gray = cv2.cvtColor(warped_cropped, cv2.COLOR_BGR2GRAY)
        warped_bw = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        
        # --- 3. OCR ---
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(warped_bw, config=custom_config)
        
        # --- 4. GENERALIZED EXTRACTION LOGIC ---
        extracted_data = {"vendor": None, "date": None, "total": None}
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # --- VENDOR (Generalized) ---
        # Rule: The vendor is usually the first line that has letters and NO numbers.
        vendor_candidates = []
        for line in lines[:5]: # Look at the top 5 lines
            if re.search(r'[a-zA-Z]', line) and not re.search(r'\d', line):
                # Clean weird symbols but keep letters
                clean_line = re.sub(r'[^a-zA-Z\s]', '', line).strip()
                if len(clean_line) > 2: 
                    vendor_candidates.append(clean_line)
        
        if vendor_candidates:
            # We pick the first one that looks like a valid word/name
            extracted_data["vendor"] = vendor_candidates[0]

        # --- DATE (Generalized) ---
        # Rule: Look for standard date formats (10/18/24, 10-18-2024, June 29, 2022)
        for line in lines:
            date_match = re.search(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b', line)
            word_date_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}', line, re.IGNORECASE)
            if (date_match or word_date_match):
                extracted_data["date"] = (date_match.group(1) if date_match else word_date_match.group(0))
                break

        # --- TOTAL (Generalized) ---
        # Rule: The final total is mathematically the largest number on the receipt.
        # 1. Find all numbers that look like prices (digits followed by . or , and 2 digits)
        text_clean = text.replace(',', '.')
        all_prices = re.findall(r'(\d+\.\d{2})', text_clean)
        
        # 2. Filter out phone numbers (anything over 500 is likely not a receipt total)
        valid_prices = [float(p) for p in all_prices if float(p) < 500]
        
        if valid_prices:
            max_price = max(valid_prices)
            extracted_data["total"] = f"{max_price:.2f}"

    with col2:
        st.image(warped_bw, caption="Processed (Flattened) Image", use_container_width=True)
        
    st.success("Pipeline Complete!")
    st.subheader("Extracted JSON Data")
    st.json(extracted_data)
    
    with st.expander("See Raw OCR Text"):
        st.text(text)