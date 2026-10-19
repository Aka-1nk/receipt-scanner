# 🧾 Document Intelligence Pipeline (Receipt Scanner)

An end-to-end Computer Vision and OCR pipeline that takes a raw photo of a receipt, flattens it using OpenCV, extracts text with Tesseract, and outputs structured JSON. 

## 📸 Screenshots

<img width="1887" height="850" alt="image" src="https://github.com/user-attachments/assets/693db5da-e586-44ff-9b6d-613b4f5135df" />
<img width="915" height="390" alt="image" src="https://github.com/user-attachments/assets/330393f8-a2a2-4742-8fc3-39bb941d74dd" />


## 🚀 Features
- **Perspective Transform:** Automatically detects receipt boundaries using OpenCV contours and warps the image to a flat, top-down view.
- **Image Preprocessing:** Applies Adaptive Gaussian Thresholding to make the text crisp and handle uneven lighting/shadows.
- **Generalized Data Extraction:** Uses Regex and mathematical heuristics (not hardcoded rules) to extract the Vendor, Date, and Total from any receipt format.
- **Interactive Web UI:** Built with Streamlit for drag-and-drop testing.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Computer Vision:** OpenCV, NumPy
- **OCR Engine:** Tesseract OCR (via `pytesseract`)
- **Web Framework:** Streamlit

## 🧠 Engineering Blockers & Decisions
*During the development of this pipeline, several real-world OCR challenges were encountered and solved:*

1. **Blocker:** Tesseract OCR failed on raw photos, returning empty strings because the text was angled and shadowed.
   * **Solution:** Implemented a 4-point perspective transform using `cv2.getPerspectiveTransform` to flatten the receipt before OCR.
2. **Blocker:** Tesseract read the price column out of order and accidentally grabbed a phone number (`440.64`) as the Total.
   * **Solution:** Implemented a heuristic that finds all decimal numbers, ignores anything over `$500` (to filter out phone numbers), and assumes the largest remaining number is the final Total.
3. **Blocker:** Tesseract's default Page Segmentation Mode (PSM) tried to read the receipt in columns, skipping the right side of the page.
   * **Solution:** Used a custom config `--psm 6` to force Tesseract to read the image as a single uniform block of text.
4. **Blocker:** The thin font at the top of the receipt was getting erased during thresholding.
   * **Solution:** Added a "safety crop" to remove noisy background borders and tuned the adaptive thresholding `blockSize` to preserve fine pixel details.

## 🏃‍♂️ How to Run Locally
1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for Windows.
2. Clone this repository.
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate
