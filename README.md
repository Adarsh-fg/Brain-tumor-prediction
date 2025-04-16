# 🧠 Brain Tumor Detection and Guidance App

This is a full-stack web application that detects brain tumors from MRI images using a pre-trained CNN model and provides tumor segmentation via K-Means clustering. It also offers AI-generated medical guidance, lifestyle suggestions, and hospital recommendations using Google's Gemini model.

---

## 🔍 Features

- 🧠 **Brain Tumor Detection:** Classifies MRI scans into `glioma`, `meningioma`, `pituitary`, or `no tumor` using a CNN model.
- 🎯 **Tumor Segmentation:** Highlights potential tumor regions with K-Means clustering and overlays.
- 🧬 **AI Medical Guidance:** Suggests next steps and lifestyle changes based on diagnosis using Gemini (Google Generative AI).
- 🏥 **Hospital Suggestions:** Lists hospitals worldwide or by country for further treatment.
- 💬 **Chatbot Support:** AI-powered chatbot responds to brain tumor–related questions.
- 📷 **Visual Output:** Displays segmented images, masks, and overlays with predictions and confidence.

---

## 🛠 Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML + Bootstrap (via `index.html`)
- **AI Models:**
  - TensorFlow Keras CNN model (`cnn_final_model.h5`)
  - Google Gemini (Generative AI API)
- **Image Processing:** OpenCV, Pillow, NumPy
- **Environment Management:** `python-dotenv`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/brain-tumor-detector.git
cd brain-tumor-detector
```

### 2. Install Dependencies

Create a virtual environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Required Libraries:**
- Flask
- TensorFlow
- OpenCV-Python
- Pillow
- NumPy
- Matplotlib
- python-dotenv
- markdown
- google-generativeai (Gemini)

> Create a `requirements.txt` using `pip freeze > requirements.txt` after installing.

### 3. Add Environment Variables

Create a `.env` file in the root directory and add your Gemini API key:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 4. Add Models & Assets

Place the trained CNN model in the root folder:

```
cnn_final_model.h5
```

Ensure there's an `uploads/` directory to store incoming and processed images.

---

## 🧪 Running the App

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000/
```

---

## 📂 File Structure

```
.
├── app.py                 # Main Flask application
├── Model_training.ipynb   # Notebook used to train CNN model
├── cnn_final_model.h5     # Pre-trained CNN model
├── templates/
│   └── index.html         # Frontend HTML (upload form & results)
├── uploads/               # Uploaded and processed images
├── .env                   # Contains your Gemini API key
└── requirements.txt       # Python dependencies
```

---

## 🤖 How it Works

1. User uploads an MRI image.
2. The app:
   - Validates and processes the image.
   - Segments the tumor region using K-Means.
   - Classifies tumor type with CNN.
   - Generates next steps, lifestyle advice, and hospital suggestions via Gemini.
3. Results and visual overlays are displayed on the web page.

---

## 🤖 Gemini Chatbot

You can chat with the AI assistant about:

- Tumor types
- Symptoms
- Treatment options
- Specialist hospitals

It will only answer **brain tumor–related** questions.

---

## 📌 Notes

- Ensure your image is in `.jpg`, `.jpeg`, or `.png` format.
- Gemini integration requires an active API key from [Google AI Studio](https://makersuite.google.com/).
- The app is **not** a replacement for medical diagnosis. Always consult professionals.

---

## 📄 License

MIT License. See `LICENSE` file .

---

## 🙏 Acknowledgments

- Trained model dataset sourced from Kaggle or public brain MRI datasets.
- Gemini by Google for AI-generated content and medical assistance.

