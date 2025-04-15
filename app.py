# --- Brain Tumor Detection and Guidance App ---
# importing necessary libraries

from flask import Flask, request, render_template, send_from_directory, url_for, jsonify
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv
import markdown
from werkzeug.utils import safe_join # For secure file serving
from flask import abort, send_from_directory
import traceback

# Load environment variables from the .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Define the folder path where uploaded files will be stored
UPLOAD_FOLDER = "uploads"  # Path to the folder where files will be uploaded

# Configure the Flask application to use the defined upload folder
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Check if the upload folder exists, if not, create it
if not os.path.exists(UPLOAD_FOLDER): 
    os.makedirs(UPLOAD_FOLDER)  # Create the folder if it does not exist

# --- Gemini Configuration ---
# This script initializes the Gemini model using an API key from the environment variables.
# It first checks if the API_KEY environment variable is set. If the key is available,
# it configures the Gemini model using the provided API key and attempts to load the 
# latest version of the Gemini model ('gemini-1.5-flash-latest' or 'gemini-1.0-pro').
# If the model is initialized successfully, it prints a success message. If any exception
# occurs during initialization, the error message is printed. If the API_KEY is not found,
# an error message is printed indicating the missing environment variable.

API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_model = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-1.0-pro'
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing Gemini: {e}")
else:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")

# --- CNN Model Loading ---
# This code attempts to load a pre-trained CNN model (saved as 'cnn_final_model.h5') using the `load_model` function.
# If the model is loaded successfully, a success message is printed to indicate that the model has been loaded.
# In case of any errors during the loading process, the exception is caught and an error message is printed.
# If an error occurs, the `model` variable is set to `None` to indicate the failure to load the model.
#
# Additionally, a list of labels (representing different types of brain tumors) is defined:
# - 'glioma': Represents glioma tumors.
# - 'meningioma': Represents meningioma tumors.
# - 'notumor': Represents cases with no tumor detected.
# - 'pituitary': Represents pituitary tumors.

try:
    model = load_model("cnn_final_model.h5")
    print("CNN model loaded successfully.")
except Exception as e:
    print(f"ERROR loading CNN model: {e}")
    model = None

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


def create_overlay(original_cv_img, mask_cv_img, color=[0, 255, 0], alpha=0.4):
    """Creates a semi-transparent overlay of the mask on the original image."""
    if original_cv_img is None or mask_cv_img is None:
        print("Warning: Cannot create overlay with invalid input images.")
        return None
    try:
        # Ensure mask is 3 channels like original (if original is color)
        # If original is grayscale, convert mask to grayscale too
        if len(original_cv_img.shape) == 3 and original_cv_img.shape[2] == 3:
            # Original is color, make mask BGR
            mask_colored = cv2.cvtColor(mask_cv_img, cv2.COLOR_GRAY2BGR)
        else:
             # Original is likely grayscale, keep mask grayscale
             mask_colored = mask_cv_img # Or cv2.cvtColor(mask_cv_img, cv2.COLOR_GRAY2BGR) if needed later

        # Create a colored version of the mask where the mask is white
        # colored_overlay = np.zeros_like(original_cv_img, dtype=np.uint8)
        # Use chosen color for the mask area
        # colored_overlay[mask_cv_img == 255] = color # Apply color only to tumor pixels

        # Blend the original and the colored mask
        # Need to ensure both are the same type and channel count for cv2.addWeighted
        if len(original_cv_img.shape) == 2: # If original was grayscale
             original_for_blend = cv2.cvtColor(original_cv_img, cv2.COLOR_GRAY2BGR)
        else:
             original_for_blend = original_cv_img

        # Ensure mask_colored is also BGR for blending
        if len(mask_colored.shape) == 2:
             mask_colored_for_blend = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)
        else:
             mask_colored_for_blend = mask_colored

        # Create the colored region based on the mask
        # Make a copy to draw on
        overlay_display = original_for_blend.copy()
        # Apply color only where mask is white (tumor region)
        overlay_display[mask_cv_img == 255] = color

        # Blend the overlay with the original using alpha transparency
        # beta = 1.0 - alpha
        # blended_image = cv2.addWeighted(overlay_display, alpha, original_for_blend, beta, 0.0)

        # Alternative: Blend only the colored part onto the original
        # Find where the mask is white
        mask_indices = mask_cv_img == 255
        # Blend only these pixels
        overlay_display[mask_indices] = cv2.addWeighted(
            original_for_blend[mask_indices],
            1 - alpha, # Weight for original pixels
            overlay_display[mask_indices],
            alpha, # Weight for colored pixels
            0.0
        )[0] # addWeighted returns tuple on single pixel access in some versions

        return overlay_display

    except Exception as e:
        print(f"Error creating overlay image: {e}")
        return None


# theese functions are used to segment the tumor and classify it using the CNN model
# This function segments the tumor from the image using K-means clustering.
# It applies CLAHE for contrast enhancement, reshapes the image for clustering, and uses K-means to find clusters.
# The function then identifies the tumor cluster based on intensity and creates a binary mask for the tumor region.
# It returns the original image, segmented image, and the mask.
# If any error occurs during processing, it returns placeholder images.
# The function also handles cases where the image is empty or invalid.
# The function uses OpenCV for image processing and K-means clustering.
# It also uses NumPy for numerical operations and PIL for image handling.


def segment_tumor(image_path, K=3):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError("Image not loaded")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image)
        pixel_values = image_clahe.reshape((-1, 1)).astype(np.float32)
        if pixel_values.size == 0: raise ValueError("Empty image data")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        actual_K = K
        attempts = 10
        compactness, kmeans_labels, centers = cv2.kmeans(pixel_values, actual_K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        kmeans_labels = np.clip(kmeans_labels, 0, centers.shape[0] - 1)
        segmented_image = centers[kmeans_labels.flatten()].reshape(image_clahe.shape)
        _, otsu_mask = cv2.threshold(image_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(otsu_mask == 255) == 0 or np.sum(otsu_mask == 0) == 0:
             tumor_cluster_intensity = np.mean(image_clahe)
        else:
             tumor_cluster_intensity = np.mean(image_clahe[otsu_mask == 255])
        if centers.size > 0:
            tumor_cluster_index = np.argmin(np.abs(centers.flatten() - tumor_cluster_intensity))
        else:
             raise ValueError("K-Means returned no centers")
        mask = (kmeans_labels.flatten() == tumor_cluster_index).astype(np.uint8).reshape(image_clahe.shape) * 255
        return image, segmented_image, mask
    except Exception as e:
        print(f"Error during segmentation: {e}")
        placeholder = np.zeros((100, 100), dtype=np.uint8)
        return placeholder, placeholder, placeholder


# This function classifies the tumor using a pre-trained CNN model.
# It loads the image, resizes it, normalizes pixel values, and reshapes it for the model input.
# The function then predicts the tumor type and returns the predicted label and confidence score.
# If any error occurs during classification, it returns an error message and a confidence score of 0.0.
# The function uses TensorFlow and NumPy for model prediction and numerical operations.
# It also uses PIL for image handling.

def classify_tumor(image_path):
    if model is None: return None, None, "Error: Model not loaded", 0.0
    try:
        img = Image.open(image_path).convert('RGB')
        resized_img = img.resize((299, 299), Image.Resampling.BICUBIC)
        img_array = np.asarray(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        probs = predictions[0]
        predicted_label_index = np.argmax(probs)
        if 0 <= predicted_label_index < len(labels):
            predicted_label = labels[predicted_label_index]
            confidence = np.max(probs)
        else:
            predicted_label = "Error: Prediction index invalid"
            confidence = 0.0
        return None, probs, predicted_label, confidence
    except Exception as e:
        print(f"Error classifying image {image_path}: {e}")
        return None, None, f"Error: Classification failed ({type(e).__name__})", 0.0

# --- Gemini Powered Functions ---
# This function generates content using the Gemini model based on the provided prompt.
# It sets safety settings to block certain categories of content and handles exceptions during the generation process.
# If the model is not available, it returns an error message.
# The function uses the Gemini API to generate content and checks for safety ratings.
# It also handles different error scenarios and returns appropriate error messages.
# The function returns the generated content or an error message if generation fails.
# The function uses the Google Generative AI library for content generation.

def generate_gemini_content(prompt):
    if not gemini_model: return "Error: Gemini model not available."
    try:
        safety_settings=[ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        if response.parts: return response.text
        else:
            reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            if reason != "Unknown" and reason != "SAFETY": return f"Content generation stopped: {reason}."
            if response.prompt_feedback and response.prompt_feedback.block_reason == "SAFETY":
                 blocked = [r.category for r in response.prompt_feedback.safety_ratings if r.blocked]
                 return f"Content blocked due to safety concerns: {', '.join(str(b) for b in blocked)}."
            return "AI could not generate a response (No content returned)."
    except Exception as e:
        print(f"Gemini API Error: {e}")
        err_str = str(e).lower()
        if "api key not valid" in err_str: return "Error: Invalid Gemini API Key."
        if "404" in err_str and "models/" in err_str: return f"Error: Gemini model not found/supported."
        if "permission" in err_str or "denied" in err_str: return "Error: Permission denied for Gemini API."
        if "quota" in err_str: return "Error: Gemini API quota exceeded."
        return f"Error generating AI content: ({type(e).__name__})."

# This function generates guidance and lifestyle tips based on the tumor type.
# It uses the Gemini model to create two sections: suggested next steps and lifestyle tips.
# The function handles different tumor types and formats the response using Markdown.
# It also includes a disclaimer and ensures the response is informative and reassuring.
# The function uses the Gemini API to generate content and handles exceptions during generation.
# It returns the generated guidance and lifestyle tips or an error message if generation fails.
# The function uses the Google Generative AI library for content generation.

def get_gemini_guidance(tumor_type):
    if tumor_type.startswith("Error:") or not tumor_type:
        return {"next_steps": "Cannot generate guidance due to prior error.", "lifestyle": ""}

    if tumor_type == "notumor":
        prompt = """
        The brain scan analysis indicates 'no tumor'. Generate two distinct sections using Markdown formatting:
        1.  **Suggested Next Steps:** Provide a concise list (using '*' or '-' for bullet points) of general health recommendations and follow-up advice. Mention consulting a doctor if symptoms persist.
        2.  **General Brain Health Lifestyle Tips:** Offer a brief list (using '*' or '-' for bullet points) of actionable lifestyle advice for maintaining overall brain health.
        Keep the tone informative and reassuring. Ensure the section titles are bold using double asterisks.
        """
    else:
        prompt = f"""
        A preliminary brain scan analysis suggests a '{tumor_type}' tumor type. Generate two distinct sections using Markdown formatting:
        1.  **Potential Next Steps (Informational Only):** Provide a concise list (using '*' or '-' for bullet points) of potential next steps someone might discuss with their healthcare provider. Use bold text (double asterisks) for emphasis where appropriate.
        2.  **Supportive Lifestyle Considerations:** Offer a brief list (using '*' or '-' for bullet points) of general lifestyle suggestions. Use bold text (double asterisks) for emphasis on key advice points.
        **IMPORTANT:** Start the entire response with a clear disclaimer in bold Markdown: **Disclaimer: This is AI-generated information, NOT medical advice. ALWAYS consult qualified medical professionals for diagnosis and treatment.**
        Ensure the section titles (**Potential Next Steps...** and **Supportive Lifestyle Considerations...**) are also Heading and make it bold to highlight using double asterisks.
        """

    response_text = generate_gemini_content(prompt)
    # --- Basic Parsing Logic (keep as before or refine) ---
    next_steps = response_text; lifestyle = ""
    steps_heading_tumor = "Potential Next Steps (Informational Only):"; lifestyle_heading_tumor = "Supportive Lifestyle Considerations:"
    steps_heading_notumor = "Suggested Next Steps:"; lifestyle_heading_notumor = "General Brain Health Lifestyle Tips:"
    response_lower = response_text.lower()
    steps_heading = steps_heading_notumor if tumor_type == "notumor" else steps_heading_tumor
    lifestyle_heading = lifestyle_heading_notumor if tumor_type == "notumor" else lifestyle_heading_tumor
    try:
        steps_start_idx = response_lower.index(f"**{steps_heading.lower()}**") + len(f"**{steps_heading.lower()}**")
        lifestyle_start_idx = response_lower.index(f"**{lifestyle_heading.lower()}**")
        next_steps_content = response_text[steps_start_idx:lifestyle_start_idx].strip()
        lifestyle_content = response_text[lifestyle_start_idx + len(f"**{lifestyle_heading.lower()}**"):].strip()
        next_steps = f"**{steps_heading}**\n{next_steps_content}" # Re-add heading
        lifestyle = f"**{lifestyle_heading}**\n{lifestyle_content}" # Re-add heading
        # Check/Prepend disclaimer
        disclaimer_heading = "**disclaimer:"
        if response_lower.startswith(disclaimer_heading):
            disclaimer_end_idx = response_lower.find(f"**{steps_heading.lower()}**")
            if disclaimer_end_idx != -1:
                disclaimer_text = response_text[:disclaimer_end_idx].strip()
                next_steps = f"{disclaimer_text}\n\n{next_steps}"
    except ValueError:
        print("Warning: Could not parse AI guidance using headings. Full response in 'next_steps'.")
        if "Error:" in response_text: lifestyle = "" # Ensure lifestyle is empty on error
    return {"next_steps": next_steps, "lifestyle": lifestyle}


# This function suggests hospitals or clinics based on the tumor type.
# It uses the Gemini model to generate a list of recognized hospitals or clinics known for expertise in the specified tumor type.
# The function allows filtering by country and formats the response using Markdown.
# It includes a disclaimer and ensures the response is informative and relevant.
# The function uses the Gemini API to generate content and handles exceptions during generation.
# It returns the generated hospital suggestions or an error message if generation fails.
# The function uses the Google Generative AI library for content generation.

def get_gemini_hospitals(tumor_type, country=None):
    """
    Suggests hospitals using Gemini, optionally filtering by country.
    Returns raw markdown string.
    """
    if tumor_type == "notumor" or tumor_type.startswith("Error:") or not tumor_type:
        return "N/A" # No need for hospitals or if prior error

    # Build the location part of the prompt
    location_specifier = ""
    if country and country.strip() and country != "Other":
        location_specifier = f"in {country.strip()}" # Specify the country
    else:
        # If no country or 'Other', ask for global/major centers
        location_specifier = "that are globally recognized or located in major regions like the USA, Europe, India, etc."

    prompt = f"""
    List some recognized hospitals, clinics, or neuroscience centers {location_specifier} known for expertise in diagnosing and treating '{tumor_type}' brain tumors.
    Format the output as a Markdown list (using '*' or '-'). Include the general location (e.g., City, Country) for each.
    Start the list with a brief disclaimer in bold Markdown: **Disclaimer: This list is AI-generated, not exhaustive, and not an endorsement. Patients MUST research options based on their specific situation, location, and consult their medical team.**
    """
    print(f"Hospital Prompt (location: {country if country else 'Global'}):\n{prompt[:200]}...") # Log part of the prompt

    hospitals_text = generate_gemini_content(prompt)
    return hospitals_text

# --- End Gemini Powered Functions ---

# This function checks if the uploaded image is valid and not corrupted.
# It uses PIL to open and verify the image, and OpenCV to read the image.
# If the image is valid, it returns True. If any error occurs during validation, it returns False.
# The function handles different image formats (JPG, JPEG, PNG) and checks for file integrity.
# The function uses PIL for image handling and OpenCV for image reading.
# It also uses NumPy for numerical operations and handles exceptions during image processing.
# The function returns True if the image is valid, otherwise it returns False.
def is_valid_image(filepath):
    
    try:
        img = Image.open(filepath); img.verify()
        img = Image.open(filepath); img.load()
        cv2_img = cv2.imread(filepath)
        # Allow if PIL worked even if OpenCV fails, but log warning
        if cv2_img is None: print(f"Warning: OpenCV couldn't read {filepath}, but PIL could.")
        return True
    except Exception as e:
        print(f"Image validation failed for {filepath}: {e}")
        return False

# --- Flask Routes ---
# The main route for the Flask application. It handles both GET and POST requests.
# On GET requests, it renders the index.html template with default context values.
# On POST requests, it processes the uploaded image file, performs tumor segmentation and classification,
# and displays the results in the index.html template.
# It also generates AI content for next steps and lifestyle tips using the Gemini model.
# The function handles file validation, error handling, and image processing using OpenCV and TensorFlow.
# The function also saves the processed images (segmented, mask, overlay) to the upload folder.
# The function uses Flask's request and render_template functions to handle form submissions and rendering HTML templates.
# It also uses OpenCV for image processing and TensorFlow for model prediction.
# The function uses PIL for image handling and Google Generative AI for content generation.
# The function also includes error handling for file operations and image processing.
# The function uses Markdown for formatting the generated content and handles different error scenarios.
# It returns the rendered HTML template with the processed results or error messages as needed.
# The function uses Flask's abort function to handle HTTP errors and send_from_directory for serving uploaded files securely.
# The function also includes debugging information for file handling and image processing.
# The function uses the os module for file path handling and environment variable loading.

@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "prediction": None, "confidence": 0, "image_path": None,
        "segmented_path": None, "mask_path": None, "probabilities": [],
        "labels": labels, "next_steps_html": "", "lifestyle_html": "",
        "hospitals_html": "", "error_message": None,
        "selected_country": None, # Add selected_country to context
        "overlay_path": None, # Add new context variable
        "selected_country": None
    }

    if request.method == "POST":
        selected_country = request.form.get("country", None) # Get selected country
        context["selected_country"] = selected_country # Store it for the template

        # --- File Handling ---
        if 'file' not in request.files:
            context["error_message"] = "No file part."
            return render_template("index.html", **context)
        file = request.files['file']
        if file.filename == '':
            context["error_message"] = "No file selected."
            return render_template("index.html", **context)
        if not file or not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            context["error_message"] = "Invalid file format (JPG, JPEG, PNG only)."
            return render_template("index.html", **context)

        filename = file.filename # Consider using secure_filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(filepath)
            print(f"File saved: {filepath}")
            if not is_valid_image(filepath):
                os.remove(filepath)
                context["error_message"] = "Invalid or corrupted image file."
                return render_template("index.html", **context)
            context["image_path"] = filename

            # --- Processing ---
            print("Segmenting...")

            original_cv = cv2.imread(filepath) # Read as color by default
            if original_cv is None: raise ValueError("Failed to read original image with OpenCV")

            _, segmented, mask = segment_tumor(filepath) # Assumes segment_tumor returns grayscale mask
            original, segmented, mask = segment_tumor(filepath)
            if segmented is None or mask is None or segmented.size == 0:
                 context["error_message"] = "Image segmentation failed."
                 if os.path.exists(filepath): os.remove(filepath)
                 return render_template("index.html", **context)

            print("Classifying...")
            _, probs, predicted_label, confidence = classify_tumor(filepath)
            print(f"Classified as: {predicted_label} ({confidence:.2f})")
            if predicted_label.startswith("Error:"):
                 context["error_message"] = f"Classification Failed: {predicted_label}"
                 if os.path.exists(filepath): os.remove(filepath)
                 return render_template("index.html", **context)

            context["prediction"] = predicted_label
            context["confidence"] = round(confidence * 100, 2) if confidence else 0
            context["probabilities"] = probs.tolist() if probs is not None else []

            # --- Save Processed Images ---
            segmented_filename = "segmented_" + filename
            mask_filename = "mask_" + filename
            overlay_filename = "overlay_" + filename
            segmented_path_full = os.path.join(app.config["UPLOAD_FOLDER"], segmented_filename)
            mask_path_full = os.path.join(app.config["UPLOAD_FOLDER"], mask_filename)
            overlay_path_full = os.path.join(app.config["UPLOAD_FOLDER"], overlay_filename) # Full path for overlay

            seg_ok = cv2.imwrite(segmented_path_full, segmented)
            mask_ok = cv2.imwrite(mask_path_full, mask)

            # Create and save the overlay image
            overlay_image = create_overlay(original_cv, mask) # Pass original CV image and mask
            overlay_ok = False
            if overlay_image is not None:
                overlay_ok = cv2.imwrite(overlay_path_full, overlay_image)
                print(f"DEBUG: Overlay saved to '{overlay_path_full}': {overlay_ok} | Exists: {os.path.exists(overlay_path_full)}")

            # --- START DEBUGGING ---
            print("-" * 20)
            print(f"DEBUG: Attempting to save processed images for '{filename}'")
            print(f"  - Original exists at '{filepath}': {os.path.exists(filepath)}")
            print(f"  - Segmented saved to '{segmented_path_full}': {seg_ok} | Exists: {os.path.exists(segmented_path_full)}")
            print(f"  - Mask saved to '{mask_path_full}': {mask_ok} | Exists: {os.path.exists(mask_path_full)}")
            # --- END DEBUGGING ---

            if seg_ok: context["segmented_path"] = segmented_filename
            if mask_ok: context["mask_path"] = mask_filename # Keep mask path if needed elsewhere
            if overlay_ok: context["overlay_path"] = overlay_filename # Add overlay path

            if not seg_ok or not mask_ok or not overlay_ok:
                print(f"Warning: Failed to save one or more processed images (seg={seg_ok}, mask={mask_ok}, overlay={overlay_ok})")

            # --- Generate AI Content ---
            if gemini_model:
                print("Generating AI Guidance...")
                guidance_raw = get_gemini_guidance(predicted_label)
                print("Generating AI Hospitals...")
                # Pass the selected country to get_gemini_hospitals
                hospitals_raw = get_gemini_hospitals(predicted_label, selected_country)

                md_extensions = ['extra', 'nl2br', 'sane_lists']
                next_steps_md = guidance_raw.get("next_steps", "Error fetching guidance.")
                lifestyle_md = guidance_raw.get("lifestyle", "")
                hospitals_md = hospitals_raw if hospitals_raw else "N/A"

                context["next_steps_html"] = markdown.markdown(next_steps_md, extensions=md_extensions)
                context["lifestyle_html"] = markdown.markdown(lifestyle_md, extensions=md_extensions)
                context["hospitals_html"] = markdown.markdown(hospitals_md, extensions=md_extensions)
                print("AI content generated and converted.")
            else:
                 context["next_steps_html"] = "<p><em>AI guidance unavailable.</em></p>"
                 context["hospitals_html"] = "<p><em>AI suggestions unavailable.</em></p>"


            # --- Render Results ---
            return render_template("index.html", **context)

        except FileNotFoundError:
            context["error_message"] = f"Error: File not found after saving. Check permissions."
            if 'filepath' in locals() and os.path.exists(filepath): 
                try: 
                    os.remove(filepath); 
                except OSError: 
                    pass
            return render_template("index.html", **context)
        except cv2.error as e:
             context["error_message"] = f"OpenCV Error during processing: {e}."
             if 'filepath' in locals() and os.path.exists(filepath):
                try: os.remove(filepath); 
                except OSError: 
                    pass
             return render_template("index.html", **context)
        except Exception as e:
            import traceback
            print(f"Unhandled Error: {e}\n{traceback.format_exc()}")
            context["error_message"] = f"Unexpected error: {type(e).__name__}."
            if 'filepath' in locals() and os.path.exists(filepath): 
                try: os.remove(filepath); 
                except OSError: pass
            return render_template("index.html", **context)

    # For GET request
    return render_template("index.html", **context)


# This route serves the uploaded files securely using Flask's send_from_directory function.
# It uses the safe_join function to ensure that the requested file is within the upload directory.
# The route handles the file serving and includes security checks to prevent path traversal attacks.
# It also includes error handling for file not found and other exceptions.
# The function uses Flask's abort function to handle HTTP errors and send_from_directory for serving files securely.
# The function uses the os module for file path handling and environment variable loading.
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    print(f"\nDEBUG: Request received for /uploads/{filename}")
    upload_dir_abs = os.path.abspath(app.config["UPLOAD_FOLDER"])
    print(f"DEBUG: Absolute upload directory: {upload_dir_abs}")
    try:
        # Use absolute path for safety check and sending
        safe_filepath = safe_join(upload_dir_abs, filename)
        print(f"DEBUG: safe_join result: {safe_filepath}")

        if safe_filepath is None:
             print(f"DEBUG: safe_join returned None for '{filename}' - likely unsafe.")
             abort(404)

        # Check if the resulting path is actually a file
        if not os.path.isfile(safe_filepath):
             print(f"DEBUG: File not found or not a file at '{safe_filepath}'")
             abort(404) # Return 404 Not Found

        # Normalize both paths before the startswith check
        normalized_safe_filepath = os.path.normpath(safe_filepath)
        normalized_upload_dir = os.path.normpath(upload_dir_abs)

        print(f"DEBUG: Normalized safe_filepath: {normalized_safe_filepath}")
        print(f"DEBUG: Normalized upload_dir:   {normalized_upload_dir}")

        # Security check: Prevent accessing files outside the upload directory
        # Add os.sep to the directory path to ensure we match the directory itself, not a partial name
        if not normalized_safe_filepath.startswith(normalized_upload_dir + os.sep):
             print(f"DEBUG: Path traversal attempt blocked: '{normalized_safe_filepath}' is outside '{normalized_upload_dir}{os.sep}'")
             abort(403) # Return 403 Forbidden

        print(f"DEBUG: Serving file: '{filename}' from directory: '{upload_dir_abs}'")
        # Use the original upload_dir_abs and filename for send_from_directory
        return send_from_directory(upload_dir_abs, filename)

    except Exception as e:
         print(f"ERROR serving file {filename}: {e}")
         # Use traceback to print the full stack trace for unexpected errors
         traceback.print_exc()
         abort(500) # Keep aborting 500 for unexpected errors


# This route handles chat messages sent to the server.
# It processes the incoming message, checks if it is related to brain tumors, and generates a response using the Gemini model.

@app.route('/chat', methods=['POST'])
def chat():
    try:
        import re  # Ensure this is at the top of your file if not already imported

        user_input = request.json.get('message', '').strip()

        if not user_input:
            return jsonify({'response': 'Please type a question.'})

        # Define keywords related to brain tumors
        brain_keywords = [
            "brain tumor", "tumour", "glioblastoma", "meningioma","glioma","pituitary", "neuro-oncology", "mri",
            "brain scan", "brain surgery", "tumor", "chemotherapy", "radiation", "oncologist",
            "brain cancer", "symptoms", "treatment", "diagnosis", "neurosurgeon", "gamma knife"
        ]

        # Check if message is related to brain tumor using keyword match
        if not any(re.search(r'\b' + re.escape(kw) + r'\b', user_input.lower()) for kw in brain_keywords):
            return jsonify({
                'response': "I'm here to help only with brain tumor–related questions. Please ask about symptoms, treatment, hospitals, or diagnosis."
            })

        # Custom prompt to guide Gemini
        prompt = f"""
You are a helpful medical assistant chatbot focused ONLY on **brain tumor–related information**.
The user just asked:

\"{user_input}\"

Reply with clear and concise information related to brain tumors, including symptoms, treatments, hospitals, medicines, or follow-up care.
If the question is outside the brain tumor domain, respond politely saying you cannot assist.
"""

        response = gemini_model.generate_content(prompt)
        reply = response.text if response and hasattr(response, 'text') else "Sorry, I couldn't generate a response."

        return jsonify({'response': reply})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'response': 'Something went wrong. Please try again later.'})


# This route clears the server-side chat history.
# It handles POST requests and resets the chat history stored in a global variable.
# The route returns a success message if the history is cleared successfully.
# If any error occurs during the process, it returns an error message.

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    try:
        # If you're using any persistent chat history, clear it here
        # Example: global variable, in-memory history, database, etc.
        global chat_history
        chat_history = []

        return jsonify({'success': True, 'message': 'Server-side chat history cleared.'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Failed to clear server history.'})


# This route handles the health check for the server.
# It returns a simple JSON response indicating the server is running.

if __name__ == "__main__":
    if model is None: print("\nWARNING: CNN Model failed to load.")
    if gemini_model is None: print("WARNING: Gemini Model failed to initialize.\n")
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=5000, debug=is_debug)