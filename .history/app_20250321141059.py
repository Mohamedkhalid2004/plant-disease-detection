import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configure Gemini AI API Key
API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your API key
genai.configure(api_key=API_KEY)

def analyze_infection(image):
    """Analyze the image to detect infected areas based on color (red/yellow/brown spots)"""
    image = np.array(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define color ranges for infected regions
    red_lower, red_upper = np.array([0, 120, 70]), np.array([10, 255, 255])
    yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
    brown_lower, brown_upper = np.array([10, 50, 50]), np.array([20, 255, 200])

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

    detected = "Healthy"
    processed_image = image.copy()

    if cv2.countNonZero(red_mask) > 500:
        detected = "Red spots detected - Possible fungal infection."
        mask = red_mask
        color = (255, 0, 0)
    elif cv2.countNonZero(yellow_mask) > 500:
        detected = "Yellow spots detected - Possible bacterial infection."
        mask = yellow_mask
        color = (0, 255, 255)
    elif cv2.countNonZero(brown_mask) > 500:
        detected = "Brown spots detected - Possible nutrient deficiency."
        mask = brown_mask
        color = (139, 69, 19)
    else:
        cv2.rectangle(processed_image, (10, 10), (image.shape[1] - 10, image.shape[0] - 10), (0, 255, 0), 5)
        return detected, processed_image

    # Draw bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(processed_image, (x, y), (x + w, y + h), color, 2)

    return detected, processed_image

def get_gemini_analysis(description):
    """Get AI analysis from Gemini AI, including disease causes and treatment recommendations."""
    model = genai.GenerativeModel("gemini-1.5-pro")  
    prompt = f"""
    Analyze the following plant disease description and provide:
    1. Possible causes
    2. Treatment methods (organic & chemical solutions)
    3. Preventive measures

    Disease detected: {description}
    """
    response = model.generate_content(prompt)
    return response.text if response else "No response received."

def generate_pdf_report(original_image, processed_image, description, ai_analysis):
    """Generate a detailed PDF report with images and AI insights"""
    pdf_bytes = io.BytesIO()
    pdf = canvas.Canvas(pdf_bytes, pagesize=letter)

    # Title
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, 750, "🌱 Plant Disease Detection Report 🌿")
    
    # Detected Issue
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 730, f"Detected Issue: {description}")

    # AI Analysis
    pdf.drawString(100, 700, "AI Analysis:")
    text = ai_analysis[:500] + "..." if len(ai_analysis) > 500 else ai_analysis
    pdf.drawString(100, 680, text)

    # Treatment Suggestions
    treatments = {
        "fungal infection": "Apply fungicides & remove infected leaves.",
        "bacterial infection": "Use copper-based sprays & improve airflow.",
        "nutrient deficiency": "Adjust soil nutrients & apply fertilizers."
    }
    for key, value in treatments.items():
        if key in description.lower():
            pdf.drawString(100, 640, f"Suggested Treatment: {value}")

    # Add Images to PDF
    pdf.drawString(100, 600, "Uploaded Image:")
    pdf.drawImage(ImageReader(original_image), 100, 400, width=200, height=150)
    
    pdf.drawString(320, 600, "Processed Image:")
    pdf.drawImage(ImageReader(processed_image), 320, 400, width=200, height=150)

    pdf.showPage()
    pdf.save()
    pdf_bytes.seek(0)
    return pdf_bytes

# Streamlit UI
st.title("🌿 Plant Disease Detection with Gemini AI")

# Upload Image or Capture from Camera
upload_option = st.radio("Choose Image Source:", ("Upload", "Capture from Camera"))

image = None

if upload_option == "Upload":
    uploaded_file = st.file_uploader("Upload a Plant Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif upload_option == "Capture from Camera":
    captured_image = st.camera_input("Take a Photo")
    if captured_image:
        image = Image.open(captured_image)

if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        description, processed_image = analyze_infection(image)
        st.write(f"**📝 Detected:** {description}")

        # Convert processed image for display
        processed_image = Image.fromarray(processed_image)
        st.image(processed_image, caption="Processed Image with Markings", use_column_width=True)

        if description != "Healthy":
            ai_response = get_gemini_analysis(description)
            st.write("**🤖 Gemini AI Analysis:**")
            st.write(ai_response)

            # Generate PDF report
            pdf_file = generate_pdf_report(image, processed_image, description, ai_response)

            # Download Button
            st.download_button(label="📥 Download Report",
                               data=pdf_file,
                               file_name="Plant_Disease_Report.pdf",
                               mime="application/pdf")
