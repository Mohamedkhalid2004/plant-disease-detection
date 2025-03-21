import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io

# Configure Gemini AI API Key
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)

def analyze_infection(image):
    """Analyze the image to detect infected areas based on color (red/yellow/brown spots)"""
    image = np.array(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define color ranges for infected regions
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    brown_lower = np.array([10, 50, 50])
    brown_upper = np.array([20, 255, 200])
    
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    
    detected = "Healthy"
    processed_image = image.copy()
    
    if cv2.countNonZero(red_mask) > 500:
        detected = "Red spots detected - Possible fungal infection."
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    elif cv2.countNonZero(yellow_mask) > 500:
        detected = "Yellow spots detected - Possible bacterial infection."
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    elif cv2.countNonZero(brown_mask) > 500:
        detected = "Brown spots detected - Possible nutrient deficiency."
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (139, 69, 19), 2)
    else:
        cv2.rectangle(processed_image, (10, 10), (image.shape[1] - 10, image.shape[0] - 10), (0, 255, 0), 5)
    
    return detected, processed_image

def get_gemini_analysis(description):
    """Get AI analysis from Gemini AI"""
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use the correct model name
    response = model.generate_content(f"Analyze the following plant disease description: {description}")
    return response.text

# Streamlit UI
st.title("Plant Disease Detection using Gemini AI")

uploaded_file = st.file_uploader("Upload a Plant Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        description, processed_image = analyze_infection(image)
        st.write(f"Detected: {description}")
        
        # Convert processed image for display
        processed_image = Image.fromarray(processed_image)
        st.image(processed_image, caption="Processed Image with Markings", use_column_width=True)
        
        if description != "Healthy":
            ai_response = get_gemini_analysis(description)
            st.write("**Gemini AI Analysis:**")
            st.write(ai_response)