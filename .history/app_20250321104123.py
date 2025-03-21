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
    
    detected = "Unknown"
    if cv2.countNonZero(red_mask) > 500:
        detected = "Red spots detected - Possible fungal infection."
    elif cv2.countNonZero(yellow_mask) > 500:
        detected = "Yellow spots detected - Possible bacterial infection."
    elif cv2.countNonZero(brown_mask) > 500:
        detected = "Brown spots detected - Possible nutrient deficiency."
    
    return detected

def get_gemini_analysis(description):
    """Get AI analysis from Gemini AI"""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Analyze the following plant disease description: {description}")
    return response.text

# Streamlit UI
st.title("Plant Disease Detection using Gemini AI")

uploaded_file = st.file_uploader("Upload a Plant Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        description = analyze_infection(image)
        st.write(f"Detected: {description}")
        
        if description != "Unknown":
            ai_response = get_gemini_analysis(description)
            st.write("**Gemini AI Analysis:**")
            st.write(ai_response)
