import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time

# st.set_page_config(page_title="Age Estimation", layout="wide", initial_sidebar_state="auto")
 
# Load the model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/efficientnet_b0.h5")   

# Load the haarcascade for face detection
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop face using OpenCV haarcascade
def detect_and_crop_face(image):
    face_cascade = load_face_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Use the first detected face
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face
    else:
        return None  # Return None if no face is detected

# Prediction function
# Prediction function
def predict(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    face = detect_and_crop_face(image)
    
    if face is None:
        return None, None, None, None, None
    
    # Resize and preprocess the cropped face
    face = cv2.resize(face, (224, 224))
    face = tf.keras.applications.efficientnet.preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # Make predictions
    model = load_model()
    age_pred, gender_pred, race_pred = model.predict(face)
    age_pred = int(age_pred[0][0])

    # Gender prediction
    raw_gender_pred = gender_pred[0][0]
    gender_pred = "Male" if raw_gender_pred < 0.5 else "Female"
    gender_confidence = 1 - raw_gender_pred if gender_pred == "Male" else raw_gender_pred

    # Race prediction
    race_labels = ["White", "Black", "Asian", "Indian", "Others"]
    race_pred_label = race_labels[np.argmax(race_pred[0])]
    race_confidence = np.max(race_pred[0])

    return age_pred, gender_pred, gender_confidence, race_pred_label, race_confidence
# Fungsi untuk membuat card metric dengan label di atas dan nilai confidence di bawah
def create_metric_card(label, value, info=None, color='yellow', width="150px"):
    info_html = f"<div style='font-size: 12px; color: #555;'>{info}</div>" if info else ""
    card_html = f"""
    <div class="metric-card" style="background-color: #d4d4d2; border-left: 5px solid {color}; padding: 10px; 
    display: inline-block; margin-right: 15px; width: {width}; text-align: center;">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {info_html}
    </div>
    """
    return card_html

# Apply custom styles
st.markdown("""
    <style>
    .metric-card {
        font-family: Arial, sans-serif;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        margin: 10px 5px 10px 0;
        width: 150px; /* Menetapkan lebar tetap */
    }
    .metric-label {
        font-size: 13px;
        color: #333;
    }
    .metric-value {
        font-size: 14px;
        font-weight: bold;
        color: #000;
    }
    </style>
""", unsafe_allow_html=True)


# Title and Description
st.title("Age, Gender, and Race Estimation")
st.markdown("Upload a facial image, and the model will estimate the age, gender, and race.")

# Using st.tabs for a Horizontal Tab Menu
tabs = st.tabs(["Prediction", "About"])

# Prediction Page
with tabs[0]:
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Generate a unique filename using the current timestamp
        timestamp = int(time.time())  # Current time as timestamp
        image_path = f"../asset/upload/{timestamp}.jpg"  # Specify a directory if needed
 
        # Save uploaded image to a unique file path
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show the original image
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        cola, colb, colc = st.columns([3,1,3])    
        with colb:    
            # Predict button
            predict_button = st.button("Predict")
        if predict_button:
            results = predict(image_path)
            
            # Check if results are None or if any result within the tuple is None
            if results is None or any(r is None for r in results):
                st.warning("Prediction could not be performed because no face was detected. Please upload a clear image showing the face.")
            else:
                age_pred, gender_pred, gender_confidence, race_pred_label, race_confidence = results
                
                # Display results if face was detected and predictions are available
                st.markdown(
                    f"""
                    <div style="padding: 10px; text-align: center; margin-top: 20px;">
                        <h3>Prediction Results</h3>
                        <div style="display: flex; justify-content: center;">
                            {create_metric_card("Age", age_pred, info="Years", color="#ffd700")}
                            {create_metric_card("Gender", gender_pred, info=f"Confidence {gender_confidence*100:.0f}%", color="#87cefa")}
                            {create_metric_card("Race", race_pred_label, info=f"Confidence {race_confidence*100:.0f}%", color="#ff7f50")}
                    """,
                    unsafe_allow_html=True
                )


with tabs[1]:
    st.title("About This Project")
    st.subheader("Project Overview")
    st.write("""
    This application predicts **age**, **gender**, and **race** from facial images using a deep learning model.
    The main goal of this project is to provide a streamlined prediction process that can support 
    demographic research, targeted marketing, and other fields requiring facial demographic analysis.
    """)

    st.subheader("Dataset Information")
    st.write("""
    The dataset used for this project is **UTKFace**, a comprehensive facial dataset containing a diverse 
    range of images labeled with three demographic categories: **age**, **gender**, and **race**.
    Each image in the dataset is tagged with the following labels:
    """)
    
    st.write("- **Age**: Age ranges from 0 to 116")
    st.write("- **Gender**: Male and Female.")
    st.write("- **Race**: White, Black, Asian, Indian, Others")

    st.subheader("Model Architecture")
    st.write("""
    The model used in this application is based on **EfficientNetB0**, a pre-trained model known for its 
    balance between accuracy and computational efficiency. The model has been fine-tuned to predict age, 
    gender, and race by unfreezing specific layers to adapt to this project’s data distribution and requirements.
    """)

    st.subheader("Model Performance")
    st.write("""
    The model was evaluated using several metrics:
    - **Gender Prediction Accuracy**: 89.76%
    - **Race Prediction Accuracy**: 83.55%
    - **Age Prediction Mean Absolute Error (MAE)**: 6.93 years
    """)
    
    st.subheader("Technologies and Techniques")
    st.write("""
    This application was built using **TensorFlow** for model development and **Streamlit** for deployment.
    Key techniques and methodologies applied include:
    - **Downsampling**: Applied to balance classes within the race and age categories to achieve fairer model 
      performance across diverse demographic groups.
    - **Fine Tuning**: Leveraged to adapt the pre-trained EfficientNetB0 model to the specific dataset, enhancing 
      the model's ability to learn from the UTKFace data.
    """)

    st.subheader("Limitations")
    st.write("""
    While the model demonstrates strong accuracy, there are some limitations:
    - **Data Bias**: Performance may vary across demographic groups due to dataset representation imbalances.
    - **Impact of Head Coverings**: The model might struggle to accurately predict age or gender for women wearing head coverings, such as hijabs, as these were not included in the training dataset.
    - **Age Prediction Range**: The model's age prediction accuracy can decrease when predicting ages outside 
      the central age range of the dataset.
    """)

    st.subheader("Future Improvements")
    st.write("""
    To further enhance model performance, we are considering:
    - **Expanding the Dataset**: Including more diverse samples across age, race, and gender.
    - **Advanced Model Tuning**: Implementing newer models or custom architectures tailored specifically to each label.
    - **Increased Data Augmentation**: Adopting more complex augmentations to cover a wider range of real-world 
      image variances.
    """)
    
    st.markdown("---")
    st.write("For more information or inquiries, please contact us at [dhikanugraha8@gmail.com](mailto:dhikanugraha8@gmail.com).")

