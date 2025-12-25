import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_my_model():
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "mask_detector.keras") 
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# Only 3 classes
class_names = ["With Mask", "Without Mask", "Mask Worn Incorrectly"]

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check mask status")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Open and convert image
        img = Image.open(uploaded_file)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize
        img_resized = img.resize((224, 224), Image.LANCZOS)

        # Preprocess
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array, verbose=0)
            result = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
            result = result[:3]

            pred_class = int(np.argmax(result))
            confidence = float(np.max(result))

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_resized, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader("Prediction Result")
            label = class_names[pred_class]

            if label == "With Mask":
                st.success(f"✅ {label}")
            elif label == "Without Mask":
                st.error(f"❌ {label}")
            else:
                st.warning(f"⚠️ {label}")

            st.metric("Confidence", f"{confidence * 100:.1f}%")

            with st.expander("View all probabilities"):
                for i, name in enumerate(class_names):
                    st.write(f"{name}: {result[i] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please try a different image (JPG or PNG format).")
