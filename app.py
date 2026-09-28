
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="PSL Sign Language AI",
    page_icon="🤟"
)

# Load model and classes
@st.cache_resource
def load_model_and_classes():

    model = tf.keras.models.load_model(
        "psl_sign_language_cnn_final.keras"
    )

    with open("psl_classes.json", "r") as f:
        classes = json.load(f)

    return model, classes


model, classes = load_model_and_classes()


# Prediction function
def predict_psl_sign(image, threshold=70):

    image = image.resize((128, 128))

    image = np.array(image).astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    confidence = prediction[predicted_index] * 100

    if confidence >= threshold:
        predicted_class = classes[predicted_index]
    else:
        predicted_class = "Uncertain"

    return predicted_class, confidence


# UI
st.title("🤟 PSL Sign Language AI")

st.write(
    "Upload a hand-sign image and AI will predict "
    "the corresponding Pakistan Sign Language sign."
)

st.info("This prototype recognizes 36 static PSL signs.")


uploaded_file = st.file_uploader(
    "Upload a hand-sign image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Hand Sign",
        use_container_width=True
    )

    if st.button("🔍 Predict Sign"):

        predicted_class, confidence = predict_psl_sign(
            image,
            threshold=70
        )

        st.subheader("Prediction")

        if predicted_class == "Uncertain":

            st.warning(
                f"⚠️ Uncertain — Model confidence: "
                f"{confidence:.2f}%"
            )

            st.write(
                "Please try another image with a clear "
                "hand pose and good lighting."
            )

        else:

            st.success(
                f"🤟 Predicted Sign: {predicted_class}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )
