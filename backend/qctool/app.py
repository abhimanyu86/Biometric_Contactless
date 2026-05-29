import streamlit as st
import requests
from PIL import Image
import io
import base64
import numpy as np

BACKEND_URL = "http://localhost:5001"

def b64_to_image(b64_string):

    image_bytes = base64.b64decode(b64_string)

    return Image.open(io.BytesIO(image_bytes))

st.set_page_config(
    page_title="UIDAI QC Tool",
    layout="wide"
)

st.title("UIDAI Fingerprint QC Tool")

tab1, tab2 = st.tabs([
    "Image Quality",
    "Contact Comparison"
])

# TAB 1
with tab1:

    st.header("Fingerprint Quality Check")

    uploaded = st.file_uploader(
        "Upload Fingerprint",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:

        st.image(uploaded)

        if st.button("Run Quality Check"):

            response = requests.post(
                f"{BACKEND_URL}/process",
                files={
                    "image": uploaded
                }
            )

            data = response.json()

            if data["success"]:

                st.success("Processing Complete")

                st.metric(
                    "Minutiae Count",
                    data["minutiae_count"]
                )
                images = data.get("images", {})

                if "preprocessed" in images:

                    st.subheader("Preprocessed")

                    st.image(
                        b64_to_image(images["preprocessed"])
                    )
                    quality = data.get("quality", {})

                    if "visualization" in images:

                        st.subheader("Minutiae Visualization")

                        st.image(
                            b64_to_image(images["visualization"])
                        )

                    if quality:

                        st.subheader("Quality Checks")

                        st.write("Guidance:", quality["guidance"])

                        st.write("Blur Score:",
                                quality["blur"]["blur_score"])

                        st.write("Brightness:",
                                quality["brightness"]["brightness"])

                        st.write("Glare:",
                                quality["glare"]["has_glare"])

                else:
                    st.error(data["error"])

# TAB 2
with tab2:

    st.header("Contact Fingerprint Processing")

    uploaded2 = st.file_uploader(
        "Upload Contact Fingerprint",
        type=["png", "jpg", "jpeg"],
        key="contact"
    )

    if uploaded2:

        st.image(uploaded2)

        if st.button("Run Contact Processing"):

            response = requests.post(
                f"{BACKEND_URL}/process_contact",
                files={
                    "image": uploaded2
                }
            )

            data = response.json()

            if data["success"]:

                st.success("Contact Fingerprint Processing Complete")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Original Image")

                    st.image(uploaded2)

                with col2:

                    if "processed_image" in data:

                        st.subheader("Processed Image")

                        processed_img = b64_to_image(
                            data["processed_image"]
                        )

                        st.image(processed_img)

                        st.write(
                            "Processed Size:",
                            processed_img.size
                        )

                st.divider()

                st.subheader("Processing Details")

                st.write(
                    "Status:",
                    "SUCCESS"
                )

                if "saved_path" in data:

                    st.write(
                        "Saved Path:",
                        data["saved_path"]
                    )

            else:

                st.error(data["error"])