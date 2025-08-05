import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

# Custom CSS for UI polish
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #e0eafc, #cfdef3 100%);
    }
    .stApp {
        background-color: #f6fafd;
    }
    .image-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 24px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .caption {
        color: #333;
        font-size: 1rem;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎨 AI Multi-Image Generator")
st.subheader("Generate multiple AI images at once with Stable Diffusion 🚀")
st.write("Try prompts like **'A futuristic city at sunset, vibrant, digital art'** or **'A photorealistic painting of an astronaut riding a horse on Mars'**!")

prompt = st.text_input("Prompt", value="A cat astronaut surfing on a pizza in space, digital art", 
                       help="Describe the image you want (e.g., 'A robot playing chess in a park').")

num_images = st.slider("How many images to generate?", min_value=1, max_value=4, value=3)

# Session state: load pipeline only once
if "pipe" not in st.session_state:
    with st.spinner("Loading model... (first time only, please wait)"):
        st.session_state.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")

if st.button("✨ Generate Images", use_container_width=True):
    with st.spinner("AI is painting your masterpieces..."):
        images = st.session_state.pipe([prompt] * num_images).images

        st.write("## Results")
        cols = st.columns(num_images)
        for i, (img, col) in enumerate(zip(images, cols)):
            with col:
                # Show image in card style
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.image(img, use_column_width=True, caption=f"Image {i+1}")

                # Download button for each image
                buf = BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name=f"ai_image_{i+1}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Enter a prompt and click **Generate Images** to create your AI artwork!")

# Footer
st.markdown(
    """
    <hr style="margin-top:2em;">
    <div style="text-align: center; color: #888;">
        Made with <b>Diffusers</b> & <b>Streamlit</b> | Demo app by <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5" target="_blank">Stable Diffusion v1.5</a>
    </div>
    """, unsafe_allow_html=True
)
