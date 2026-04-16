import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffe3ef 0%, #ffd6ea 45%, #fff0f7 100%);
}

h1, h2, h3 {
    color: #d63384;
}

section[data-testid="stSidebar"] {
    background: #fff0f7;
    border-right: 2px solid #f8b6d2;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #b02a6b;
}

.stSlider > div {
    color: #b02a6b;
}

.stButton > button {
    background: linear-gradient(135deg, #ff4fa3, #ff7bbf);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(255, 79, 163, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ff3d98, #ff69b4);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.canvas-card {
    background: rgba(255,255,255,0.72);
    border: 2px solid #f6b2d0;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 12px 30px rgba(214, 51, 132, 0.12);
    backdrop-filter: blur(10px);
}

.info-card {
    background: rgba(255,255,255,0.78);
    border: 2px solid #f6b2d0;
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(214, 51, 132, 0.10);
    margin-bottom: 16px;
}

.result-card {
    background: white;
    border: 2px solid #f3b3d1;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(214, 51, 132, 0.10);
    text-align: center;
    margin-top: 18px;
}

.small-note {
    color: #9c4674;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.title('Reconocimiento de Dígitos escritos a mano')

st.markdown("""
<div class="info-card">
    <h3 style="margin-top:0;">Dibuja tu dígito</h3>
    <p class="small-note">Dibuja el número en el panel y luego presiona <b>Predecir</b>.</p>
</div>
""", unsafe_allow_html=True)

# Add canvas component
# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.markdown('<div class="canvas-card">', unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 105, 180, 0.18)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button('Predecir', use_container_width=True):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
            input_image.save('prediction/img.png')
            img = Image.open("prediction/img.png")
            res = predictDigit(img)

            st.markdown(
                f"""
                <div class="result-card">
                    <h2 style="margin:0; color:#d63384;">El Dígito es: {res}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-card">
                    <h3 style="margin:0; color:#d63384;">Por favor dibuja en el canvas el dígito.</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

# Add sidebar
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalúa")
st.sidebar.text("la capacidad de un RNA de reconocer") 
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")
#st.sidebar.text("GitHub Repository")
#st.sidebar.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
