import streamlit as st
from PIL import Image
import io
from model_wrapper import ModelWrapper

@st.cache_resource
def load_wrapper():
  wrapper = ModelWrapper("brain_tumor_detector.h5", "id_to_class.txt")
  return wrapper

wrapper = load_wrapper()

lang_code_map = {
  "English": "en",
  "Chinese": "ch",
}

st.title("Brain Tumor Detector")
lang_opt = st.selectbox("Select your language", list(lang_code_map.keys()))
uploaded_image = st.file_uploader("Choose a MRI Image", type=["jpg"])
if uploaded_image is not None:
  img_byte = io.BytesIO(uploaded_image.getvalue())
  image = Image.open(img_byte)
  image = image.convert("RGB")
  st.image(image, caption='MRI')

  if st.button('Detect', type="primary"):
    classname, prob, alert = wrapper.predict_from_PIL(image)
    lang_code = lang_code_map[lang_opt]
    cls_local = wrapper.classname_to_local(classname, lang=lang_code)
    prob_percent = round(prob*100.0, 2)
    markdown = ""
    if alert:
      markdown = f"""
      ## :warning: 
      ### :red[{cls_local}]
      ### :red[{prob_percent} %]"""
    else:
      markdown = f"""
      ## :ok_hand: 
      ### :green[{cls_local}] 
      ### :green[{prob_percent}%]"""
    st.markdown(markdown)



