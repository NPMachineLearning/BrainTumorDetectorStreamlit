import streamlit as st
from PIL import Image
import io
from Localization import Localization
from model_wrapper import ModelWrapper

st.set_page_config(page_title="Brain Tumor Detector",
                    page_icon="title_icon.jpg")

if "current_lang" not in st.session_state:
  st.session_state["current_lang"] = "en"

def on_lang_change():
  st.session_state.current_lang = st.session_state.selected_lang

@st.cache_resource(show_spinner="Preparing model...")
def load_wrapper():
  wrapper = ModelWrapper("id_to_class.txt")
  wrapper.download_model()
  wrapper.load_model()
  return wrapper

def load_localization():
  local = Localization("localizations", st.session_state.current_lang)
  return local

wrapper = load_wrapper()
localizer = load_localization()

lang_code_map = {local["name"]:local["code"] for local in localizer.available_locals()}
code_lang_map = {v:k for k, v in lang_code_map.items()}

titl_image = Image.open('title_icon.jpg')
titl_image = titl_image.resize((60,60))
st.image(titl_image)

st.title(localizer.localize("title"))
st.markdown(localizer.localize("desc"))

st.selectbox(localizer.localize("select_language"), 
              list(lang_code_map.values()), 
              index=localizer.current_language_index(),
              format_func=lambda x: code_lang_map[x],
              on_change=on_lang_change,
              key="selected_lang")

uploaded_image = st.file_uploader(localizer.localize("select_mri_image"), type=["jpg"])
if uploaded_image is not None:
  img_byte = io.BytesIO(uploaded_image.getvalue())
  image = Image.open(img_byte)
  image = image.convert("RGB")
  st.image(image, caption='MRI')

  if st.button(localizer.localize('detect'), type="primary"):
    classname, prob, alert = wrapper.predict_from_PIL(image)
    cls_local = localizer.localize(classname)
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



