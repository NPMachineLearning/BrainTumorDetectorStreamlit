
import tensorflow as tf
import cv2
import imutils
import streamlit as st
import numpy as np
import os
import urllib.request

class STProgressBar():
  def __init__(self, text=""):
    self.text = text
    self.progressbar = None
  def __call__(self, block_num, block_size, total_size):
    progress = block_num * block_size / total_size
    progress = np.clip(progress, 0.0, 1.0)
    if self.progressbar is None:
      self.progressbar = st.progress(value=0.0, text=self.text)
    if self.progressbar:
      self.progressbar.progress(value=progress, text=self.text)

FILE_PATH = "./detector_model.h5"

class ModelWrapper:
  def __init__(self, model_url, id_to_cls_path, download_progress:STProgressBar=None, download_complete=None):
    self.image_size = (256,256)
    self.id_to_cls_map = self.load_id_to_class(id_to_cls_path)
    self.en_local = {"glioma": "Glioma tumor",
                    "meningioma": "Meningioma tumor",
                    "notumor": "No tumor",
                    "pituitary": "Pituitary tumor"}   
    self.ch_local = {"glioma": "膠質細胞瘤",
                    "meningioma": "腦膜瘤",
                    "notumor": "正常",
                    "pituitary": "垂體瘤"}
    
    if not os.path.exists(FILE_PATH):
      if download_progress is not None:
        urllib.request.urlretrieve(model_url, FILE_PATH, 
                                  reporthook=download_progress)
      else:
        urllib.request.urlretrieve(model_url, FILE_PATH)
      if download_complete is not None:
        download_complete()

    self.model = tf.keras.models.load_model(FILE_PATH,
                                           compile=False)
  
  def predict(self, img_tensor):
    output = self.model.predict(img_tensor, verbose=0)
    label = tf.argmax(output, axis=1)
    label = tf.squeeze(label, axis=0)
    cls_id = label.numpy()
    prob = tf.squeeze(output, axis=0)[cls_id]
    prob = prob.numpy()
    cls = self.id_to_cls_map[cls_id]
    alert = False if cls_id==2 else True
    return cls, prob, alert

  def predict_from_path(self, img_path):
    input_data = self.get_input_data_path(img_path)
    return self.predict(input_data)
  
  def predict_from_PIL(self, img):
    input_data = self.get_input_data_array(img)
    return self.predict(input_data)

  def classname_to_local(self, classname, lang="en"):
    if lang=="en":
      return self.en_local[classname]
    elif lang=="ch":
      return self.ch_local[classname]
    else:
      return classname

  def load_id_to_class(self, file_path):
    id_to_cls = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            print()
            target, classname = line.replace("\n", "").split("\t")
            id_to_cls[int(target)] = classname
    return id_to_cls

  def get_input_data_path(self, img_path):
    input_img = tf.io.read_file(img_path)
    input_img = tf.io.decode_jpeg(input_img, 3)
    input_img = self.crop_img(input_img.numpy())
    input_img = tf.cast(input_img, tf.float32)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img

  def get_input_data_array(self, img):
    input_img = tf.keras.utils.img_to_array(img, dtype="uint8")
    input_img = self.crop_img(input_img)
    input_img = tf.cast(input_img, tf.float32)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img

  def crop_img(self, img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    # resize image
    new_img = cv2.resize(new_img, self.image_size)
    
    return new_img
