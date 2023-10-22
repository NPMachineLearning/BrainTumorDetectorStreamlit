# Brain Tumor Detector Introduction

This is an streamlit app of brain tumor detector that
can automatically detect brain tumor of image from MRI
machine.

## Local development

1. Open terminal and cd to this directory
2. Create python environment `python -m venv .\`
3. Activate environment `venv\scripts\activate`
4. Install dependencies `pip install tensorflow streamlit opencv-python imutils pipreqs`
5. Start streamlit `streamlit run tumor_detector_app.py`

## Deploy to streamlit

1. Open terminal and cd to this directory
2. Create python environment `python -m venv .\` if not
3. Activate environment `venv\scripts\activate`
4. Install dependencies `pip install tensorflow streamlit opencv-python imutils pipreqs`
5. Make sure streamlit app is working `streamlit run tumor_detector_app.py`
6. Ouput **requirements.txt** `pipreqs .\`
7. Modify **requirements.txt**, remove Pillow, streamlit, tensorflow-intel
