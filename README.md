# Brain Tumor Detector Introduction

This is an streamlit app of brain tumor detector that
can automatically detect brain tumor of image from MRI
machine.

## Local development

1. Open terminal and cd to this directory
2. Create python environment `python -m venv .\`
3. Activate environment `venv\scripts\activate`
4. Install dependencies `pip install tensorflow streamlit opencv-python-headless imutils pipreqs`
5. Start streamlit `streamlit run tumor_detector_app.py`

## Deploy to streamlit

1. Open terminal and cd to this directory
2. Create python environment `python -m venv .\` if not
3. Activate environment `venv\scripts\activate`
4. Install dependencies `pip install tensorflow streamlit opencv-python imutils pipreqs`
5. Make sure streamlit app is working `streamlit run tumor_detector_app.py`
6. Ouput **requirements.txt** `pipreqs .\`
7. Modify **requirements.txt**, remove any packages except following:
   - imutils
   - opencv_python_headless
   - tensorflow

**Note**:

1. We must use [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) package instead of [opencv-python](https://pypi.org/project/opencv-python/) when deploy to streamlit otherwise an error will appear when app start `Importerror: libgl.so.1: cannot open shared object file: no such file or directory opencv error`. Same error was discussed [here](https://discuss.streamlit.io/t/streamlit-sharing-importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory-opencv-error/12367)
2. Python 3.9 is recommend
