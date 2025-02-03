# Brain Tumor Detector Introduction

This is an streamlit app of brain tumor detector that
can automatically detect brain tumor of image from MRI
machine.

## Setup info for connecting GCS

The model is hosted on GCS(Google Cloud Storage) and this app use client library to connect to
GCS server.

### For local development

- With in project root directory create file `.streamlit/secrets.toml`. This is
  only for local development refer to [here](https://blog.streamlit.io/secrets-in-sharing-apps/).
- Add following to secrets.toml.

```
[connections.gcs]
type="XXX"
project_id="XXX"
private_key_id="XXX"
private_key="XXX"
client_email="XXX"
client_id="XXX"
auth_uri="XXX"
token_uri="XXX"
auth_provider_x509_cert_url="XXX"
client_x509_cert_url="XXX"
universe_domain="XXX"
```

Where XXX are value to be filled. This are data from google's service account which
is used to connect to GCS(Google Cloud Storage). The deep learning model is hosted in
GCS(Google Cloud Storage) These data are exposed to streamlit app.

- Add `.streamlit/secrets.toml` file to .gitignore as this contain sensitive data.

### For production

- Add content of `.streamlit/secrets.toml` to streamlit app's secret on cloud.
  Refer to [here](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## Local development

Make sure you have done this **Setup info for connecting GCS** step

- Open terminal and cd to this directory
- Create python environment `python -m venv .\`
- Activate environment `venv\scripts\activate`
- Install dependencies `pip install tensorflow-cpu==2.12.0 streamlit opencv-python-headless imutils google_cloud_storage pipreqs`
- Start streamlit `streamlit run tumor_detector_app.py`

## Deploy to streamlit

Make sure you have done this **Setup info for connecting GCS** step

- Follow **Local development** step
- Run command `pipreqs --force .`
- In **requirements.txt** remove any packages that are not in the following list
  and add `google_cloud_storage==3.0.0`
  - imutils
  - opencv_python_headless
  - tensorflow_cpu

**Note**:

1. We must use [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) package instead of [opencv-python](https://pypi.org/project/opencv-python/) when deploy to streamlit otherwise an error will appear when app start `Importerror: libgl.so.1: cannot open shared object file: no such file or directory opencv error`. Same error was discussed [here](https://discuss.streamlit.io/t/streamlit-sharing-importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory-opencv-error/12367)
2. Python 3.9 is recommend
