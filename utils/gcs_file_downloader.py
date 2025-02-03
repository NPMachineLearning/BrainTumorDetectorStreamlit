import streamlit as st
import pathlib
from google.cloud.storage import Client
from google.oauth2 import service_account

class GCSFileDownloader(object):
    def __init__(self, credentials:dict[str, str]):
        self.gcs_credentials = service_account.Credentials.from_service_account_info(credentials)
        self.gcs_client = Client(project=credentials.project_id, credentials=self.gcs_credentials)
    
    def download_file(self, bucket_name:str, path_to_file:str, filename:str, override:bool=False) -> str:
        file_path = pathlib.Path(filename)

        # if redownload not True, return file path if file already exists
        if not override:
            if file_path.is_file():
                return filename
        
        # access to bucket
        bucket = self.gcs_client.bucket(bucket_name)
        if not bucket.exists():
            raise Exception("GCS bucket name:{bucket_name} does not exits")
        
        # access to file
        blob = bucket.blob(path_to_file)
        if not blob.exists():
            raise Exception("GCS file:{path_to_file} does not exits")
        
        # create parent directory
        if not file_path.parent.is_dir():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # download file
        blob.download_to_filename(filename)

        return filename

# if __name__ == "__main__":
#     try:
#         gcs_secrets = st.secrets.connections.gcs
#         downloader = GCSFileDownloader(gcs_secrets)
#         model_path = downloader.download_file("np-machine-learning-models",
#                                               "tf2/image_classification/brain_tumor_detector.h5",
#                                               "model.h5")

#     except Exception as e:
#         print(e)

