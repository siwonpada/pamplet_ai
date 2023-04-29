from typing import Union
from typing_extensions import Annotated
import cv2 
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import numpy as np 
from full import full_pipeline

app = FastAPI()

@app.post("/")
async def root():
    return {"message": "Hello World"}
    

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    contents = await file.read()

    data_io = io.BytesIO(contents)
    img = Image.open(data_io)

    numpy_image=np.array(img)  
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    bbox = full_pipeline(opencv_image)

    return {"bbox": bbox}
