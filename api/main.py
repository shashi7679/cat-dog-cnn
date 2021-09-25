from fastapi import FastAPI,File,UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

Model = tf.keras.models.load_model('../models')
class_names = ['cat','dog']
app = FastAPI()

@app.get('/hello')

def read_file_as_image(data)->np.array:
    img = np.array(Image.open(BytesIO(data)))
    return img
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,0)
    pred = Model.predict(image_batch)
    pred_class = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0])
    return{
        'class':pred_class,
        'confidence':float(confidence)
    }

async def hello():
    return "Link working"

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8080)