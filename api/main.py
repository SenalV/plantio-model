from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import pickle
from keras.models import load_model

app = FastAPI()

# MODEL = tf.keras.models.load_model("../models/1")
# with open('../models/plantio.pkl', 'rb') as f:
#     MODEL = pickle.load(f)

MODEL = load_model('../plantio.h5')

MODEL = pickle.load(open('../models/plantio.pkl', 'rb'))

CLASS_NAMES = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "Running..."

# def readImage(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     # image = readImage(await file.read())
#     # image_batch = np.expand_dims(image, 0)
#     # predction = MODEL.predict(image_batch)

#     # predicted_class = CLASS_NAMES[np.argmax(predction[0])]
#     predicted_class = CLASS_NAMES[0]

#     return {
#         'class': predicted_class
#     }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)