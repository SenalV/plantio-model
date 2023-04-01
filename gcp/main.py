import tensorflow as tf
from PIL import Image
import numpy as np
from google.cloud import storage
from keras.models import load_model

BUCKET_NAME = 'plantio_model'
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

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "model/plantio-model-v2.h5",
            "/tmp/plantio-model-v2.h5"
        )
        model = load_model("/tmp/plantio-model-v2.h5")
    
    image = request.files.get("file")
    
    if image is None:
        return {"error": "No file found in request."}
    
    try:
        image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    except:
        return {"error": "Unable to process image file."}

    image = image/255

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}
