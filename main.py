from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Load the saved model
model = load_model('chest_xray_model.h5')

class_labels = ['NORMAL', 'PNEUMONIA']

def load_and_preprocess_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB").resize((256, 256))  # Ensure the image is in RGB format
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = load_and_preprocess_image(io.BytesIO(contents))
    prediction = model.predict(img_array)
    predicted_class = class_labels[int(np.round(prediction))]
    confidence = float(prediction[0][0])
    return JSONResponse(content={"predicted_class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
