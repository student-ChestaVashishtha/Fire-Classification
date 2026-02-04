import tensorflow as tf
import tf_keras as keras
import gradio as gr
import numpy as np
from PIL import Image

# 1. Recreate the exact architecture
# This matches the MobileNetV2 + 256 Dense + 256 Dense + Sigmoid setup
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None # Not needed since we load your custom weights
)
base_model.trainable = False 

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

# 2. Load the weights
# Using load_weights() avoids the 'DTypePolicy' and 'batch_shape' errors
model.load_weights("best_fire_model.h5")
def predict_fire(image):
    image = image.convert("RGB")   # IMPORTANT
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    print(pred)  # DEBUG LINE

    if pred.shape[-1] == 1:   # sigmoid model
        confidence = pred[0][0]
        label = "🔥 Fire Detected" if confidence < 0.23 else "✅ No Fire"
    # else:  # softmax model
    #     confidence = pred[0][1]
    #     label = "🔥 Fire Detected" if np.argmax(pred[0]) == 1 else "✅ No Fire"

    return f"{label} | Confidence: {1-confidence:.2f}"



# Gradio Interface
interface = gr.Interface(
    fn=predict_fire,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="🔥 Fire Detection App",
    description="Upload an image and the model will detect fire or non-fire."
)

interface.launch()
