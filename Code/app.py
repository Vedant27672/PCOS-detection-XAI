from __future__ import division, print_function
import sys
import os
import numpy as np

# Keras
from keras.models import load_model
import tensorflow as tf

# Flask utils
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import imutils
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, and the layer to be used to visualize activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to find the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer with 4D output
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # create a new model with output both,feature map and final prediction
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output],
        )

        # record operations performed by Model on the variables
        with tf.GradientTape() as tape:
            # cast image to float-32 data type, pass the image through the gradient model, and grab the loss associated with the specific class index
            inputs = tf.cast(
                image, tf.float32
            )  # casting image into 32 bit float format for calculations in tensorflow
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # computes the partial derivative of the loss function with respect to each activation in convOutputs
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(
            convOutputs > 0, "float32"
        )  # array with only 1s(+ve) and 0s(-ve)
        castGrads = tf.cast(grads > 0, "float32")  ##array with only 1s(+ve) and 0s(-ve)
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

    # initialize our 'gradient class activation map and build the heatmap


# load the input image from disk (in Keras/TensorFlow format) and preprocess it
# image = load_img("C:/Users/divya/OneDrive/Desktop/infected2.jpg`", target_size=(224, 224))
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
# image = imagenet_utils.preprocess_input(image)


# Define a Flask app
app = Flask(__name__)

# Load your trained model
model = load_model("D:/project/Sem 5/Minor/PCOS1/bestmodel.keras")
print("Model loaded. Check http://127.0.0.1:5000/")


def predictimage(path):
    # Load and preprocess the image
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize to [0, 1]
    input_arr = np.array([img_array])  # Convert to batch format

    # Make prediction
    pred = model.predict(input_arr)[0][0]  # Get the prediction confidence score

    # Determine label based on prediction
    label = (
        "Not Affected" if pred >= 0.5 else "Affected"
    )  # Assuming 0.5 threshold for binary classification
    return label, float(pred)  # Convert pred to float for JSON compatibility


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template("index.html")


import os
from flask import jsonify, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array, load_img


@app.route("/predict", methods=["POST"])
def upload():
    if request.method == "POST":
        try:
            # Get the file from the POST request
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No file provided"}), 400

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            label, pred = predictimage(file_path)

            # Prepare result based on label
            result = "PCOS Positive" if label == "Affected" else "PCOS Negative"

            # Initialize GradCAM and compute the heatmap
            cam = GradCAM(model, classIdx=0)
            image = np.expand_dims(
                img_to_array(load_img(file_path, target_size=(224, 224))), axis=0
            )
            heatmap = cam.compute_heatmap(image)

            # Load the original image from disk (OpenCV format)
            orig = cv2.imread(file_path)

            # Resize the heatmap to match the original image dimensions
            heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

            # Overlay the heatmap on top of the original image
            _, output = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

            # Save the output image to a static folder
            output_image_filename = (
                secure_filename(f.filename).split(".")[0] + "_output.jpg"
            )
            output_image_path = os.path.join(
                basepath, "static", "output", output_image_filename
            )
            cv2.imwrite(output_image_path, output)

            # Return the result and image URL as JSON
            image_url = f"/static/output/{output_image_filename}"
            return jsonify({"result": result, "image_url": image_url})

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
