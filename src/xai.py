import numpy as np
import tensorflow as tf
from PIL import Image
from .settings import LAST_CONV_LAYER_NAME

def grad_cam(model, img_batch, last_conv_name=LAST_CONV_LAYER_NAME):
    # Extract last conv layer
    last_conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_batch)

        # Ensure preds shape (num_classes,)
        preds = tf.squeeze(preds)  

        # ✅ predicted class index
        class_idx = int(tf.argmax(preds).numpy())

        # ✅ class logit/score for Grad-CAM
        score = preds[class_idx]

    # ✅ compute gradients
    grads = tape.gradient(score, conv_output)[0]

    # ✅ average gradients (global average pooling)
    weights = tf.reduce_mean(grads, axis=(0, 1))

    conv_output = conv_output[0]
    cam = tf.reduce_sum(conv_output * weights, axis=-1)

    # ✅ normalize heatmap
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = cam.numpy()

    # ✅ resize heatmap to input size
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(
        (img_batch.shape[2], img_batch.shape[1])
    )

    # return (heatmap image, predicted class index, raw preds array)
    return cam_img, class_idx, preds.numpy()
