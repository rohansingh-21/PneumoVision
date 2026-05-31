# pyrefly: ignore [missing-import]
import tensorflow as tf
from functools import reduce

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    """Generates a Grad-CAM heatmap for the given input image.

    Uses the gradient of the output w.r.t. the last conv layer
    to produce a class activation map highlighting important regions.
    """
    base_model = model.layers[0]

    base_model_ext = tf.keras.models.Model(
        base_model.inputs,
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    head_input = tf.keras.Input(shape=base_model.output.shape[1:])
    head_model = tf.keras.models.Model(
        head_input,
        reduce(lambda x, layer: layer(x), model.layers[1:], head_input)
    )

    with tf.GradientTape() as tape:
        last_conv_output, base_preds = base_model_ext(img_array, training=False)
        tape.watch(last_conv_output)
        class_channel = head_model(base_preds, training=False)[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.squeeze(last_conv_output[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()
