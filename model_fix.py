import tensorflow as tf
model = tf.keras.models.load_model("mask_detector.h5", compile=False)
