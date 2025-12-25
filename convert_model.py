import tensorflow as tf
old_model = tf.keras.models.load_model("mask_detector.h5", compile=False)
old_model.save("mask_detector.keras")
