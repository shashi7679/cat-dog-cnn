import tensorflow as tf

original_model = tf.keras.models.load_model("catdog.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(original_model)

tflite_model = converter.convert()

print("Model converted sucessfully!!")

file = open('tflite_model.tflite','wb')
file.write(tflite_model)