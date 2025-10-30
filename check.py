from tensorflow.keras.models import load_model
model = load_model("assets/best_model.h5", compile=False)
print(model.output_shape)
