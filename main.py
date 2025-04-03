from keras.models import load_model

model = load_model("Model.keras")  # No need for custom objects!
print("\n\n\nModel loaded successfully!\n\n\n")

accuracy = model.evaluate()


