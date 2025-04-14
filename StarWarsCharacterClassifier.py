import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# deep learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image

import cv2
import warnings
warnings.filterwarnings('ignore')




# datasets
labels = pd.read_csv("TrainingLabels.csv")
labels2 = pd.read_csv("ValidLabels.csv")

#Img folder path
train_path = "Train_Images_Resized"
validation_path = "Validation_Images_Resized"

print(labels.columns)




# Data agumentation and pre-processing using tensorflow
gen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip = True
)

#validation_split=0.2 # training: 80% data, validation: 20% data

#Training Data
train_generator = gen.flow_from_dataframe(
    labels, # dataframe
    directory = train_path, # images data path / folder in which images are there
    x_col = 'ID',
    y_col = 'Character',
    color_mode="rgb",
    target_size = (331,331), # image height , image width
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    seed=42,
)

#Validation Data
validation_generator = gen.flow_from_dataframe(
    labels2, # dataframe
    directory = validation_path, # images data path / folder in which images are there
    x_col = 'ID',
    y_col = 'Character',
    color_mode="rgb",
    target_size = (331,331), # image height , image width
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    seed=42,
)



#display characters and their classes
x,y = next(train_generator)
x1,y1 = next(validation_generator)

print(x.shape)
print(x1.shape)

a = train_generator.class_indices
class_names = list(a.keys())  # storing class/breed names in a list

plt.figure(figsize=[15, 10])
for i in range(12):
    plt.subplot(5, 5, i+1)
    plt.imshow(x[i])
    plt.title(class_names[np.argmax(y[i])])
    plt.axis('off')

#plt.show()



# load the InceptionResNetV2 architecture with imagenet weights as base
base_model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(331,331,3)
    )

base_model.trainable=False

# Build the model and pass an input with a defined shape 
# so the model can infer the shapes of all the layers
input_tensor = tf.keras.Input(shape=(331,331,3))
output_tensor = base_model(input_tensor)


# Now build the rest of the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()



early = tf.keras.callbacks.EarlyStopping(patience=10,
    min_delta=0.001,
    restore_best_weights=True)
# early stopping call back

batch_size=16
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

# fit model
history = model.fit(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=10,
    callbacks=[early])

model.save("Model.keras", save_format="keras")


accuracy = model.evaluate(validation_generator)

print("Accuracy: {:.4f}%".format(accuracy[1] * 100)) 
print("Loss: ",accuracy[0])
