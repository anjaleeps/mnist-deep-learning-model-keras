#Import Libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

#Load data
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Plot an Image
from matplotlib import pyplot as plt
plt.imshow(X_train[1])

#Normalize Images
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

#Reshape Images
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

#Encode Image Labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Create the Model
model = Sequential()

model.add(Convolution2D(32, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

#Compile the model
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

#Evaluate the Model
score = model.evaluate(X_test, y_test, verbose=0)

print("accuracy", score[1])
print("loss", score[0])

#Make Predictions
predictions = model.predict(X_test[:20])
print("predictions:", np.argmax(predictions, axis=1))
print("labels     :", np.argmax(y_test[:20], axis=1))