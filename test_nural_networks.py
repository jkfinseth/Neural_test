import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

#Load the dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#Normalize the images, changing pixel values from [0,255] to [0,1]
train_images = (train_images/255)
test_images = (test_images/255)

#Flatten images to pass into neural network
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

print(train_images.shape)
print(test_images.shape)

#Build the model
model = Sequential()
model.add( Dense(128, activation='relu', input_dim=784))
model.add( Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compile model
#Loss: how well the model did on training, tries to improve using optimizer
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics = ['accuracy']
    )

#Train the model
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs = 10
    )

#Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
    )

predictions = model.predict(test_images[:5])
#print our models prediction
print(np.argmax(predictions, axis = 1))
print(test_labels[:5])
