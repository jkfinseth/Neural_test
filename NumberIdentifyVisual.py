# Import necessary libraries
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
img_share = mnist.test_images()

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

# Create predictions using the model
predictions = model.predict(test_images)

# Create a plot for each prediction and display it to the user
for i in range(len(test_images)):
    plt.grid(False)
    plt.imshow(img_share[i], cmap=plt.cm.binary)
    # Display the actual value on the bottom
    plt.xlabel("Actual: " + str(test_labels[i]))
    # Display the prediction as well as the confidence on the top
    plt.title("Prediction: " + str(np.argmax(predictions[i])) + "\nConfidence: " + str(predictions[i][np.argmax(predictions[i])]))
    plt.show()
