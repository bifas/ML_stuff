import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential([
    Dense(32, input_shape=(100,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

#or use model.add(Dense(32, input_dim=100))
#model.add(Activation('relu'))

#configuration of the training proccess
#arsgs: optimizer, loss function, list of metrics For any classification problem you will want
# to set this to metrics=['accuracy']
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)