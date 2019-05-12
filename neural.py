from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#######################################
inputs= Input(shape=(748,)) # returns a tensor

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions =  Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training

