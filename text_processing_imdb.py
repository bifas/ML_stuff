import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

LOADED_DATA = False

def convert_ints_to_words(imdb):
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return word_index, reverse_word_index


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def build_model():
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16)) # first layer
    model.add(keras.layers.GlobalAveragePooling1D()) # returns a fixed-length output vector for
    # each  example by averaging over the sequence dimension
    model.add(keras.layers.Dense(16, activation=tf.nn.relu)) # layer with 16 units fully connected
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) #output layer, 1 value between 0
    # and 1 representing prob
    model.summary()

    #optimizer and loss function
    model.compile(optimizer='adam', loss= "binary_crossentropy", metrics=['acc'])

    return model


def display_data(history, type= "acc"):
    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    if type == "loss":
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    elif type == "acc":
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        print("wrong type")

    plt.show()


def main():
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    # print(train_data[0])
    print(len(train_data), len(train_data[1]))

    word_index, reverse_word_index = convert_ints_to_words(imdb)
    text = decode_review(train_data[0],reverse_word_index)
    print(text)

    #  since the arrays have different sizes er must pad the arrays to all of them have the same
    # length. -> pad_sequences_function to standardize the length

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)
    print(train_data[0])

    # Building model
    model = build_model()

    # Create validation set
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # Train the model (40 epochs, batched of 512 samples)
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)


    # Evaluate the model

    results = model.evaluate(test_data, test_labels)
    print(results)

    display_data(history)


main()