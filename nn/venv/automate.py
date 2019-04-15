import argparse as ap
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.constraints import max_norm
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from math import floor
from numpy import round

def initialize_parser():
    parser = ap.ArgumentParser()

    parser.add_argument("-f", "--file",
                        help="File to process",
                        default='basic.h5')
    parser.add_argument("-a", "--action", type=int, choices=[0, 1],
                        help="Action performed",
                        default=0)
    parser.add_argument("-t", "--type", type=int, choices=[0,1],
                        help="Type of problem",
                        default=0)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of epochs",
                        default=250)

    return parser


def separate_data(data, train_percentage=70, test_percentage=20):
    """
    Separates the data into a training, test and validation sets
    according to two input parameters assigning the percentage split
    :param data:
    :param train_percentage: How much of the data to split into train set
    :param test_percentage: How much of the data to split into test set
    :return: train, validation, test
    """
    validation_percentage = 100 - train_percentage - test_percentage

    separator_one = floor(train_percentage * len(data) / 100)
    separator_two = floor((train_percentage + validation_percentage) * len(data) / 100)

    np.random.shuffle(data)

    train = data[:separator_one, :]
    validation = data[separator_one:separator_two, :]
    test = data[separator_two:, :]

    return train, validation, test

def task(data, file, action, type, epochs=250):
    train, validation, test = separate_data(data)

    train_in = train[:, 0:11]
    validation_in = validation[:, 0:11]
    test_in = test[:, 0:11]

    if type == 0:
        train_out = to_categorical(train[:, 11], num_classes=3)
        validation_out = to_categorical(validation[:, 11], num_classes=3)
        test_out = to_categorical(test[:, 11], num_classes=3)
    else:
        train_out = train[:, 11]
        validation_out = validation[:, 11]
        test_out = test[:, 11]

    model = Sequential()
    model.add(Dense(164,activation='relu',input_shape=(11,)))
    model.add(Dense(164, activation='relu', kernel_constraint=max_norm(3.)))
    model.add(Dense(164, activation='relu', kernel_constraint=max_norm(3.)))

    if type == 0:
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae'])

    if action==0:
        train_model(model, train_in, train_out,
                    validation_in, validation_out,
                    test_in, test_out, file, epochs)

    if action==1:
        test_model(model, test_in, test_out, type, file)
    return

def train_model(model, train_in, train_out,
                validation_in, validation_out,
                test_in, test_out,
                file, epochs=250):
    model.fit(train_in, train_out,
              validation_data=(validation_in, validation_out),
              batch_size=64,
              nb_epoch=epochs,
              verbose=1)

    model.save_weights(file)
    test_model(model, test_in, test_out, 0, file)
    return


def test_model(model, test_in, test_out, type, file=''):
    model.load_weights(file)
    score = model.evaluate(test_in, test_out, batch_size=64, verbose=1)
    print(score)

    predictions = model.predict(test_in)
    if type == 0:
        acc = accuracy_score(test_out, predictions.round())
    else:
        acc = r2_score(test_out, predictions)

    print(acc)
    return


def main():
    parser = initialize_parser()
    args = parser.parse_args()
    #args.file, args.action, args.type

    df = pd.read_csv('winequality-white.csv', sep=';', header=0)
    df.iloc[:, 0:11] = (df.iloc[:, 0:11] - df.min()) / (df.max() - df.min())

    # Transforming target values according to problem
    if args.type==0:
        df.quality = df.quality.map(lambda x: 0 if x <= 5 else 1 if x == 6 else 2)
    else:
        df.iloc[:,11] = (df.iloc[:, 11] - 3) / (9 - 3)

    data = df.values
    task(data, args.file, args.action, args.type, args.epochs)
    return


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main()
