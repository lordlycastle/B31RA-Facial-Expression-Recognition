import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import os
import pickle
# import dill as pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Conv2D, Dropout, MaxPooling2D
from keras.utils import to_categorical
from keras import metrics
from keras.callbacks import *
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, model_from_json



def get_data_and_labels(df):
    test_data = df.as_matrix(columns=['pixels'])[:, 0]
    test_data = np.array(test_data.tolist())
    # train_data = train_data.reshape(-1, 48, 48)
    test_data = test_data.astype('float32')
    test_data /= 255
    # print(test_data[0])
    test_label = df.as_matrix(columns=['emotion'])
    test_label.shape
    test_label_1hot = to_categorical(test_label)
    test_label_1hot.shape
    return test_data, test_label_1hot, test_label

training = pd.read_pickle('/data/training_set.pkl')
full_test = pd.read_pickle('/output/full_test.pkl')
train_data, train_label_1hot, train_label = get_data_and_labels(training)
test_data, test_label_1hot, test_label = get_data_and_labels(full_test)

train_data_2d = np.empty(shape=(len(train_data), 48, 48))
for i in range(len(training_2d)):
    im = np.reshape(train_data[i], (48, 48))
    training_2d[i] = im
train_data_2d = training_2d.reshape((len(train_data), 48, 48, 1))



def save_model(model, name, history):
    model.save(name)
    with open(name + '.p', 'wb') as history_file:
        pickle.dump(history.history, history_file)
    with open(name + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    return


def load_model_(name):
    model = load_model(name)
    with open(name + '.p', 'rb') as history_file:
        history = pickle.load(history_file)
    return model, history


def plot_history(history):
    # Plot the Loss Curves
    plt.figure(figsize=[8, 3])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=11)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Plot the Accuracy Curves
    plt.figure(figsize=[8, 3])
    plt.plot(history.history['categorical_accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_categorical_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=11)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
        log_string = '\n'
        for key in logs:
            log_string = log_string + '{{"metric":"{}", "value":{}}}\n'.format(
                key, logs[key])
        
        sys.__stdout__.write(str(log_string) + "\n")


plot = PlotLearning()

nClasses = len(train_label_1hot[0])
dimData = len(train_data[0])


cnn_model = Sequential()
cnn_model.add(Reshape((48, 48, 1), input_shape=(dimData, )))
cnn_model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     # input_shape=(48, 48, 1),
                     kernel_regularizer=regularizers.l2(0.001)
                     ))
cnn_model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)
                    ))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              kernel_regularizer=regularizers.l2(0.001)
             ))
cnn_model.add(Conv2D(64, (3, 3), activation='relu',
             kernel_regularizer=regularizers.l2(0.001)
             ))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                    ))
cnn_model.add(Conv2D(64, (3, 3), activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)
                    ))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(512, activation='relu',
#               kernel_regularizer=regularizers.l2(0.001)
                   )
             )
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(512, activation='relu',
#                    kernel_regularizer=regularizers.l2(0.001)
                   )
             )
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(nClasses, activation='softmax'))

cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=[metrics.categorical_accuracy, 'acc'],)


cnn_history = cnn_model.fit(train_data, train_label_1hot, batch_size=1024,
                             epochs=100,
                             validation_data=(test_data, test_label_1hot),
                             verbose=True,
                            callbacks=[plot, ReduceLROnPlateau()],
                            )

save_model(cnn_model, '/output/cnn_model', cnn_history)