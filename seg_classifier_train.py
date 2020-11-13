from tensorflow import keras
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from joblib import dump
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import time
from sklearn.metrics import precision_score, recall_score
import cv2
from sklearn.base import clone
# from sklearn.model_selection import cross_val_predict
# from scipy.stats import reciprocal
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import cross_val_score

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(['Precison', 'Recall'])
    plt.grid(True)

def build_model(n_hidden=1, n_neurons=100, learning_rate=0.02, decay_rate=0.00005, dropout_val=0.2, input_shape=256):

    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    #keras.layers.AlphaDropout(rate=dropout_val),

    # Hidden Layers and Output layer
    RegularizedDense = partial(keras.layers.Dense, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0004))#, )
    for layer in range(n_hidden):
        model.add(RegularizedDense(n_neurons))
        #keras.layers.AlphaDropout(rate=dropout_val)
    model.add(RegularizedDense(1, activation="sigmoid"))

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum = 0.9, decay = decay_rate)#, momentum=0.9, decay=decay_rate)#lr=learning_rate,  momentum=0.9, nesterov=True, decay=decay_rate)
    model.compile(loss="BinaryCrossentropy", optimizer=optimizer, metrics=["accuracy", "Recall"])

    return model
random = 21

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

# RegularizedDense = partial(keras.layers.Dense, activation="selu", kernel_initializer='lecun_normal')
#
# model = keras.models.Sequential([
# keras.layers.InputLayer(input_shape=[256]),
# keras.layers.AlphaDropout(rate=0.05),
# RegularizedDense(300),
# keras.layers.AlphaDropout(rate=0.05),
# RegularizedDense(300),
# keras.layers.AlphaDropout(rate=0.05),
# RegularizedDense(1, activation="sigmoid")
# ])


# Compile Model in Keras Classifier Wrapper
model = build_model()


# Load Data
segments = np.load(r'correct data\segments_combined_0310.npy')  # 16x16 candidate segments
yt = np.loadtxt(r'correct data\bnb_combined_0310.csv',  delimiter=',')  # target classes
y = yt.astype(int)

# Flatten 16x16 into 256
segments_flat = []
for i in range(len(segments)):
    segments_flat.append(np.ndarray.flatten(segments[i]))
print(len(segments_flat))
print(len(segments_flat[0]))

# Split and Scale Data
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(segments_flat, y, test_size=0.1, shuffle=True, random_state=random)  # split into training and test data

#X_valid, X_test, y_valid, y_test = model_selection.train_test_split(X_validtest, y_validtest, test_size=0.1, shuffle=True, random_state=random)  # split test data into validation and final test data

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
#X_test = scaler.transform(X_test)


#dump(scaler, 'std_scaler_08nov1255.bin', compress=True)
#X_test = scaler.transform(X_test)


# Callback Definitions
#checkpoint_cb = keras.callbacks.ModelCheckpoint("model3_Test_1220.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True, monitor="val_loss")
# exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
# lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# Training


t0 = time.process_time()
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stopping_cb])
t1 = time.process_time()

#keras.models.save_model(model, "model3_Test_1255.h5")



print('-----------')
print(str(t1-t0) + 'seconds')



# history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid), class_weight={0: 1., 1:2.6}, callbacks=[checkpoint_cb, early_stopping_cb])
#history = model.fit(X_train, y_train, epochs=300, validation_data=(X_valid, y_valid), class_weight={0: 1., 1: 2.26}, callbacks=[checkpoint_cb, early_stopping_cb])


# Plot Training Statistics

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()


# Evaluation

#model.evaluate(X_test, y_test)

# print('-----')

