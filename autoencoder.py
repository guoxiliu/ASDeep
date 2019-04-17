from filenames import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Activation

def deep_autoencoder(compressed_size):
    """
    The function builds the model of autoencoder, and save the generated feature to file.

    Parameters
    ---------------
    compressed_size: int
        The compressed size for autoencoder.
    """
    np.random.seed(123)

    # Read the dataset
    X_train = np.genfromtxt(data_path + norm_express_file, delimiter=',').astype(np.float32)

    # Set parameters for learning
    num_epoch = 60
    batch_size = 60 
    acti_func = "relu"

    # Set parameters for network
    num_input = 524

    input_X = Input(shape=(num_input,))
    encoded = Dense(compressed_size, activation=acti_func)(input_X)
    decoded = Dense(num_input, activation=acti_func)(encoded)
    autoencoder = Model(input_X, decoded)
    encoder = Model(input_X, encoded)

    autoencoder.compile(loss="mean_squared_error", optimizer="adam")
    autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=num_epoch, shuffle=True)
    features = encoder.predict(X_train)

    np.savetxt(feature_path + express_feature_file, features, fmt='%f', delimiter=' ')

if __name__ == "__main__":
    deep_autoencoder(256)
