from DeviceSelector import get_numpy, is_gpu_available
import utils
from Network import NeuralNetwork
from Losses import BCELoss, CrossEntropyLoss, MSELoss
from Layers import Dense, Dropout
from Activations import ReLU, Sigmoid, Softmax

np = get_numpy()
_GPU_AVAILABLE = is_gpu_available()

# Setting random seed for reproducibility

np.random.seed(42)

# type of problem :
# 1 for binary classification on a modified version of mnist
# 2 for multi class classification on mnist
# 3 for regression, not yet implemented

problem = 1
# Loading Mnist data
if __name__ == "__main__":
    # need to make all of this a function
    try:
        if problem == 3:
            print("loading data : Regression Data")
            n_samples = 30000
            n_features = 1
            X, y = utils.generate_regression_data(
                n_samples=n_samples, n_features=n_features, noise=0.005, np=np
            )

        elif problem == 2:
            print("loading data : MNIST")
            X, y = utils.load_mnist()

        else:
            print("loading data : binary MNIST")
            X, y = utils.load_binary_mnist()
            problem = 1

    except Exception as e:
        print(e)
        # Falling back to generating XOR in case of errors
        print("Falling back to xor")
        problem = 1
        n_samples = 2000
        X, y = utils.generate_xor_data(n_samples, np)

    n_features = X.shape[0]
    n_classes = y.shape[0]
    # specifying percantage of data to be used for validation

    ratio = 0.2

    # if GPU is available, transform X and y to cupy ndarray
    # effectivaly loading the data into the gpu

    if _GPU_AVAILABLE:
        X, y = np.asarray(X), np.asarray(y)

    # split data into train and validation data
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=ratio)

    dropout_p = 0.8

    # Defining the architecture of our neural network
    layers = [
        Dense(
            input_size=n_features, output_size=512, initializer="he"
        ),  # Input layer, input size = n_features, output_size (n of units) = 64, HE init because it uses ReLU
        ReLU(),  # ReLU Activation Function
        Dense(
            input_size=512, output_size=256, initializer="he"
        ),  # First hidden layer, input size = 64, output size = 64, he init too because it uses ReLU
        ReLU(),  # ReLU again
        Dropout(
            keep_prob=dropout_p
        ),  # Dropout layer, turns off (1 - keep_prob) * 100 % of units
        Dense(
            input_size=256, output_size=128, initializer="he"
        ),  # Second Hidden layer, input size = 64, output size = 32, he init again because it uses ReLU
        ReLU(),  # relu again
        Dense(
            input_size=128, output_size=32, initializer="he"
        ),  # Third Hidden layer input size = 32, output size = 32 he init again
        ReLU(),  # relu again
        Dropout(
            keep_prob=dropout_p
        ),  # Dropout layer, turns off (1 - keep_prob) * 100 % of units
    ]

    if problem == 1:
        learning_rate = 1e-2
        loss = BCELoss()

        layers.append(
            Dense(input_size=32, output_size=n_classes, initializer="glorot")
        )  # Output layer, input size = 32, output size = n_classes (1), glorot init because it uses sigmoid
        layers.append(Sigmoid())

    elif problem == 2:
        learning_rate = 0.1
        loss = CrossEntropyLoss()

        layers.append(
            Dense(input_size=32, output_size=n_classes, initializer="glorot")
        )  # Output layer, input size = 32, output size = n_classes (multiple), glorot init because it uses softmax
        layers.append(Softmax())

    else:
        learning_rate = 0.0003
        loss = MSELoss()

        layers.append(
            Dense(input_size=32, output_size=n_classes, initializer="random")
        )  # Output layer, input size = 32, output size = n_classes (1), random init because it uses no activation function

    model = NeuralNetwork(
        n_classes=n_classes,  # Needed
        layers=layers,  # Needed
        learning_rate=learning_rate,  # Needed
        criterion=loss,  # Needed
    )

    # Training the model
    print("starting training")

    History = model.fit(
        X_train=X_train,  # Needed
        y_train=y_train,  # Needed
        batch_size=64,  # Optional, defaults to 64
        shuffle=True,  # Optional, defaults to True
        epochs=200,  # Needed
        validation_data=(X_test, y_test),  # Optional if you dont need plotting
        early_stopping_patience=15,  # Optional
        early_stopping_delta=0.001,  # Optional
    )

    # Print Time Elapsed and Device used to train

    print(
        f"\nTime Elapsed : {History['Time_Elapsed']:.2F} seconds on : {'GPU' if _GPU_AVAILABLE else 'CPU'}\n"
    )

    utils.plot_metrics(History)

    if problem != 3:
        # using the model to make predictions on the train set

        y_pred_train = model.predict(X_train)

        # using predictions to calculate model's accuracy on the train set

        train_accuracy = model.accuracy_score(y_pred_train, y_train)
        print(f"Train Accuracy : {float(train_accuracy):.4f}")

        # using the model to make predictions on the test set

        y_pred_test = model.predict(X_test)

        # using predictions to calculate model's accuracy on the test set

        test_accuracy = model.accuracy_score(y_pred_test, y_test)
        print(f"Test Accuracy : {float(test_accuracy):.4f}")

        # Plotting random n_images from the test set with their predictions
        utils.plot_image(
            X=X_test,
            model=model,
            n_images=6,
            original_image_shape=(28, 28),
            n_classes=n_classes,
        )
