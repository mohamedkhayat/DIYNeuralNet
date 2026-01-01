import pytest
from diy_neural_net import get_numpy, NeuralNetwork
from diy_neural_net.Layers import Dense
from diy_neural_net.Activations import ReLU
from diy_neural_net.Losses import MSELoss
from diy_neural_net.Optimizer import Adam

np = get_numpy()


def test_overfitting_sanity_check():
    """
    A small network should be able to perfectly memorize a tiny dataset.
    If loss doesn't decrease, something is broken.
    """
    # 1. Create tiny dataset (XOR-like logic)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Shape (2, 4)

    y = np.array([[0], [1], [1], [0]]).T  # Shape (1, 4)

    # 2. Define Model
    model = NeuralNetwork.Sequential(
        [Dense(2, 8, initializer="he"), ReLU(), Dense(8, 1, initializer="random")]
    )

    model.set_loss(MSELoss())

    # 3. Optimize
    # Use a high learning rate to ensure quick convergence for this test
    optimizer = Adam(model, lr=0.05)

    # 4. Train
    history = model.fit(
        X, y, optimizer=optimizer, epochs=100, batch_size=4, shuffle=False
    )

    # 5. Check if loss decreased significantly
    initial_loss = history["Train_losses"][0]
    final_loss = history["Train_losses"][-1]

    print(f"Initial Loss: {initial_loss}, Final Loss: {final_loss}")

    assert final_loss < initial_loss
    assert final_loss < 0.1  # It should be very close to 0
