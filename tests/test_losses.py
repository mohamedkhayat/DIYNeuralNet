from diy_neural_net import get_numpy
from diy_neural_net.Losses import MSELoss, CrossEntropyLossWithLogits, BCELoss

np = get_numpy()


def test_mse_loss():
    criterion = MSELoss()
    y_true = np.array([[1.0, 2.0, 3.0]])
    y_pred = np.array([[1.0, 2.0, 3.0]])

    # Loss should be 0
    loss = criterion(y_true, y_pred)
    assert loss == 0.0


def test_crossentropy_shapes():
    # 3 Classes, Batch size 4
    criterion = CrossEntropyLossWithLogits()
    y_true = np.zeros((3, 4))
    y_true[0, :] = 1  # One-hot

    logits = np.random.randn(3, 4)

    loss = criterion(y_true, logits)
    assert isinstance(loss, (float, np.floating)) or (
        hasattr(loss, "ndim") and loss.ndim == 0
    )

    grad = criterion.backward(y_true, logits)
    assert grad.shape == logits.shape


def test_bce_loss_value():
    criterion = BCELoss()
    # Perfect prediction
    y_true = np.array([[1.0]])
    y_pred = np.array([[0.9999999]])

    loss = criterion(y_true, y_pred)
    assert loss < 0.01
