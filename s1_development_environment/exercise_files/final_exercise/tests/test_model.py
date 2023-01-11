import torch

def test_model():

    from model import MyAwesomeModel

    train_data = torch.randn(1,28,28)
    model = MyAwesomeModel()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.train()

    output = model(train_data)

    assert output.shape == (1,10)

    