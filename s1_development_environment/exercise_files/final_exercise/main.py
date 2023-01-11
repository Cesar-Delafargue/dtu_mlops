import argparse
import sys

import torch
import click
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')

def train (batch_size = 128, epochs = 5, lr = 0.01):
    
    print("Training day and night")
    print("Learning rate: ", lr)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _ = mnist()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = MyAwesomeModel()
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()
        print("Epoch {i}/{j}...".format(i=epoch, j=epochs))
        overall_loss = 0
        for images, labels in train_loader:
            
            optimizer.zero_grad()
            output = model(images)
            probabilities = nn.functional.softmax(output, dim=1)
            loss = loss_fn(probabilities,labels)
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        train_loss, train_acc = compute_validation_metrics(model,train_loader)
        val_loss, val_acc = compute_validation_metrics(model,test_loader)

        print('Average loss for epoch : {i}'.format(i=overall_loss/len(train_loader)))
    
    return model  


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    