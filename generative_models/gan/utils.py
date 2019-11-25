import torch
import numpy as np

def train_model(model, training_loader, optimizer, epochs, device):
    """
    Function to train model
    """

    bce = torch.nn.BCELoss()
    
    for e in range(epochs):

        data_point = np.random.randint(400)

        data, label = training_loader[data_point]

        #data, label = training_loader[0]
        data = data.to(device)
        label = label.to(device)

        #print ("label: ", label)

        out = model(data)

        optimizer.zero_grad()
    
        loss = bce(out, label)

        print ("Epoch: " , e, " loss: ", loss.item())
        loss.backward()
        optimizer.step()

        