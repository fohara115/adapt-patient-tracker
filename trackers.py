import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TrackerNetwork(nn.Module):
    def __init__(self):
        super(TrackerNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(5, 5), padding='same')
        self.conv2 = nn.Conv2d(10, 1, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class SimpleTracker():
    def __init__(self):
        self.model = TrackerNetwork()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.bbox = None

    def init(self, frame, bbox):
        x, y, w, h = bbox
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        target = torch.zeros((1, 1, 480, 640))
        target[0, 0, x:x+w, y:y+h] = 1

        print(frame_tensor.shape)
        print(target.shape)

        num_training_cycles = 48
        losses = []
        for _ in tqdm(range(num_training_cycles)):
            
            output = self.model(frame_tensor)
            #print(output.shape)
            
            loss = self.loss_fn(output, target)
            losses.append(float(loss.data))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.bbox = (x, y, w, h)

        # plt.plot(losses)
        # plt.show()
        # input()

    def update(self, frame):
        x, y, w, h = self.bbox
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        output = self.model(frame_tensor)

        # try without relearning anything
        _, _, c, d = torch.where(output == torch.max(output))

        # roi = frame[x:x+w, y:y+h]
        # roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # # Forward pass through the tracker
        # output = self.model(roi_tensor)

        # # Compute the loss by comparing the output with a zero-filled tensor
        # target = torch.zeros_like(output)
        # loss = self.loss_fn(output, target)

        # # Backpropagation and optimization
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # Find the position of the highest activation in the output tensor
        


        x = c
        y = d

        self.bbox = (x, y, w, h)

        return True, self.bbox

