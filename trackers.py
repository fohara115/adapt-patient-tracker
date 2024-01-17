import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class TrackerNetwork(nn.Module):
    def __init__(self, template_size):
        super(TrackerNetwork, self).__init__()
        self.template_size = template_size
        self.conv = nn.Conv2d(3, 1, kernel_size=template_size)

    def forward(self, x):
        return self.conv(x)
    

class SimpleTracker():
    def __init__(self):
        self.model = TrackerNetwork((5, 5))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.bbox = None

    def init(self, frame, bbox):
        x, y, w, h = bbox
        roi = frame[x:x+w, y:y+h]
        roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        num_training_cycles = 100
        for i in tqdm(range(num_training_cycles)):
            # Forward pass through the tracker
            output = self.model(roi_tensor)

            # Compute the loss by comparing the output with a zero-filled tensor
            target = torch.zeros_like(output)
            loss = self.loss_fn(output, target)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.bbox = (x, y, w, h)

    def update(self, frame):
        x, y, w, h = self.bbox
        roi = frame[x:x+w, y:y+h]
        roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Forward pass through the tracker
        output = self.model(roi_tensor)

        # Compute the loss by comparing the output with a zero-filled tensor
        target = torch.zeros_like(output)
        loss = self.loss_fn(output, target)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Find the position of the highest activation in the output tensor
        _, _, w_idx, h_idx = torch.where(output == torch.max(output))
        

        x = int(x + w_idx)
        y = int(y + h_idx)

        self.bbox = (x, y, w, h)

        return True, self.bbox

