import torch
import torch.nn as nn
import torch.optim as optim

# Given data
X = torch.tensor([[-2, 4, 7, -2]], dtype=torch.float32)
y = torch.tensor([5], dtype=torch.float32)

# Initial weights
w = torch.tensor([0.1, 0.2, 0.15, 0.7, 0.21, -0.3], dtype=torch.float32, requires_grad=True)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.1, 0.2, 0.15, 0.7, 0.21, -0.3], dtype=torch.float32))
    
    def forward(self, x):
        h1 = x[:, 0] * self.w[0] + x[:, 1] * self.w[2] + x[:, 2] * self.w[4] + x[:, 3] * self.w[5]
        h2 = x[:, 0] * self.w[1] + x[:, 1] * self.w[3] + x[:, 2] * self.w[4] + x[:, 3] * self.w[5]
        y_pred = h1 * self.w[4] + h2 * self.w[5]
        return y_pred

# Instantiate model, loss, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Forward pass
y_pred = model(X)
loss = criterion(y_pred, y)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print updated weights
print("Updated Weights:")
for param in model.parameters():
    print(param.data)
