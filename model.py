import numpy as np
from torch import nn
import torch


class FeedForward(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        hidden_dim = 128
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


def main():
    model = FeedForward(8, 2)
    # load model state_dict from file
    model.load_state_dict(torch.load("models/position_model.pt"))
    test_start_pose = np.load("data/test_start_pose.npy")
    test_contour_pt = np.load("data/test_contour_pt.npy")
    test_ee_vel = np.load("data/test_ee_vel.npy")
    test_X = np.concatenate([test_start_pose, test_contour_pt, test_ee_vel], axis=1)
    print(test_X.shape)
    pred = model(torch.tensor(test_X))


if __name__ == "__main__":
    main()
