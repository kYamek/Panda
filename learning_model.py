from model import FeedForward
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def plot_displacement_vs_l1_norm(inputs, outputs):
    displacements = []
    errors = []

    test_dataset = TensorDataset(torch.tensor(inputs), torch.tensor(outputs))
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=32, shuffle=False)

    for i, data in enumerate(test_loader):
        X, y = data
        pred = model(X).detach().numpy()
        for j in range(pred.shape[0]):
            disp = np.linalg.norm(np.array(X[j, :2]) - np.array(y[j, :2]))
            err = np.linalg.norm(np.array(pred[j, :]) - np.array(y[j, :2]))
            displacements.append(disp)
            errors.append(err)

    bins = np.arange(0, 0.31, 0.01)
    inds = np.digitize(displacements, bins=bins)
    # create mapping from bins to list of values
    bin_values = defaultdict(list)
    for i in range(len(displacements)):
        bin_values[bins[inds[i] - 1]].append(errors[i])

    # only plot bins with values
    bins_plt = np.array([b for b in bins if bin_values[b]])
    means = np.array([np.mean(bin_values[b]) for b in bins if bin_values[b]])
    std = np.array([np.sqrt(np.var(bin_values[b])) for b in bins if bin_values[b]])

    plt.plot(bins_plt, means, '-', color="black")
    plt.fill_between(bins_plt, means - std, means + std, color='gray', alpha=0.4)
    plt.scatter(displacements, errors)
    plt.xlabel("Displacement (m)")
    plt.ylabel("L1 Norm")
    plt.show()


if __name__ == "__main__":
    model_name = "position_model.pt"
    model = FeedForward(8, 2)
    model.load_state_dict(torch.load("models/" + model_name))
    model.train(False)

    # load test data set
    test_start_pose = np.load("data/test_start_pose.npy")
    test_contour_pt = np.load("data/test_contour_pt.npy")
    test_ee_vel = np.load("data/test_ee_vel.npy")
    test_X = np.concatenate([test_start_pose, test_contour_pt, test_ee_vel], axis=1)
    test_end_pose = np.load("data/test_end_pose.npy")

    plot_displacement_vs_l1_norm(test_X, test_end_pose)
