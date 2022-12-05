import gzip
import h5py
import os
import shutil
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans


def gunzip(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)


def extract_files():
    root_dir = "/content/drive/MyDrive/pcamv1"
    dist_dir = "/content/drive/MyDrive/new_data/extracted_files"

    files = os.listdir(root_dir)
    for file_name in files:
        file_src = os.path.join(root_dir, file_name)
        if file_name.endswith(".gz"):
            dist = os.path.join(dist_dir, file_name[:-3])
            gunzip(file_src, dist)
        else:
            dist = os.path.join(dist_dir, file_name)
            shutil.copyfile(file_src, dist)


def open_h5():
    root_dir = "/content/drive/MyDrive/new_data/extracted_files"
    filename_x = os.path.join(root_dir, "camelyonpatch_level_2_split_train_x.h5")
    filename_y = os.path.join(root_dir, "camelyonpatch_level_2_split_train_y.h5")

    h5_x = h5py.File(filename_x, 'r')
    h5_y = h5py.File(filename_y, 'r')
    return h5_x['x'], h5_y['y']


class Mydataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.len = len(X)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x[idx])
        y = torch.Tensor(self.y[idx])
        return torch.unbind(x, dim=2)

    def __len__(self):
        return self.len


def get_mean(data):
    mean = data.mean()
    meansq = (data ** 2).mean()

    return mean, meansq


def mean_cal():
    mean0 = 0.
    meansq0 = 0.

    mean1 = 0.
    meansq1 = 0.

    mean2 = 0.
    meansq2 = 0.

    for data in tqdm(loader):
        mean0, meansq0 = get_mean(data[0])
        mean1, meansq1 = get_mean(data[1])
        mean2, meansq2 = get_mean(data[2])

    std0 = torch.sqrt(meansq0 - mean0 ** 2)
    std1 = torch.sqrt(meansq1 - mean1 ** 2)
    std2 = torch.sqrt(meansq2 - mean2 ** 2)

    mean = (mean0, mean1, mean2)
    std = (std0, std1, std2)

    return mean, std


class Mydataset_mean(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.len = len(x)
        self.transform = transform

    def __getitem__(self, idx):
        x = torch.Tensor(self.x[idx])
        y = torch.Tensor(self.y[idx])
        x = torch.reshape(x, (3, x.size()[0], x.size()[1]))
        y = torch.reshape(y, (1,))

        if self.transform:
            sample = self.transform(x)

        return sample, y

    def __len__(self):
        return self.len


class FeatureExtraction(nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.avaragepooling = nn.AvgPool1d(3)
        self.fully_conected = nn.Linear(1000, 1000)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x = torch.concat((x1, x2, x3), dim=1)

        x = self.avaragepooling(x)
        x = F.softmax(self.fully_conected(x), dim=1)

        return x


if __name__ == "__main__":
    X_train, Y_train = open_h5()
    dataset = Mydataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )

    mean, std = mean_cal()
    compose = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=mean, std=std)])
    dataset = Mydataset_mean(X_train, Y_train, compose)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    resnet = torchvision.models.resnet152(pretrained=True)
    efficientnet = torchvision.models.efficientnet_b7(pretrained=True)
    regnet = torchvision.models.regnet_y_32gf(pretrained=True)

    model = FeatureExtraction(resnet, efficientnet, regnet)
    model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = []
    n = 0
    with torch.no_grad():
        for data, label in tqdm(loader):
            data = data.to(device, dtype=torch.float32)

            output = model(data)

            result.append((output, label))
            n += 1

    del output, data
    del model, resnet, efficientnet, regnet

    with torch.no_grad():

        clustered_data = []
        for data in result:
            label = data[1]
            data = data[0].view(-1)
            data = data.unsqueeze(1)
            n_cluster = 8

            data = data.cpu()
            kmeans = KMeans(n_clusters=n_cluster).fit(data)
            clustered_result = [[] for _ in range(n_cluster)]

            for i, l in enumerate(kmeans.labels_):
                clustered_result[l].append(data[i])

            clustered_data.append((clustered_result, label))

    t = []
    tt = []
    with torch.no_grad():
        for i in clustered_data:
            t = []
            for j in i[0]:
                j = torch.stack(j)
                j = j.cuda()

                j = torch.unsqueeze(j, 0)
                t.append(j.permute(0, 2, 1))

            tt.append((t, i[1]))

        clustered_data = tt.copy()
    del t, i, tt, j

    with open("clustered_data.pickle", "wb") as file_path:
        pickle.dump(clustered_data, file_path)
