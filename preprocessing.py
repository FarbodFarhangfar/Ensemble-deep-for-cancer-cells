import pandas
import torch
import os
import numpy as np
import imagecodecs
import pickle
import tifffile
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os.path
import h5py


def open_h5(filename):
    with h5py.File(filename, "r") as f:
        # List all groups
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])

    return data


def preprocess():
    class Mydataset(torch.utils.data.Dataset):
        def __init__(self, data, transform=None):
            self.x = data
            self.len = len(data)
            # self.transform = transform

        def __getitem__(self, index):
            sample = self.x[index]
            if self.transform:
                sample = self.transform(sample)

            return sample

        def __len__(self):
            return self.len

    class feature_extraction(nn.Module):
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

    resnet = torchvision.models.resnet152(pretrained=True)
    efficientnet = torchvision.models.efficientnet_b7(pretrained=True)
    regnet = torchvision.models.regnet_y_32gf(pretrained=True)

    model = feature_extraction(resnet, efficientnet, regnet)
    model.cuda()

    def crop_data_came():
        train_data_path = "D:/data/train_images"
        train_label_path = "D:/data/pickle data"

        result_data = []
        max_size = 0

        labels = open_h5(train_label_path)
        data = open_h5(train_data_path)

        print("loading files complate")
        """convert images into 224 size patches and cropping white spots"""

        result_data = []
        max_size = 0

        with torch.no_grad():

            num = len(data)
            counter = 0

            if counter % 500 == 0:
                print(counter)
            for img in data:

                img_shape = img.shape

                value = [255, 255, 255, 255]

                if not img_shape[0] % 24 == 0:
                    img = cv2.copyMakeBorder(img, (24 - img_shape[0] % 24) // 2, (24 - img_shape[0] % 24) // 2,
                                             0,
                                             0,
                                             cv2.BORDER_CONSTANT, None, value)
                    if not img.shape[0] % 24 == 0:
                        img = cv2.copyMakeBorder(img, 1, 0, 0, 0, cv2.BORDER_CONSTANT, None, value)

                if not img_shape[1] % 24 == 0:
                    img = cv2.copyMakeBorder(img, 0, 0, (24 - img_shape[1] % 24) // 2,
                                             (24 - img_shape[1] % 24) // 2,
                                             cv2.BORDER_CONSTANT, None, value)
                    if not img.shape[1] % 24 == 0:
                        img = cv2.copyMakeBorder(img, 0, 0, 1, 0, cv2.BORDER_CONSTANT, None, value)

                images = []
                for h in range(0, img_shape[0], 24):
                    for w in range(0, img_shape[1], 24):
                        temp = img[h:h + 24, w:w + 24]

                        if not (temp == 255).all():
                            images.append(temp)

                if len(images) == 0:
                    continue
                images = np.array(images)

                images = torch.FloatTensor(images)
                images = images.permute(0, 3, 1, 2)

                del img, temp

                # images = Mydataset(images)
                loader = DataLoader(
                    images,
                    batch_size=1,
                    num_workers=1,
                    shuffle=True,
                )

                result = []
                for image_data in loader:
                    image_data = image_data.cuda()
                    output = model(image_data)

                    result.append(output)

                result = torch.stack(result)

                result = result.view(-1)
                result = result.unsqueeze(1)
                n_cluster = 12

                result = result.cpu()
                kmeans = KMeans(n_clusters=n_cluster).fit(result)

                clustered_result = [[] for _ in range(n_cluster)]

                for i, l in enumerate(kmeans.labels_):
                    clustered_result[l].append(result[i])

                del kmeans

                t = []
                for j in clustered_result:
                    j = torch.stack(j)
                    j = j.cuda()

                    j = torch.unsqueeze(j, 0)
                    t.append(j.permute(0, 2, 1))

                result = t.copy()
                del t, i, j

                with open("data", 'wb') as f:
                    pickle.dump(result, f)
            counter = counter + 1

    def crop_data_tiff():
        folder_path = "D:/data/train_images"
        result_path = "D:/data/pickle data"

        csv_file_path = "D:/data/train.csv"

        csv_file = pandas.read_csv(csv_file_path)

        """open csv file and get labels"""

        data = []

        for files in os.listdir(folder_path):

            if files.endswith(".tiff"):
                label = csv_file.loc[csv_file['image_id'] == files[:-5], "isup_grade"].iloc[0]
                with torch.no_grad():
                    label = torch.from_numpy(np.array(label)).long()
                    label = torch.unsqueeze(label, 0)
                data.append((os.path.join(folder_path, files), label))
        del files, csv_file, csv_file_path
        print("loading files complate")
        """convert images into 224 size patches and cropping white spots"""

        result_data = []
        max_size = 0
        with torch.no_grad():

            num = len(data)
            counter = 0
            for tiff_images in data:

                if counter % 500 == 0:
                    print(counter)

                file_path = tiff_images[0]
                label = tiff_images[1]

                filename = os.path.basename(file_path)
                filename = os.path.splitext(filename)[0]
                filename = result_path + "/" + filename + ".pkl"
                if not os.path.isfile(filename):
                    img = tifffile.imread(file_path)

                    img_shape = img.shape

                    value = [255, 255, 255, 255]

                    if not img_shape[0] % 224 == 0:
                        img = cv2.copyMakeBorder(img, (224 - img_shape[0] % 224) // 2, (224 - img_shape[0] % 224) // 2,
                                                 0,
                                                 0,
                                                 cv2.BORDER_CONSTANT, None, value)
                        if not img.shape[0] % 224 == 0:
                            img = cv2.copyMakeBorder(img, 1, 0, 0, 0, cv2.BORDER_CONSTANT, None, value)

                    if not img_shape[1] % 224 == 0:
                        img = cv2.copyMakeBorder(img, 0, 0, (224 - img_shape[1] % 224) // 2,
                                                 (224 - img_shape[1] % 224) // 2,
                                                 cv2.BORDER_CONSTANT, None, value)
                        if not img.shape[1] % 224 == 0:
                            img = cv2.copyMakeBorder(img, 0, 0, 1, 0, cv2.BORDER_CONSTANT, None, value)

                    images = []
                    for h in range(0, img_shape[0], 224):
                        for w in range(0, img_shape[1], 224):
                            temp = img[h:h + 224, w:w + 224]

                            if not (temp == 255).all():
                                images.append(temp)

                    if len(images) == 0:
                        continue
                    images = np.array(images)

                    images = torch.FloatTensor(images)
                    images = images.permute(0, 3, 1, 2)

                    del img, temp, tiff_images

                    # images = Mydataset(images)
                    loader = DataLoader(
                        images,
                        batch_size=1,
                        num_workers=1,
                        shuffle=True,
                    )

                    result = []
                    for image_data in loader:
                        image_data = image_data.cuda()
                        output = model(image_data)

                        result.append(output)

                    result = torch.stack(result)

                    result = result.view(-1)
                    result = result.unsqueeze(1)
                    n_cluster = 12

                    result = result.cpu()
                    kmeans = KMeans(n_clusters=n_cluster).fit(result)

                    clustered_result = [[] for _ in range(n_cluster)]

                    for i, l in enumerate(kmeans.labels_):
                        clustered_result[l].append(result[i])

                    del kmeans

                    t = []
                    for j in clustered_result:
                        j = torch.stack(j)
                        j = j.cuda()

                        j = torch.unsqueeze(j, 0)
                        t.append(j.permute(0, 2, 1))

                    result = t.copy()
                    del t, i, j

                    with open(filename, 'wb') as f:
                        pickle.dump((result, label), f)
                counter = counter + 1


if __name__ == "__main__":
    preprocess()

# finding mean and std
"""mean = 0.
    std = 0.
    nb_samples = 0.
    with torch.no_grad():
        for data in result_data:
            data = data[0]

            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

    del data, batch_samples

    compose = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=mean, std=std)])"""
