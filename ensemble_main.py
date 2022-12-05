import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchmetrics
from scipy.stats import nbinom
from tqdm import tqdm


class MI_FCN(nn.Module):
    def __init__(self, n_cluster):
        super().__init__()
        self._n = n_cluster
        self.MI_FCN = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(64 * self._n, 128),
            nn.Tanh(),
            nn.Linear(128, 64)
        )
        self.out = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(512, 50176),
            nn.ReLU()
        )

    def forward(self, input):
        temp = []
        for minibatch in input:
            r = []
            for pheno in minibatch:
                pheno = torch.squeeze(pheno, dim=1)
                t = self.MI_FCN(pheno)

                t = t.mean(2)
                r.append(t)

            r = torch.stack(r, 1)
            r = torch.flatten(r, start_dim=1)

            r = r.unsqueeze(1)

            a = nn.functional.softmax(self.attention(r), dim=2)

            a = torch.transpose(a, 1, 2)
            output = torch.bmm(a, r)
            output = self.out(output)

            output = output.reshape(3, 224, 224)
            temp.append(output)
        output = torch.stack(temp)
        return output


class ensemble(nn.Module):
    def __init__(self, pre_model, models, base_learner):
        super().__init__()
        self.pre_model = pre_model
        self.models = models
        self.base_learner = base_learner

        self.out_linear = nn.Sequential(
            nn.Linear(1000, 50176),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.ReLU(),
        )

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.pre_model(x)
        temp = []
        for model in self.models:
            o = model(x)
            if not torch.is_tensor(o):
                o = o[-1]

            o = self.out_linear(o)
            temp.append(o)

        x = torch.cat((temp[0], temp[1], temp[2]), 1)
        del temp

        x = x.reshape(x.size(0), 3, 224, 224)
        x = self.base_learner(x)

        x = self.linear(x)

        num_dims = len(x.shape)
        n, p = torch.unbind(x, -1)

        n = torch.unsqueeze(n, -1)
        p = torch.unsqueeze(p, -1)

        n = self.softplus(n)
        p = self.sigmoid(p)

        x = torch.cat((n, p), dim=num_dims - 1)

        return x


def make_batch(clustered_data, batch_size=10):
    temp = []
    labels = []

    for i in range(0, len(clustered_data), batch_size):
        temp.append([_[0] for _ in clustered_data[i:i + batch_size]])
        labels.append([_[1][0] for _ in clustered_data[i:i + batch_size]])

    return temp, labels


def likelihood_loss(output, target):
    n, p = torch.unbind(output, -1)

    n = torch.unsqueeze(n, -1)
    p = torch.unsqueeze(p, -1)

    nll = []
    nll = torch.lgamma(n) + torch.lgamma(target + 1) - torch.lgamma(n + target) - n * torch.log(p) - target * torch.log(
        1 - p)

    nll = torch.mean(nll)

    return nll


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __getitem__(self, idx):
        y = torch.Tensor(self.y[idx])

        return self.x[idx], y

    def __len__(self):
        return self.len


if __name__ == "__main__":
    #    with open('/content/drive/MyDrive/data/forcast/clustered_data.pickle', 'rb') as file_path:
    #        clustered_data = pickle.load(file_path)

    # clustered_data, labels = make_batch(clustered_data, 10)

    models = []

    model2 = torchvision.models.densenet201()
    model2 = model2.cuda()
    models.append(model2)

    from condensenet import CondenseNet

    model2 = CondenseNet()
    model2.cuda()
    models.append(model2)

    from msdnet import MSDNet

    model2 = MSDNet()
    model2.cuda()
    models.append(model2)

    base_learner = torchvision.models.vgg19_bn(pretrained=False, progress=False)

    n_cluster = 10

    pre_model = MI_FCN(n_cluster)
    pre_model = pre_model.cuda()

    base_lr = 0.001
    weight_decay = 1e-4
    momentum = 1e-4

    model = ensemble(pre_model, models, base_learner)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    accuracy = torchmetrics.Accuracy()
    accuracy = accuracy.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    # dataset = Mydataset(clustered_data, labels)
    # loader = torch.utils.data.DataLoader(
    #    dataset,
    #    shuffle=False
    # )

    train_acc = []

    epochs = 10

    loss_data = []


"""    for epoch in range(epochs):
        outputs = []
        for input, target in tqdm(loader):
            target = target.cuda()

            optimizer.zero_grad()
            output = model(input)

            output = output.cuda()
            loss = likelihood_loss(output, target)
            target = torch.reshape(target, (output.shape[0], 1))

            pred = output.cpu().data.numpy()

            n = pred[:, 0]
            p = pred[:, 1]

            y_pred = nbinom.median(n, p)
            y_pred = torch.from_numpy(y_pred).cuda()

            outputs.append(y_pred)

            target = target.to(device, torch.int)
            y_pred = torch.unsqueeze(y_pred, 1)
            acc = accuracy(y_pred, target)
            train_acc.append(acc)

            loss.backward()
            optimizer.step()
            loss_data.append(loss.data.item())
        outputs = (epoch, outputs, loss.data.item())
        with open(f"/{epoch}.pickle") as file_path:
            pickle.dump(outputs, file_path)
        PATH = f"/ensemble_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)

        print(f"epoch: {epoch}")
        print("loss : ", loss.data.item())
        acc = accuracy.compute()
        print(f"Accuracy on all data: {acc}")
        print()"""
