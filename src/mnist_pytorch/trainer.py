import torch.nn as nn
from torch import optim
import torch
import numpy as np
from sklearn.metrics import accuracy_score


class MNISTTrainer:
    def __init__(self):
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, cnn, n_epochs, loaders):
        optimizer = optim.Adam(cnn.parameters(), lr=0.01)
        # move cnn to train
        cnn.to('cuda')

        for e in range(n_epochs):
            cnn.train()
            running_loss = []
            for i, (images, labels) in enumerate(loaders['train']):
                b_x = images.to('cuda')  # batch x
                b_y = labels.to('cuda')  # batch y

                output = cnn(b_x)[0]
                loss = self.loss_func(output, b_y)

                # clear gradients for this training step
                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, n_epochs, i + 1, loss.item()))
                running_loss.append(loss.item())
            avg_loss = np.mean(running_loss)

            cnn.eval()
            with torch.no_grad():
                accuracies = []
                for images, labels in loaders['test']:
                    b_x = images.to('cuda')  # batch x
                    b_y = labels.to('cuda')  # batch y
                    test_output = cnn(b_x)[0].detach().cpu().numpy()
                    # pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    pred_y = np.argmax(test_output, axis=-1)
                    y_true = b_y.detach().cpu().numpy()
                    # accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                    acc_it = accuracy_score(y_true, pred_y)
                    accuracies.append(acc_it)
            avg_test_acc = np.mean(accuracies)
            print(avg_loss, avg_test_acc)
