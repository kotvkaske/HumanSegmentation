from time import time
import torch
import numpy as np

DEVICE = torch.device('cuda')


def train(model, opt, loss_fn, epochs, data_tr, data_val, path_to_save):
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch + 1, epochs))
        avg_loss = 0
        patience = 3
        counter = 0
        max_loss = np.float('inf')
        model.train()
        for X_batch, Y_batch in data_tr:
            Y_batch = Y_batch.to(DEVICE)
            X_batch = X_batch.to(DEVICE)
            opt.zero_grad()
            outputs = torch.sigmoid(model(X_batch))
            loss = loss_fn(Y_batch, outputs)
            loss.backward()
            opt.step()

            avg_loss += loss / len(data_tr)
        toc = time()
        print('train_loss: %f' % avg_loss)
        avg_loss = 0
        model.eval()
        for X_batch, Y_batch in data_val:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            with torch.set_grad_enabled(False):
                outputs = torch.sigmoid(model(X_batch))

                loss = loss_fn(Y_batch, outputs)
            avg_loss += loss / len(data_val)
        print('val_loss: %f' % avg_loss)
        if avg_loss >= max_loss:
            print(f'сработал датчик на {epoch}-й эпохе')
            counter += 1
            if counter == patience:
                return model

        else:
            print(f'Эпоха {epoch} стабильна')
            max_loss = avg_loss
            counter = 0
            torch.save(model, path_to_save)
