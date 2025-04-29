import os

import random
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def num_types_loading(path):
        icd10_ls = None
        with open(path, 'r') as fp:
            for i, line in enumerate(fp):
                if i == 0:
                    icd10_ls = line.strip().split('\t')
                    break
        return icd10_ls, len(icd10_ls)

    def load_data(path_name):
        data = []
        with open(path_name, 'r') as fp:
            for line in fp:
                data.append(line.strip().split('\t'))
        return data

    print('[Info] Loading data...')
    _, num_types = num_types_loading(opt.data + 'historical_diag.txt')
    d_seqs, t_seqs = load_data(opt.data + 'train_d_seqs.txt'), load_data(opt.data + 'train_t_seqs.txt')
    assert len(d_seqs) == len(t_seqs)
    data_num = len(d_seqs)
    train_data = d_seqs[:int(data_num*0.8)], t_seqs[:int(data_num*0.8)]
    test_data = d_seqs[int(data_num*0.8):int(data_num*0.9)], t_seqs[int(data_num*0.8):int(data_num*0.9)]

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types+1


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_non_event_ll = 0
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        # loss = event_loss + pred_loss

        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += event_ll.sum().item()
        total_non_event_ll += non_event_ll.sum().item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        # break
    
    # print(f'total_event_ll: {total_event_ll};')
    # print(f'total_num_event: {total_num_event}; total_num_pred: {total_num_pred}')

    rmse = np.sqrt(total_time_se / total_num_pred)
    
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse
    # return total_event_ll / total_num_event, \
    #         total_event_rate / total_num_pred, \
    #             total_non_event_ll / total_num_event


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse
    # return total_event_ll / total_num_event, total_event_rate / total_num_pred, 0


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))


        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
            'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

    
        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
            'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
            .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))
        
        if epoch % 10 == 0:
            # model params
            params = {
                'state_dict': model.state_dict(), 
                'num_types': model.num_types,
                'd_model': model.d_model,
                'd_rnn': model.d_rnn,
                'd_inner': model.d_inner,
                'n_layers': model.n_layers,
                'n_head': model.n_head,
                'd_k': model.d_k,
                'd_v': model.d_v,
                'dropout': model.dropout}
            torch.save(params, opt.params + f'transformer_hawkes_epoch_{epoch}.pth')

        scheduler.step()


#set randome seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #fix hash seed
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed = 1
seed_torch(seed)

def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default='./example_data/')

    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=256)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=256)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=8)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.)

    parser.add_argument('-log', type=str, default='new_log.txt')
    parser.add_argument('-params', type=str, default='./new_params/')
    
    parser.add_argument('-pre_trained_model', type=str, default='')    

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda:1')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    # setup params folder
    if os.path.exists(opt.params) is False:
        os.makedirs(opt.params)

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    if opt.pre_trained_model != '':
        params = torch.load(opt.pre_trained_model)
        model = Transformer(
            num_types=params['num_types'],
            d_model=params['d_model'],
            d_rnn=params['d_rnn'],
            d_inner=params['d_inner'],
            n_layers=params['n_layers'],
            n_head=params['n_head'],
            d_k=params['d_k'],
            d_v=params['d_v'],
            dropout=params['dropout'],
        )
        model.load_state_dict(params['state_dict'])
        print('[Info] Successfully loading model')
    else:
        model = Transformer(
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
        )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
