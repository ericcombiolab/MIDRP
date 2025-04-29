import os
import torch
import numpy as np

from .preprocess.Dataset import get_dataloader
from .transformer.Models import get_non_pad_mask


def med_data_prepare(model, data, device):
    device = torch.device(device)
    model.to(device)
    model.eval()
    """ export med output """
    loader = get_dataloader(data, 1, shuffle=False)
    med_outputs = None
    d_seqs = None
    t_seqs = None

    for batch_i, batch in enumerate(loader):
        model.eval()
        with torch.no_grad():
            event_time, event_type = map(lambda x: x.to(device), batch)
            output, _ = model(event_type, event_time)
        if batch_i == 0:
            med_outputs = output
            d_seqs = event_type
            t_seqs = event_time
        else:
            med_outputs = torch.cat([med_outputs, output], dim=0)
            d_seqs = torch.cat([d_seqs, event_type], dim=0)
            t_seqs = torch.cat([t_seqs, event_time], dim=0)

    med_outputs = med_outputs.cpu().numpy()
    d_seqs = d_seqs.cpu()
    t_seqs = t_seqs.cpu()
    
    return med_outputs, d_seqs, t_seqs


def history_diag_output(outputs, d_seqs, types):
    non_pad_mask = get_non_pad_mask(d_seqs).squeeze(2)
    mask = non_pad_mask
    for type in types:
        type_mask = (d_seqs != type + 1)
        mask = mask * type_mask
    p_outputs = []
    for k in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if j == 0 and mask[k, j] == 0:
                p_outputs.append(np.zeros_like(outputs[k, j]))
                break
            else:
                if j+1 >= mask.shape[1] or mask[k, j+1] == 0:
                    p_outputs.append(outputs[k, j])
                    break
    return np.array(p_outputs)

    
def search_disease_index(disease_icds):
    icd10_npy = './icd10/icd10_list.npy'
    icd10_npy = np.load(icd10_npy)

    icd2idx = []
    for dis in disease_icds:
        wh = np.where(icd10_npy == dis)[0]
        if wh.shape[0] == 0:
            wh = -999
        else:
            wh = wh.item()
        icd2idx.append(wh)
    return icd2idx
        