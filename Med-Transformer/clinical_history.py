import os
import sys
import argparse

import re
import numpy as np
import torch

from .transformer.Models import Transformer
from common import *

dir_path = os.getcwd()

# The icd-10 in input_file must exist in ./data/icd10_list.npy when using this function
def disease_icd_to_index(input_file, save_file):
    icd10_npy = './icd10/icd10_list.npy'
    icd10_npy = np.load(icd10_npy)

    d_seqs = []
    with open(input_file, 'r') as fp:
        for line in fp:
            line = re.split(r'\s+', line.strip())
            index_list = []
            for icd in line:
                if not(icd in icd10_npy):
                    continue
                icd_index = np.where(icd10_npy == icd)[0][0]
                index_list.append(icd_index)
            d_seqs.append(index_list)
    
    with open(save_file, 'w') as fp:
        for line in d_seqs:
            for k, d in enumerate(line):
                if k == (len(line) - 1):
                    fp.write(str(d) + '\n')
                else:
                    fp.write(str(d) + '\t')


def load_data(path_name):
    data = []
    with open(path_name, 'r') as fp:
        for line in fp:
            data.append(line.strip().split('\t'))
    return data


def med_transfer_encoding(disease_icds:list, d_seqs_path:str, t_seqs_path:str, save_path:str):
    device = 'cpu'

    # Model Loading
    print('[Info] Med-Transformer Modle Loading...')
    transformer_hakes_param = './params/med-transformer.pth'
    params = torch.load(transformer_hakes_param, map_location=torch.device('cpu'))
    med_transformer_model = Transformer(
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
    med_transformer_model.load_state_dict(params['state_dict'])

    print('[Info] Disease and Time sequences Loading...')
    d_seqs = load_data(d_seqs_path)
    t_seqs = load_data(t_seqs_path)

    # Model hidden encoding
    types = search_disease_index(disease_icds)

    """ prepare output data which contains the information of the medical history """
    print('[Info] Med-Transformer Modle Hidden Encoding Running...')
    med_output, d_seqs, t_seqs = med_data_prepare(med_transformer_model, (d_seqs, t_seqs), device)
    med_output = history_diag_output(med_output, d_seqs, types)
    np.save(save_path, med_output)


def __main__():
    parser = argparse.ArgumentParser(description='Med-Transformer')

    parser.add_argument('--disease_icds', type=str, required=True, help='Disease ICDs for a target disease')
    parser.add_argument('--d_seqs_path', type=str, required=True, help='Path to disease sequences, e.g., ./data/d_seqs.txt')
    parser.add_argument('--t_seqs_path', type=str, required=True, help='Path to time sequences, e.g., ./data/t_seqs.txt')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output')

    args = parser.parse_args()
    disease_icds = args.disease_icds.split(',')
    d_seqs_path = args.d_seqs_path
    t_seqs_path = args.t_seqs_path
    save_path = args.save_path

    med_transfer_encoding(disease_icds, d_seqs_path, t_seqs_path, save_path)