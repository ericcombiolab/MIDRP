import sys

import numpy as np
import torch
import torch.utils.data

from ..transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """

        self.time = [[float(elem) for elem in t_seq] for t_seq in data[1]]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[int(elem) + 1 for elem in d_seq] for d_seq in data[0]]

        assert len(self.time) == len(self.event_type)

        self.length = len(self.time)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = 9

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = 9

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, event_type = list(zip(*insts))
    time = pad_time(time)
    event_type = pad_type(event_type)
    return time, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=0,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
