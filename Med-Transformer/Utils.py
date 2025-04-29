import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask

from transformer import Constants


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask
    temp_time = diff_time.unsqueeze(-1).unsqueeze(-1) * \
                torch.rand([*diff_time.size(), model.num_types, num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(-1).unsqueeze(-1)

    temp_hid = model.linear(data).unsqueeze(-1)
    
    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)

    unbiased_integral = torch.sum(all_lambda, dim=-1) * diff_time.unsqueeze(-1) / num_samples

    unbiased_integral = torch.sum(unbiased_integral, dim=-1)

    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    data = data[:,:-1].contiguous()
    types = types[:,1:].contiguous()

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i).bool().to(data.device)


    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(all_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # move labels to correct device to enable model parallelism
    labels = types

    # print(labels)

    # Shift so that tokens < n predict n
    shift_logits = prediction[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    batch_size, seq_length, vocab_size = shift_logits.shape

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    loss = loss_fct(
        shift_logits.view(batch_size * seq_length, vocab_size),
        shift_labels.view(batch_size * seq_length)
    )

    pred_type = torch.max(shift_logits, dim=-1)[1]
    correct_num = torch.sum((pred_type == shift_labels) * shift_labels.ne(Constants.PAD).type(torch.float))

    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    pred = prediction.squeeze(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    pred = pred[:, :-1]

    # event time gap prediction
    diff = (pred - true) * (event_time[:, 1:].ne(Constants.PAD).type(torch.float))
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
