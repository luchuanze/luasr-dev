

#from contextlib import nullcontext
# if your python version < 3.7 use the below one
import logging
from contextlib import suppress as nullcontext

import torch
from torch.nn.utils import clip_grad_norm_


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, configs):

        model.train()
        grad_clip = configs.get('grad_clip', 50.0)
        log_interval = configs.get('log_interval', 10)
        epoch = configs.get('epoch', 0)
        accum_grad = configs.get('accum_grad', 1)

        model_context = nullcontext

        with model_context():
            for idx, batch in enumerate(data_loader):
                keys, feats, feats_len, targets, targets_len = batch
                feats = feats.to(device)
                feats_len = feats_len.to(device)
                targets = targets.to(device)
                targets_len = targets_len.to(device)
                num_utts = len(keys)
                if num_utts == 0:
                    continue

                print(keys)
                #
                # context = nullcontext
                #
                # with context():
                #     loss, loss_att, loss_ctc = model(
                #         feats, feats_len, targets, targets_len
                #     )
                #
                #     loss = loss / accum_grad
                #
                #     loss.backward()
                #
                # if idx % accum_grad == 0:
                #     grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
                #     if torch.isfinite(grad_norm):
                #         optimizer.step()
                #
                #     optimizer.zero_grad()
                #     scheduler.step()
                #     self.step += 1
                #
                # if idx % log_interval == 0:
                #     lr = optimizer.param_groups[0]['lr']
                #     log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                #         epoch, idx,
                #         loss.item()
                #     )
                #
                #     log_str += 'lr {:.8f}'.format(lr)
                #
                #     print(log_str)
                #
                #     logging.debug(log_str)




