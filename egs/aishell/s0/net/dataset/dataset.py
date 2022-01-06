
import codecs
import json
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import dataset.features as features


def read_symbol_table(unit_file):
    symbol_table = {}
    with open(unit_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


class AudioData(Dataset):
    def __init__(self,
                 symbol_table,
                 data_file,
                 batch_conf, sort):

        #assert batch_type in ['static', 'dynamic']
        data = []
        with codecs.open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_json = line.strip()
                obj = json.loads(line_json)
                assert 'key' in obj
                assert 'wav' in obj
                assert 'txt' in obj

                key = obj['key']
                wav_file = obj['wav']
                txt = obj['txt']
                dur = 0
                if 'dur' in obj:
                    dur = obj['dur']

                tokens = []
                for ch in txt:
                    if ch == ' ':
                        ch = '_'
                    if ch in symbol_table:
                        tokens.append(symbol_table[ch])
                    elif '<unk>' in symbol_table:
                        tokens.append(symbol_table['<unk>'])
                #print(wav_file)
                #waveform, sample_rate = torchaudio.load(wav_file)
                #duration = waveform.size(1)
                data.append((key, wav_file, tokens, dur))

        if sort:
            data = sorted(data, key=lambda x: x[3])

        self.minibatch = []
        num = len(data)
        assert 'batch_size' in batch_conf
        batch_size = batch_conf['batch_size']

        cur = 0
        while cur < num:
            end = min(cur + batch_size, num)
            item = []

            for i in range(cur, end):
                item.append(data[i])

            self.minibatch.append(item)
            cur = end

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, item):
        return self.minibatch[item]


class CollateFunc(object):
    def __init__(self,
                 filter_conf,
                 resample_conf,
                 speed_perturb,
                 fbank_conf,
                 spec_aug,
                 spec_aug_conf,
                 shuffle,
                 shuffle_conf,
                 sort,
                 sort_conf
                 ):

        self.filter_conf = filter_conf
        self.resample_conf = resample_conf
        self.speed_perturb = speed_perturb
        self.fbank_conf = fbank_conf
        self.spec_aug = spec_aug
        self.spec_agu_conf = spec_aug_conf
        self.shuffle = shuffle
        self.shuffle_conf = shuffle_conf
        self.sort = sort
        self.sort_conf = sort_conf

    def __call__(self, batch):
        assert len(batch) == 1

        keys, xs, ys = features.extract(batch[0], self.fbank_conf)
        #padding
        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))
        if len(xs) > 0:
            xs_pad = pad_sequence([torch.from_numpy(x).float() for x in xs]
                                  , True, 0)
        else:
            xs_pad = torch.Tensor(xs)

        if ys is None:
            ys_pad = None
            ys_lengths = None
        else:
            ys_lengths = torch.from_numpy(
                np.array([y.shape[0] for y in ys], dtype=np.int32))
            if len(ys) > 0:
                ys_pad = pad_sequence([torch.from_numpy(y).int() for y in ys], True, -1)

        return keys, xs_pad, xs_lengths, ys_pad, ys_lengths


class AudioDataset(object):
    def __init__(self,
                 distributed,
                 symbol_table,
                 data_file,
                 batch_conf,
                 filter_conf,
                 resample_conf,
                 speed_perturb,
                 fbank_conf,
                 spec_aug,
                 spec_aug_conf,
                 shuffle,
                 shuffle_conf,
                 sort,
                 sort_conf
                 ):
        self.distributed = distributed
        self.dataset = AudioData(symbol_table, data_file, batch_conf, sort)
        self.collate = CollateFunc(filter_conf,
                 resample_conf,
                 speed_perturb,
                 fbank_conf,
                 spec_aug,
                 spec_aug_conf,
                 shuffle,
                 shuffle_conf,
                 sort,
                 sort_conf)
        if distributed:
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=True
            )
        else:
            self.sampler = None

    def get_loader(self,
                  pin_memory,
                  num_workers):

        return DataLoader(self.dataset,
                          collate_fn=self.collate,
                          sampler=self.sampler,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          batch_size=1)

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)





