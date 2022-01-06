
import argparse
import os
import yaml
import logging
import torch
import torch.optim as optim

from dataset.dataset import AudioDataset
from dataset.dataset import read_symbol_table
from executor import Executor
from scheduler import LRScheduler

from wenet.transformer.asr_model import init_asr_model

from ctcaed.model import create_model


def get_args():
    parser = argparse.ArgumentParser(description='Do Training...')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--dev_data', required=True, help= 'cross validate data file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory buffers used for reading')
    parser.add_argument('--num_workers', type=int, default=0, help='num of subprocess workers for processing data')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table', required=True, help='model unit symbol')
    parser.add_argument('--checkpoint', default=None, help='checkpoint model')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    distributed = args.world_size > 1
    torch.manual_seed(7777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table = read_symbol_table(args.symbol_table)

    dataset_conf = configs.get('dataset_conf', {})

    print(dataset_conf)
    train_dataset = AudioDataset(distributed, symbol_table, args.train_data,  **dataset_conf)

    train_data_loader = train_dataset.get_loader(args.pin_memory, args.num_workers)

    input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

    #raw_wav = configs['raw_wav']
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True

    #model = init_asr_model(configs)
    model = create_model(configs)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    if args.rank == 0:
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(args.model_dir, 'init.zip'))

    start_epoch = 0
    cv_loss = 0.0
    step = -1

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir

    if distributed:
        assert (torch.cuda.is_available())
        device = torch.device('cuda' if use_gpu else 'cpu')
        pass
    else:
        use_gpu = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_gpu else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = LRScheduler(optimizer, **configs['scheduler_conf'])
    executor = Executor()
    configs['rank'] = args.rank

    for epoch in range(start_epoch, num_epochs):
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device, configs)


if __name__ == '__main__':
    main()

