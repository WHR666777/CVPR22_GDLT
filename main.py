import torch
import numpy as np
import options
from datasets import AQADataset
from torch.utils.data import DataLoader
from models import model, loss
import os

import train
from test import test_epoch

try:
    from tensorboardX import SummaryWriter
except ImportError as exc:
    raise ImportError(
        "tensorboardX is required. Install it with 'pip install tensorboardX' in your conda environment."
    ) from exc


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested via --device cuda, but no CUDA device is available.")
        return torch.device('cuda')
    if device_arg == 'cpu':
        return torch.device('cpu')
    raise ValueError("Unsupported device '{}'. Use auto/cuda/cpu.".format(device_arg))


def ensure_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError("{} not found: {}".format(description, path))


def apply_paper_defaults(args):
    dataset = args.dataset.lower()

    if dataset == 'fisv':
        if args.clip_num is None:
            args.clip_num = 124
        if args.dropout is None:
            args.dropout = 0.7
        if args.score_key.upper() == 'PCS':
            if args.epoch is None:
                args.epoch = 400
        else:
            if args.epoch is None:
                args.epoch = 320
        # In the paper, lambda in Eq. (11) is 0.5 for Fis-V.
        if args.alpha is None:
            args.alpha = 0.5
    elif dataset == 'rg':
        if args.clip_num is None:
            args.clip_num = 68
        if args.dropout is None:
            args.dropout = 0.3
        if args.epoch is None:
            epoch_map = {
                'Ball': 250,
                'Clubs': 400,
                'Hoop': 500,
                'Ribbon': 150,
            }
            if args.action_type not in epoch_map:
                raise ValueError(
                    "For RG, action_type must be one of {} to match the paper defaults.".format(
                        list(epoch_map.keys())
                    )
                )
            args.epoch = epoch_map[args.action_type]
        # In the paper, lambda in Eq. (11) is 1.0 for RG.
        if args.alpha is None:
            args.alpha = 1.0
    else:
        raise ValueError("Unsupported dataset '{}'".format(args.dataset))

    return args


if __name__ == '__main__':
    args = options.parser.parse_args()
    args = apply_paper_defaults(args)
    setup_seed(0)
    device = resolve_device(args.device)

    ensure_path(args.video_path, "Feature directory")
    ensure_path(args.train_label_path, "Training label file")
    ensure_path(args.test_label_path, "Test label file")

    '''
    1. load data
    '''
    '''
    train data
    '''
    train_data = AQADataset(
        args.video_path,
        args.train_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        score_key=args.score_key,
        score_max=args.score_max,
        dataset=args.dataset,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    print(len(train_data))

    '''
    test data
    '''
    test_data = AQADataset(
        args.video_path,
        args.test_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        score_key=args.score_key,
        score_max=train_data.score_max,
        dataset=args.dataset,
        train=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    print("Resolved score_max: {:.4f}".format(train_data.score_max))
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    model = model.GDLT(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                       args.n_decoder, args.n_query, args.dropout).to(device)
    loss_fn = loss.LossFun(args.alpha, args.margin)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        ckpt_path = os.path.join('./ckpt', args.ckpt + '.pkl')
        ensure_path(ckpt_path, "Checkpoint")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
    print('=============Load model successfully=============')

    print(args)

    '''
    test mode
    '''
    if args.test:
        test_loss, coef = test_epoch(0, model, test_loader, None, device, args)
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}'.format(test_loss, coef))
        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_epoch = -1, -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    '''
    4. train
    '''
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print('=============Begin training=============')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    for epc in range(args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()
        # print(optim.state_dict()['param_groups'][0]['lr'])
        avg_loss, train_coef = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()
        test_loss, test_coef = test_epoch(epc, model, test_loader, logger, device, args)
        if test_coef > best_coef:
            best_coef, best_epoch = test_coef, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best.pkl')

        print('Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.3f}'
              .format(epc, avg_loss, train_coef, test_loss, test_coef))
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef = \
                avg_loss, train_coef, test_loss, test_coef
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Eopch: {}'.format(best_coef, best_epoch))
