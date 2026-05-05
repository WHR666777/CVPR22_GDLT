import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='fisv', choices=['rg', 'fisv'])
parser.add_argument('--video-path', type=str, default='../action_assessment/rg_feat/swintx_avg_fps25_clip32')
parser.add_argument('--clip-num', type=int, default=None)

parser.add_argument('--train-label-path', type=str, default='../action_assessment/rg_feat/train.txt')
parser.add_argument('--test-label-path', type=str, default='../action_assessment/rg_feat/test.txt')

parser.add_argument('--action-type', type=str, default='all')
parser.add_argument('--score-key', type=str, default='TES')
parser.add_argument('--score-max', type=float, default=None)

parser.add_argument('--model-name', type=str, default='action_net', help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")
parser.add_argument('--device', type=str, default='auto', help='auto/cuda/cpu')
parser.add_argument('--num-workers', type=int, default=0)

parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--optim', type=str, default='sgd')

parser.add_argument("--lr-decay", type=str, default='cos', help='use what decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.01, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")

parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--n_encoder', type=int, default=1)
parser.add_argument('--n_decoder', type=int, default=2)
parser.add_argument('--n_query', type=int, default=4)

parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--margin', type=float, default=1.0)

parser.add_argument('--dropout', type=float, default=None)
