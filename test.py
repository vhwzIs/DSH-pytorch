import os
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import *
from utils import *


def hashing_loss(b, cls, m, alpha):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1)
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=2).view(-1)
    loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)

    loss = loss.mean() + alpha * (b.abs() - 1).abs().sum(dim=1).mean() * 2

    return loss


@torch.no_grad()
def test(epoch, dataloader, net, m, alpha):
    accum_loss = 0
    net.eval()
    for img, cls in dataloader:
        img, cls = [Variable(x.cuda(), volatile=True) for x in (img, cls)]

        b = net(img)
        loss = hashing_loss(b, cls, m, alpha)
        accum_loss += loss.item()

    accum_loss /= len(dataloader)
    print(f'[{epoch}] val loss: {accum_loss:.4f}')
    return accum_loss


def main():
    parser = argparse.ArgumentParser(description='train DSH')
    parser.add_argument('--cifar', default='../dataset/cifar', help='path to cifar')
    parser.add_argument('--weights', default='', help="path to weight (to continue training)")
    parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpointing after batches')

    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=0, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=12, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.01, help='weighting of regularizer')

    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.outf, exist_ok=True)
    choose_gpu(opt.ngpu)
    feed_random_seed()
    train_loader, test_loader = init_cifar_dataloader(opt.cifar, opt.batchSize)
    logger = SummaryWriter()

    # setup net
    net = DSH(opt.binary_bits)
    print(net)
    if opt.weights:
        print(f'loading weight form {opt.weights}')
        net.load_state_dict(torch.load(opt.weights, map_location=lambda storage, location: storage))

    net.cuda()

    # compute mAP by searching testset images from trainset
    trn_binary, trn_label = compute_result(train_loader, net)
    tst_binary, tst_label = compute_result(test_loader, net)
    mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label)
    print(f'retrieval mAP: {mAP:.4f}')



if __name__ == '__main__':
    main()
