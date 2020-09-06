import torch
import torch.nn as nn
import torch.optim as optim
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        end = time.time()
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)

            # Update meters
            losses.update(loss.item(), n=x.shape[0])
            top1.update(prec1.item(), n=x.shape[0])
            top5.update(prec5.item(), n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 1 == 0:
                print(f'\rN: {i*x.shape[0]} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Loss: {losses.val:.4f} ({losses.avg:.4f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    print('') #Endline

    return top1.avg

def coverage_size(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
