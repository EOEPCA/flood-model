import torch


def computeIOU(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()

    no_ignore = target.ne(255)
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    intersection = torch.sum(output * target)
    union = torch.sum(target) + torch.sum(output) - intersection
    iou = (intersection + 0.0000001) / (union + 0.0000001)

    if iou != iou:
        print("failed, replacing with 0")
        iou = torch.tensor(0).float()

    return iou


def computeAccuracy(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()

    no_ignore = target.ne(255)
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output.eq(target))

    return correct.float() / len(target)
