from DataSet import Data
import torch
from torch.utils.data import DataLoader
from DoubleUnet import DoubleUnet
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class args(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.01
    epoches = 7


class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=5, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / self.class_num
        return dice_loss


def main(args, model, data_loader):
    writer = SummaryWriter('./log/')
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = Multi_DiceLoss()
    model.to(args.device)
    for epoch in range(args.epoches):
        model.train()
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        loss_all = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr - 0.002
        for index, (image1, image2, target) in loop:
            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            target = target.to(args.device).long()
            optimizer.zero_grad()
            out1, out2 = model(image1, image2)
            loss1 = criterion(out1, target)
            loss2 = F.nll_loss(out2, target.squeeze(dim=1))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            loss_avg = loss_all / (index + 1)
            # print('step:', step, ', loss:', loss)
            writer.add_scalar('loss', loss.item(), epoch*2799+index+1)
            writer.add_scalar('average loss', loss_avg, epoch*2799+index+1)
            loop.set_description(f'Epoch [{epoch}/{args.epoches}]')
            loop.set_postfix(loss=loss_avg)
    torch.save(model.state_dict(), "./ModelStateDict/DoubleUnet.pth")
    writer.close()


if __name__ == '__main__':
    image_path: str = 'D:/DisasterSegment/train'
    dataset_train = Data(image_path, scale=(256, 256), mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True
    )
    args = args()
    model = DoubleUnet(3, 5)
    main(args, model, dataloader_train)
