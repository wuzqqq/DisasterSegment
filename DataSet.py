import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch


class Data(Dataset):
    def __init__(self, data_path, scale=(320, 320), mode='train'):
        super().__init__()
        self.mode = mode
        self.image_path = data_path + '/images'
        self.target_path = data_path + '/targets'
        self.images_list, self.targets_list = self.read_pic(self.image_path)
        self.resize = scale

    def __getitem__(self, index):
        image1 = Image.open(self.images_list[index][0])
        image1 = image1.resize(self.resize)
        image1 = np.array(image1)
        image2 = Image.open(self.images_list[index][1])
        image2 = image2.resize(self.resize)
        image2 = np.array(image2)

        image2 = torch.from_numpy(image2.transpose((2, 0, 1)) / 255.0).float()
        image1 = torch.from_numpy(image1.transpose((2, 0, 1)) / 255.0).float()

        if self.mode != 'test':
            target_ori = Image.open(self.targets_list[index])
            target_ori = target_ori.resize(self.resize)
            target_ori = np.array(target_ori)
            target = torch.from_numpy(target_ori).float().unsqueeze(dim=0)
            return image1, image2, target
        else:
            return image1, image2

    def __len__(self):
        return len(self.targets_list)

    def read_pic(self, image_path: str):

        images_list = []
        targets_list = []

        fold = os.listdir(image_path)

        for item in fold:
            name = item.split('_')
            if name[-2] == 'pre':
                name1 = item
                name2 = item.replace('pre', 'post')
                images_list.append([image_path+'/'+name1, image_path+'/'+name2])
                targets_list.append(
                    image_path.replace('images', 'targets')+'/'+name2.replace('disaster', 'disaster_target')
                )

        return images_list, targets_list


if __name__ == '__main__':
    # image_path: str = 'D:/DisasterSegment/train'
    # dataset_train = Data(image_path, scale=(1024, 1024), mode='train')
    # dataloader_train = DataLoader(
    #     dataset_train,
    #     batch_size=1,
    #     shuffle=False
    # )
    # for image1, image2, target in dataloader_train:
    #     # print(target.size())
    #     # print(image1.squeeze(dim=0).numpy().transpose(1, 2, 0).shape)
    #     # img = Image.open('D:/DisasterSegment/train/images/guatemala-volcano_00000000_post_disaster.png')
    #     # print(np.array(img).shape)
    #     print(np.unique(target.squeeze(dim=0).numpy().transpose(1, 2, 0)))
    #     # img = Image.fromarray((image2.squeeze(dim=0).numpy().transpose(1, 2, 0)*255).astype('uint8'))
    #     # img.show()
    #     # break
    image_path: str = 'D:/DisasterSegment/train/targets/guatemala-volcano_00000000_post_disaster_target.png'
    image = Image.fromarray(np.array(Image.open(image_path))*63)
    image.show()
    # disasters = set()
    # fold = os.listdir(image_path)
    # for item in fold:
    #     disasters.add(item.split('_')[0].split('-')[1])
    # print(disasters)
