import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image


class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name2 = os.listdir(os.path.join(path, "1st_manual"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name1)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        # 拿到的图片和标签
        name1 = self.name1[index]
        name2 = self.name2[index]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]
        # 读取原始图片和标签，并转RGB
        img_o = cv2.imread(os.path.join(img_path[0], name1))
        _, img_l = cv2.VideoCapture(os.path.join(img_path[1], name2)).read()
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        img_o = self.__trans__(img_o, 256)
        img_l = self.__trans__(img_l, 256)

        return self.trans(img_o), self.trans(img_l)

if __name__ == '__main__':
    i = 1
    dataset = Datasets(r"D:\DRIVE\training")
    for a, b in dataset:
        print(i)
        # print(a.shape)
        # print(b.shape)
        save_image(a, f"./img/{i}.jpg", nrow=1)
        save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 5:
            break
