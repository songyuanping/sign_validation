"""
/*
* image_judge class
* this class is for judge whether two signatures are written by the same person
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
import os
import random

import cv2
import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from .config import Config
from .idn import Net3, ForwardIDN

np.set_printoptions(threshold=np.inf)

GLOBAL_SEED = 1


# 设置随机种子
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(GLOBAL_SEED)
GLOBAL_WORKER_ID = None


# 设置传入签名的存储名字
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


#  用于判断两张签名是否为同一人所写的判断流程
class JudgePreprocess(object):
    def __init__(self, test_image_name, refer_image_name):
        super(JudgePreprocess, self).__init__()
        # 输入神经网络的图片宽高
        self.img_w = Config.img_w
        self.img_h = Config.img_h
        # 对数据进行分层抽样
        self.pre_process_images(test_image_name, refer_image_name)
        # 将待鉴别的图片配对
        self.judge_pairs = [(test_image_name, refer_image_name)]

    def pre_process_images(self, test_image_name, refer_image_name):

        # 对图像进行灰度化处理Y = 0.299R + 0.587G + 0.114B
        test_image = cv2.imread(test_image_name, cv2.IMREAD_GRAYSCALE)
        refer_image = cv2.imread(refer_image_name, cv2.IMREAD_GRAYSCALE)
        # 对图像进行中值滤波，去除图片中的噪声
        test_image = cv2.bilateralFilter(test_image, 0, 50, 10)
        refer_image = cv2.bilateralFilter(refer_image, 0, 50, 10)
        size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(size, size))
        # 对白底黑字的签名图片进行腐蚀
        test_image = cv2.erode(test_image, kernel)
        test_image = cv2.erode(test_image, kernel)

        refer_image = cv2.erode(refer_image, kernel)
        refer_image = cv2.erode(refer_image, kernel)

        # print('orig_img1 shape:', image.shape)

        # 将图像中的签名剪切出来
        test_image = self.crop_signature(test_image)
        refer_image = self.crop_signature(refer_image)
        # 将图片进行存储
        cv2.imwrite(test_image_name, test_image)
        cv2.imwrite(refer_image_name, refer_image)
        # print('identify_image_name:',identify_image_name)

    def crop_signature(self, image, background=255):
        """
        :param image: 含有签名的灰度图片
        :param background: 签名图片的背景灰度值
        :return: image1: 固定大小的带有空洞的图片,image2:固定大小的原始签名图片
        """
        # 从签名图片中找到签名的左上角坐标和右下角坐标
        img_height, img_width = image.shape
        min_h, min_w = 0, 0
        max_h, max_w = img_height, img_width

        # 将图片中的签名部分剪切出来
        for i in range(img_height):
            # 如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true。
            if any(image[i] <= 255 - background + 200):
                min_h = i
                break
        for i in range(img_height - 1, -1, -1):
            if any(image[i] <= 255 - background + 200):
                max_h = i
                break
        for j in range(img_width):
            if any(image[:, j] <= 255 - background + 200):
                min_w = j
                break
        for j in range(img_width - 1, -1, -1):
            if any(image[:, j] <= 255 - background + 200):
                max_w = j
                break
        new_image = image
        if min_h < max_h and min_w < max_w:
            new_image = image[min_h:max_h + 1, min_w:max_w + 1]
        return new_image


# 用于将传入的签名图片更改为鉴别所需要的数据类型
class JudgeDataset(Dataset):
    # image_size传入为img_height,img_width的形式
    def __init__(self, judge_pairs, batch_size=1):
        super(JudgeDataset, self).__init__()
        self.judge_pairs = judge_pairs
        # 输入神经网络的图片大小
        self.image_size = (Config.img_h, Config.img_w)
        # 每一次输入神经网络的图片大小
        self.batch_size = batch_size
        # 一张原始图片被处理成多少张图片
        self.val_ratio = Config.val_ratio
        self.img_height, self.img_width = Config.img_h, Config.img_w
        # 图片对和每一对图片的标签及长度
        self.pair_length = len(self.judge_pairs)
        # 计算数据的批次数量
        self.batch_length = self.pair_length // self.batch_size
        if self.pair_length != self.batch_size * self.batch_length:
            self.batch_length += 1

    def __len__(self):
        return self.batch_length

    # 不能在该函数体内定义对于self.all_pairs，self.all_labels的操作，当多线程时会引发错误
    def __getitem__(self, index):
        label_length = self.pair_length
        length = self.batch_length

        # 处于测试阶段时每张图片变成黑白对应的1对图片
        batch_size = self.batch_size * self.val_ratio
        if index == length - 1:
            batch_size = (label_length - index * self.batch_size) * self.val_ratio
        # 记录已经放到pairs中的图片数目
        k = 0
        pairs = [torch.zeros((batch_size, self.img_height, self.img_width, 1)) for i in range(2)]
        # 一批次图片的标签
        start_pos = index * self.batch_size
        for index, (test_image_name, refer_image_name) in enumerate(self.judge_pairs, start_pos):

            # 灰度图模式加载一幅彩图，读入的shape为(img_h,img_w)
            img1_list = self.visualize_signature(self.judge_pairs[index][0])
            img2_list = self.visualize_signature(self.judge_pairs[index][1])
            # print('pair:', pair[0], pair[1], 'label:', label, 'batch_size:', batch_size)
            for img1, img2 in zip(img1_list, img2_list):
                img1 = torch.from_numpy(np.array(img1).astype(np.float))
                img2 = torch.from_numpy(np.array(img2).astype(np.float))
                # 还需要数据标准化
                img1, img2 = img1 / 255, img2 / 255
                img1, img2 = torch.unsqueeze(img1, -1), torch.unsqueeze(img2, -1)
                # print('img1 shape:', img1.shape, 'img2 shape:', img2.shape)
                # print('k=',k)
                pairs[0][k, :, :, :] = img1.clone().detach()
                pairs[1][k, :, :, :] = img2.clone().detach()
                # targets[k] = self.all_labels[iter_]
                k += 1
                if k == batch_size:
                    pairs[0] = pairs[0].permute(0, 3, 1, 2)
                    pairs[1] = pairs[1].permute(0, 3, 1, 2)
                    # print('pairs[0] shape:', pairs[0].shape, 'targets :', targets)
                    # print('pairs[1] shape:', pairs[1].shape, 'targets :', targets)
                    # print(orig_images)
                    # 函数会从上一次暂停的位置开始，一直执行到下一个yield 表达式，将yield 关键字后的表达式列表返回给调用者，
                    # 并再次暂停。注意，每次从暂停恢复时，生成器函数的内部变量、指令指针、内部求值栈等内容和暂停时完全一致。
                    return pairs

    def visualize_signature(self, image_name):
        # print('image_name:', image_name)

        # 对图像进行灰度化处理Y = 0.299R + 0.587G + 0.114B
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        # 生成大小为224x224的白底黑字的图片,以及减去图片一部分元素的图片
        new_image = self.crop_signature(image, background=255)
        # 保存图片数据增强后的结果
        image_list = []
        # 白底黑字的签名图片
        image_list.append(new_image)
        # 白底黑字的签名图片
        image_list.append(np.array(np.full_like(new_image, 255) - new_image, dtype=np.uint8))

        # for image1 in image_list:
        #     image1=np.array(image1)
        #     cv2.imshow('image1', image1)
        #     cv2.waitKey(0)
        return image_list

    def resize_image(self, image, background=255):
        """
        :param image: 需要进行大小修改的灰度图片
        :param background: 灰度图片的背景，可以取值为255 或者 0
        :return: 进行大小转换后的图片，如224x224
        """
        # 将签名图片缩放为神经网络输入图片的大小
        img_h, img_w = self.img_height, self.img_width
        # 将图片按照长宽比不变进行放大
        scale = min(img_h / image.shape[0], img_w / image.shape[1])
        image_height, image_width = int(round(scale * image.shape[0])), int(round(scale * image.shape[1]))
        # 通常的，缩小使用cv.INTER_AREA，放缩使用cv.INTER_CUBIC(较慢)
        # 和cv.INTER_LINEAR(较快效果也不错)。默认情况下，所有的放缩都使用cv.INTER_LINEAR。
        # 设置的参数为(image_width,image_height) image.shape=(image_height,image_width)
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        # print('after resize_image image shape:', image.shape)
        new_image = np.full((img_h, img_w), background)
        top_h, top_w = (img_h - image_height) // 2, (img_w - image_width) // 2
        new_image[top_h:top_h + image_height, top_w:top_w + image_width] = image
        new_image = np.array(new_image, dtype=np.uint8)
        return new_image
    def crop_signature(self, image, background=255):
        """
        :param image: 含有签名的灰度图片
        :param background: 签名图片的背景灰度值
        :return: image1: 固定大小的带有空洞的图片,image2:固定大小的原始签名图片
        """
        # 将图片的一部分变成背景色，将图片进行遮挡
        image1 = self.resize_image(image, background)
        return image1

# 用于对传入的两张签名图片进行判断，判断其是否为一人所写
class JudgeImage(object):
    def __init__(self, test_image_name, refer_image_name):
        super(JudgeImage, self).__init__()
        self.judgePreprocess = JudgePreprocess(test_image_name, refer_image_name)
        self.judgeDataset = JudgeDataset(self.judgePreprocess.judge_pairs, batch_size=1)
        # print('in IdentifyImage(object):',self.identifyDataset.orig_pairs)

    def judge_image(self):
        predict_list = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        idn = Net3()
        idn.load_state_dict(torch.load(os.path.join('.', 'model', 'idn.pth')))
        # idn.load_state_dict(torch.load(os.path.join('.', 'model', 'idn.pth'), map_location='cpu'))
        idn.eval()
        forward_idn = ForwardIDN()
        forward_idn.eval()

        # 可以调节num_works进行多线程加载数据
        judge_generator = DataLoader(self.judgeDataset, shuffle=True, worker_init_fn=worker_init_fn)
        with torch.no_grad():
            for index, pair in enumerate(judge_generator):
                pair[0], pair[1] = torch.squeeze(pair[0], dim=0), torch.squeeze(pair[1], dim=0)
                predicts = forward_idn(pair[0].to(device), pair[1].to(device), None, idn.to(device), train=False)
                predict_list = torch.cat((torch.tensor(predict_list), predicts.cpu()), dim=0)

        # print('predict_list len:', len(predict_list))
        predict_list = [float(i) for i in predict_list]
        # print('judge_image predict_list:', predict_list)

        return predict_list
