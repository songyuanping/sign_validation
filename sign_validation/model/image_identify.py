"""
/*
* image_identify class
* this class is for identify image is genuine
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
import itertools
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


# 设定随机种子
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(GLOBAL_SEED)
GLOBAL_WORKER_ID = None


# 用于生成传入图片存入数据库后的名字
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


# 用 图片的鉴别进程
class IdentifyPreprocess(object):
    def __init__(self, identify_image_name):
        super(IdentifyPreprocess, self).__init__()
        # 签名图片二值化后的存储路径
        self.tmp_path = Config.temp_path
        # 输入神经网络的图片宽高
        self.img_w = Config.img_w
        self.img_h = Config.img_h
        # 对数据进行分层抽样
        self.pre_process_images(identify_image_name)
        # 将待鉴别的图片与数据库中原有的真实签名两两配对
        self.orig_pairs = self.make_pair_images(identify_image_name)

    def make_pair_images(self, identify_image_name):
        tmp_path = os.path.join('.', 'model', self.tmp_path)
        # print('tmp_path:', tmp_path)
        signer_list = sorted(os.listdir(tmp_path))
        # print('signer_list:', signer_list)
        orig_list = []
        orig_pairs = []
        for signer in signer_list:
            # 得到每个真签名下的图片名称
            orig_img_list = sorted(os.listdir(os.path.join(tmp_path, signer, 'orig')))
            for index, item in enumerate(orig_img_list):
                orig_img_list[index] = os.path.join(tmp_path, signer, "orig", item)
            orig_list.extend(orig_img_list)
        # print('orig_list:', orig_list)
        orig_pairs.extend(list(itertools.product([identify_image_name], orig_list)))
        return orig_pairs

    def pre_process_images(self, identify_image_name):

        # 对图像进行灰度化处理Y = 0.299R + 0.587G + 0.114B
        image = cv2.imread(identify_image_name, cv2.IMREAD_GRAYSCALE)
        # 对图像进行中值滤波，去除图片中的噪声
        image = cv2.bilateralFilter(image, 0, 50, 10)
        size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(size, size))
        # 对白底黑字的签名图片进行腐蚀
        image = cv2.erode(image, kernel)
        image = cv2.erode(image, kernel)

        # print('orig_img1 shape:', image.shape)
        # 将图像二值化
        # image = self.otsu(image)
        # 将图像中的签名剪切出来
        image = self.crop_signature(image)
        # 将图片进行存储
        cv2.imwrite(identify_image_name, image)
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


# 用于将初始传入图片更改为鉴别所需要的数据类型
class IdentifyDataset(Dataset):
    # image_size传入为img_height,img_width的形式
    def __init__(self, orig_pairs, batch_size=1):
        super(IdentifyDataset, self).__init__()
        self.orig_pairs = orig_pairs
        # 输入神经网络的图片大小
        self.image_size = (Config.img_h, Config.img_w)
        # 每一次输入神经网络的图片大小
        self.batch_size = batch_size
        # 一张原始图片被处理成多少张图片
        self.train_ratio = Config.train_ratio
        self.val_ratio = Config.val_ratio
        self.img_height, self.img_width = Config.img_h, Config.img_w
        # 图片对和每一对图片的标签及长度
        self.pair_length = len(self.orig_pairs)
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
        orig_images = []
        pairs = [torch.zeros((batch_size, self.img_height, self.img_width, 1)) for i in range(2)]
        # 一批次图片的标签
        start_pos = index * self.batch_size
        for index, (identify_image_name, orig_image_name) in enumerate(self.orig_pairs, start_pos):
            # 灰度图模式加载一幅彩图，读入的shape为(img_h,img_w)
            img1_list = self.visualize_signature(self.orig_pairs[index][0])
            img2_list = self.visualize_signature(self.orig_pairs[index][1])

            # print('pair:', pair[0], pair[1], 'label:', label, 'batch_size:', batch_size)
            for img1, img2 in zip(img1_list, img2_list):
                img1 = torch.from_numpy(np.array(img1).astype(np.float))
                img2 = torch.from_numpy(np.array(img2).astype(np.float))
                # 还需要数据标准化
                img1, img2 = img1 / 255, img2 / 255
                # 增加通道维度
                img1, img2 = torch.unsqueeze(img1, -1), torch.unsqueeze(img2, -1)
                pairs[0][k, :, :, :] = img1.clone().detach()
                pairs[1][k, :, :, :] = img2.clone().detach()
                orig_images.append(self.orig_pairs[index][1])
                k += 1
                if k == batch_size:
                    pairs[0] = pairs[0].permute(0, 3, 1, 2)
                    pairs[1] = pairs[1].permute(0, 3, 1, 2)
                    # 函数会从上一次暂停的位置开始，一直执行到下一个yield 表达式，将yield 关键字后的表达式列表返回给调用者，
                    # 并再次暂停。注意，每次从暂停恢复时，生成器函数的内部变量、指令指针、内部求值栈等内容和暂停时完全一致。
                    return pairs, orig_images

    def visualize_signature(self, image_name):
        # print('image_name:', image_name)
        # 对图像进行灰度化处理Y = 0.299R + 0.587G + 0.114B
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        assert image is not None
        # print('image:', image)
        new_image = self.resize_image(image, image.shape[0], image.shape[1], background=255)
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

    def resize_image(self, image, height, width, background=255):
        """
        :param height: 原始图片的高度（以像素为单位）
        :param width: 原始图片的宽度（以像素为单位）
        :param image: 需要进行大小修改的灰度图片
        :param background: 灰度图片的背景，可以取值为255 或者 0
        :return: 进行大小转换后的图片，如224x224
        """
        # 将签名图片缩放为神经网络输入图片的大小
        img_h, img_w = self.img_height, self.img_width
        scale = min(img_h / height, img_w / width)
        image_height, image_width = int(round(scale * height)), int(round(scale * width))
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


# 用于鉴别用户上传的签名图片的真伪
class IdentifyImage(object):
    def __init__(self, identify_image_name):
        super(IdentifyImage, self).__init__()
        # 完成图片的预处理、灰度化和图片大小格式化等工作
        self.identifyPreprocess = IdentifyPreprocess(identify_image_name)
        # 完成图片数据增强和批量化操作
        self.identifyDataset = IdentifyDataset(self.identifyPreprocess.orig_pairs, batch_size=64)

    # 将鉴别图片的操作流程封装为一个函数接口，供其他模块调用
    def identify_image(self):
        # 存放模型的预测结果
        predict_list = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 生成模型并加载训练好的模型权重
        idn = Net3()
        idn.load_state_dict(torch.load(os.path.join('.', 'model', 'idn.pth')))
        # 将模型指定为推理模式
        idn.eval()
        forward_idn = ForwardIDN()
        forward_idn.eval()
        # 可以调节num_works进行多线程加载数据
        identify_generator = DataLoader(self.identifyDataset, shuffle=True, num_workers=4,
                                        worker_init_fn=worker_init_fn)
        # 存放推理结果中的图片路径
        orig_image_list = []
        with torch.no_grad():
            # 对数据进行推理操作
            for index, (pair, orig_images) in enumerate(identify_generator):
                pair[0], pair[1] = torch.squeeze(pair[0], dim=0), torch.squeeze(pair[1], dim=0)
                orig_image_list.extend(orig_images)
                predicts = forward_idn(pair[0].to(device), pair[1].to(device), None, idn.to(device), train=False)
                predict_list = torch.cat((torch.tensor(predict_list), predicts.cpu()), dim=0)

        # print('predict_list len:', len(predict_list), 'orig_image_list len:', len(orig_image_list))
        predict_list = [float(i) for i in predict_list]
        for index in range(0, len(predict_list) - 1):
            min = predict_list[index]
            min_pos = index
            for j in range(index + 1, len(predict_list)):
                if predict_list[j] < min:
                    min = predict_list[j]
                    min_pos = j
            if min_pos != index:
                temp = predict_list[index]
                predict_list[index] = predict_list[min_pos]
                predict_list[min_pos] = temp

                temp = orig_image_list[index]
                orig_image_list[index] = orig_image_list[min_pos]
                orig_image_list[min_pos] = temp

        # 计算相似度
        score_list = predict_list[:5]
        score_list = [np.exp(-score_list[i]) for i in range(len(score_list))]
        return score_list, orig_image_list[:5]
