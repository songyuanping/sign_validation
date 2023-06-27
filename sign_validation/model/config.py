"""
/*
* config class
* this class is for config photos
* created by SuLinyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
# Config photos
class Config:
    # 存放图片经过二值化处理后的文件
    temp_path = 'tmp'
    img_h, img_w = 128, 128
    train_ratio = 6
    val_ratio = 2
    class_number = 128
    # 通过验证集得到的thresh
    thresh = 0.5792