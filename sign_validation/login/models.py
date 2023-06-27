"""
/*
* models class
* this class is for create your models
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
from django.db import models

# Create your models here.

# 模型即存入数据库部分
# 用户模型包含用户名、密码、邮件和是否已经注册过
class User(models.Model):

    name = models.CharField(max_length=128, unique=True)
    password = models.CharField(max_length=256)
    email = models.EmailField(unique=True)
    has_confirmed = models.BooleanField(default=False)

    def __str__(self):

        return self.name

    class Meta:
        verbose_name = "用户"
        verbose_name_plural = "用户"


class ConfirmString(models.Model):
    code = models.CharField(max_length=256)
    user = models.OneToOneField('User', on_delete=models.CASCADE)

    def __str__(self):
        return self.user.name + ":   " + self.code

    class Meta:

        verbose_name = "确认码"
        verbose_name_plural = "确认码"

# 图片模型，包含标题和图片文件
class Images(models.Model):
    # Title = models.CharField(max_length=16, unique=True)
    Image = models.ImageField(upload_to='images', null=True)

    # def __str__(self):
    #     return self.Image

class ImagesResult(models.Model):
    user_name = models.CharField(max_length=16, null=True)
    Images = models.ImageField(upload_to='imagesResult', null=True)
    ImagesRef = models.ImageField(upload_to='imagesResult', null=True)
    result = models.CharField(max_length=16,null=True)
    # def __str__(self):
    #
    #     return self.Images


# 两张图片模型，包含测试图片标题和测试图片文件及参考图片标题和参考图片文件
class DoubleImages(models.Model):
    # TestImg_title = models.CharField(max_length=16, unique=True)
    TestImg_image = models.ImageField(upload_to='doubleimage/testimg_image', null=True)
    # ReferenceImg_title = models.CharField(max_length=16, unique=True)
    ReferenceImg_image = models.ImageField(upload_to='doubleimage/referenceimg_image', null=True)

    # def __str__(self):
    #     return self.TestImg_image + self.ReferenceImg_image

class DoubleImagesResult(models.Model):
    user_name = models.CharField(max_length=16,null=True)
    TestImg_image = models.ImageField(upload_to='doubleimageResult/testimg_image', null=True)
    ReferenceImg_image = models.ImageField(upload_to='doubleimageResult/referenceimg_image', null=True)
    result = models.CharField(max_length=16, null=True)
    score_info = models.CharField(max_length=16,null=True)




