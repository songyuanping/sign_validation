"""
/*
* admin class
* this class is for register models
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
from django.contrib import admin
from . import models
# Register your models here.

# 后台管理用户类
admin.site.register(models.User)
# 后台管理单张图片鉴别类
admin.site.register(models.Images)
# 后台管理两张签名判断类
admin.site.register(models.DoubleImages)
# 后台管理注册验证码
admin.site.register(models.ConfirmString)