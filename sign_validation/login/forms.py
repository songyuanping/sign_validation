"""
/*
* forms class
* this class is for creat froms
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
from django import forms
from captcha.fields import CaptchaField
from .models import Images, DoubleImages, ImagesResult, DoubleImagesResult

# 用户登录时的表单，包括用户名、密码和验证码
class UserForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': "Username",'autofocus': ''}))
    password = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control','placeholder': "Password"}))
    captcha = CaptchaField(label='验证码')

# 用户注册时的表单，包括用户名、密码1、密码2、邮件和验证码
class RegisterForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password1 = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    password2 = forms.CharField(label="确认密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label="邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    captcha = CaptchaField(label='验证码')

# 图片表单，模型为Image，包含字段为图片文件
class ImagesForm(forms.ModelForm):
    class Meta:
        model = Images
        # fields = ('Title', 'Image')
        fields = ('Image',)

# 图片结果表单，模型为ImagesResult,包含字段为测试图片，返回图片,及相似度
class ImagesResultForm(forms.ModelForm):
    class Meta:
        model = ImagesResult
        fields = ('Images', 'ImagesRef', 'result')

# 图片表单，模型为DoubleImages，包含字段为测试图片文件及参考图片文件
class DoubleImagesForm(forms.ModelForm):
    class Meta:
        model = DoubleImages
        # fields = ('TestImg_title', 'TestImg_image', 'ReferenceImg_title', 'ReferenceImg_image')
        fields = ('TestImg_image', 'ReferenceImg_image')

# 判断图片结果表单，模型为DoubleImagesResult,包含字段为测试图片，参考图片，及相似度
class DoubleImgesResultForm(forms.ModelForm):
    class Meta:
        model = DoubleImagesResult
        fields = ('TestImg_image', 'ReferenceImg_image', 'result', 'score_info')


