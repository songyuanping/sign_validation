"""
/*
* views class
* this class is for realize the main function
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
import datetime
import hashlib
import os

import numpy as np
from django.shortcuts import redirect
from django.shortcuts import render

from model.config import Config
from model.image_identify import IdentifyImage
from model.image_judge import JudgeImage
from . import forms
from . import models
from .forms import ImagesForm, DoubleImagesForm
from .models import Images, DoubleImages, ImagesResult, DoubleImagesResult


# 主页函数
def index(request):
    # 若未登录，则返回登录界面
    if not request.session.get('is_login', None):
        return redirect('/login/')
    # 否则进入主页界面
    return render(request, 'login/index.html')


#  登录函数
def login(request):
    # 不允许重复登录
    if request.session.get('is_login', None):
        return redirect('/index/')
    # 若读到POST请求
    if request.method == 'POST':
        # 将UserForm表单内容存入login_form
        login_form = forms.UserForm(request.POST)
        message = '请检查填写的内容！'
        # 各项内容正确填写时
        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')
            # 在数据库中读入用户信息，存在则赋值给user
            try:
                user = models.User.objects.get(name=username)
            except:
                message = '用户不存在！'
                return render(request, 'login/login.html', locals())
            # 且用户名与密码相匹配
            if user.password == hash_code(password):
                # 修改用户状态
                request.session['is_login'] = True
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                # 进入主页
                return redirect('/index/')
            # 密码不匹配
            else:
                message = '密码不正确！'
                return render(request, 'login/login.html', locals())
        # 未正确填写内容时候
        else:
            return render(request, 'login/login.html', locals())

    login_form = forms.UserForm()
    return render(request, 'login/login.html', locals())


# 注册函数
def register(request):
    # 读到post请求时
    if request.method == 'POST':
        # 将RegisterForm表单内容存入register_form
        register_form = forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():
            # 赋值给对应对象
            username = register_form.cleaned_data.get('username')
            password1 = register_form.cleaned_data.get('password1')
            password2 = register_form.cleaned_data.get('password2')
            email = register_form.cleaned_data.get('email')
            # 验证两次密码是否一致
            if password1 != password2:
                message = '两次输入的密码不同！'
                return render(request, 'login/register.html', locals())
            # 一致的情况下进行下一步
            else:
                # 同名不可注册
                same_name_user = models.User.objects.filter(name=username)
                if same_name_user:
                    message = '用户名已经存在'
                    return render(request, 'login/register.html', locals())
                # 邮箱也不可使用两次
                same_email_user = models.User.objects.filter(email=email)
                if same_email_user:
                    message = '该邮箱已经被注册了！'
                    return render(request, 'login/register.html', locals())
                # 将内容存入新的用户信息中
                new_user = models.User()
                new_user.name = username
                new_user.password = hash_code(password1)
                new_user.email = email
                new_user.save()
                # 注册成功返回登录界面
                return redirect('/login/')
        # 注册未成功停留在注册界面
        else:
            return render(request, 'login/register.html', locals())
    register_srform = forms.RegisterForm()
    return render(request, 'login/register.html', locals())


# 登出函数
def logout(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("/login/")
    # 清除所有状态
    request.session.flush()
    # 返回登录界面
    return redirect("/login/")


# 密码加密函数 用户登录密码加密
def hash_code(s, salt='mysite'):
    # 用哈希256方式加密
    h = hashlib.sha256()
    s += salt
    h.update(s.encode())  # update方法只接收bytes类型
    return h.hexdigest()


# 确认函数
def make_confirm_string(user):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    code = hash_code(user.name, now)
    models.ConfirmString.objects.create(code=code, user=user, )
    return code


# 签名鉴别函数
def identify(request):
    if request.method == 'POST':
        # 将ImagesForm表单内容存入form
        form = ImagesForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # 上传成功进入结果界面
            return redirect('/index/identify_result')
    # 不然需要重新上传
    else:
        form = ImagesForm()
    return render(request, 'login/identify.html', {'form': form})


# 签名鉴别结果展示界面
def identify_result(request):
    # 获得用户上传的图片对象
    images = Images.objects.last()
    # 得到用户上传图片的路径
    image_name = os.path.join('.', 'model', str(images.Image))
    # 将用户上传的图片与数据库中原有的图片进行比对分析，判断用户上传的图片是否为真签名
    identify_image = IdentifyImage(image_name)
    # 返回相似度最高的的预测概率及图片的路径名称
    predict_list, orig_image_list = identify_image.identify_image()
    # 获得相似度最高的5张图片的路径及其得分
    for _index, item in enumerate(orig_image_list):
        dirs = str(item[0]).split(os.path.sep)
        path = os.path.sep + os.path.sep.join(dirs[-5:])
        orig_image_list[_index] = path
        score = predict_list[_index] * 100
        predict_list[_index] = str(score)[:5]

    results = {}
    for item in zip(orig_image_list, predict_list):
        # print(item[0],item[1])
        results[item[0]] = item[1]
        # print(results)
    imageRef = orig_image_list[0]
    score = predict_list[0]
    # print(orig_image_list[0])
    # print(predict_list[0])
    id = request.session.get('user_id')
    # print(id)
    user = models.User.objects.get(id=id)
    # 将结果存入结果模型
    new_result = models.ImagesResult()
    new_result.user_name = user.name
    new_result.Images = images.Image.url
    new_result.ImagesRef = imageRef
    new_result.result = score
    new_result.save()
    return render(request, 'login/identify_result.html',
                  {'images': images, 'image_list': orig_image_list,
                   'predict_list': predict_list, 'result': results, 'imageRef': imageRef,
                   'score': score})


# 查看个人历史上传记录
def identify_history(request):
    if request.method == 'GET':
        id = request.session.get('user_id')
        user = models.User.objects.get(id=id)
        try:
            imagesResult = ImagesResult.objects.filter(user_name=user.name).order_by('-id')
        except Exception as e:
            print(e)
            return render(request, 'login/identify.html', locals())
        return render(request, 'login/identify_history.html', locals())


# 查看个人历史上传记录
def identify_history(request):
    if request.method == 'GET':
        id = request.session.get('user_id')
        user = models.User.objects.get(id=id)
        try:
            imagesResult = ImagesResult.objects.filter(user_name=user.name).order_by('-id')
            # print(imagesResult)
            # for result in doubleIamgesResult:
            #     print(result.id)
        except Exception as e:
            print(e)
            return render(request, 'login/identify.html', locals())
        return render(request, 'login/identify_history.html', locals())
    else:
        imgId = request.POST.get('deleteOBJ')
        # print(imgId)
        models.ImagesResult.objects.get(id=imgId).delete()
        id = request.session.get('user_id')
        # print(id)
        user = models.User.objects.get(id=id)
        try:
            imagesResult = ImagesResult.objects.filter(user_name=user.name).order_by('-id')
            # print(imagesResult)
            # for result in doubleIamgesResult:
            #     print(result.id)
        except Exception as e:
            print(e)
            return render(request, 'login/identify.html', locals())
        return render(request, 'login/identify_history.html', locals())


# 签名判断函数
def judge(request):
    if request.method == 'POST':
        # # 将DoubleImagesForm表单内容存入form
        form = DoubleImagesForm(request.POST, request.FILES)
        # 合理即保存
        if form.is_valid():
            form.save()
            # 并跳转至结果界面
            return redirect('/index/judge_result')
    # 不然重新输入
    else:
        form = DoubleImagesForm()
    return render(request, 'login/judge.html', {'form': form})


# 签名判断界面展示界面
def judge_result(request):
    # 显示上次上传得界面
    doubleiamges = DoubleImages.objects.last()
    # 获得用户上传签名图片的路径
    test_image_name = os.path.join('.', 'model', str(doubleiamges.TestImg_image))
    refer_image_name = os.path.join('.', 'model', str(doubleiamges.ReferenceImg_image))
    # 传入签名图片的路径信息
    judgeImage = JudgeImage(test_image_name, refer_image_name)
    # 完成签名识别并返回结果
    predict_list = judgeImage.judge_image()
    true_thresh = np.exp(-Config.thresh) * 100
    # score即为用户上传的两张照片的相似度new_test_image_name和new_refer_image_name为两张图片处理后的路径
    score = np.exp(-np.mean(predict_list)) * 100
    # 根据阈值来判断需鉴别的签名是否为同一人所写
    score_info = "真签名" if score >= true_thresh else "假签名"
    id = request.session.get('user_id')
    user = models.User.objects.get(id=id)
    # 将结果存入结果模型
    new_result = models.DoubleImagesResult()
    new_result.user_name = user.name
    new_result.TestImg_image = doubleiamges.TestImg_image
    new_result.ReferenceImg_image = doubleiamges.ReferenceImg_image
    new_result.result = str(score)[:5]
    new_result.score_info = score_info
    new_result.save()
    # print(new_result.ReferenceImg_image.url)
    # print(new_result.ReferenceImg_image)
    return render(request, 'login/judge_result.html',
                  {'result': new_result})


# 查看个人历史上传记录
def judge_history(request):
    if request.method == 'GET':
        id = request.session.get('user_id')
        user = models.User.objects.get(id=id)
        try:
            doubleIamgesResult = DoubleImagesResult.objects.filter(user_name=user.name).order_by('-id')
            # print(doubleIamgesResult)
            # for result in doubleIamgesResult:
            #     print(result.id)
        except Exception as e:
            print(e)
            return render(request, 'login/judge.html', locals())
        return render(request, 'login/judge_history.html', locals())
    else:
        imgId = request.POST.get('deleteOBJ')
        # print(imgId)
        models.DoubleImagesResult.objects.get(id=imgId).delete()
        id = request.session.get('user_id')
        # print(id)
        user = models.User.objects.get(id=id)
        try:
            doubleIamgesResult = DoubleImagesResult.objects.filter(user_name=user.name).order_by('-id')
            # print(doubleIamgesResult)
            # for result in doubleIamgesResult:
            #     print(result.id)
        except Exception as e:
            print(e)
            return render(request, 'login/judge.html', locals())
        return render(request, 'login/judge_history.html', locals())


def usercenter(request):
    if request.method == 'GET':
        id = request.session.get('user_id')
        try:
            user = models.User.objects.get(id=id)
            name = user.name
            password = user.password
            email = user.email
        except Exception as e:
            print(e)
            return render(request, 'login/userCenter.html', locals())
        return render(request, 'login/userCenter.html', locals())
    else:
        # 获取表单信息
        name = request.POST.get('name')
        password = request.POST.get('password')
        email = request.POST.get('email')
        id = request.session.get('user_id')
        try:
            user = models.User.objects.get(id=id)
            models.User.objects.filter(id=id).update(name=name, password=hash_code(password), email=email)
            request.session.flush()
            return redirect('/index/')
        except Exception as e:
            print(e)
            print('输入姓名有误')
            return render(request, 'login/userCenter.html', locals())
