"""
/*
* urls class
* this class is for this project's URL Configuration
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
"""signature URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from login import views
from django.urls import include
from django.conf import settings
from django.conf.urls.static import static
# 各项路径信息及调用函数
urlpatterns = [
    # 用户登录处理逻辑
    path('', views.login),
    # 管理员主页面
    path('admin/', admin.site.urls),
    # 系统主页
    path('index/', views.index),
    # 系统登录主页面
    path('login/', views.login),
    # 系统注册处理逻辑
    path('register/', views.register),
    # 退出系统处理逻辑
    path('index/logout/', views.logout),
    # 鉴定一张图片真伪处理逻辑
    path('index/identify/', views.identify),
    # 鉴定两张图片的真伪处理逻辑
    path('index/judge/', views.judge),
    # 显示鉴定真伪结果页面
    path('index/identify_result/', views.identify_result),
    # 显示鉴定真伪结果的历史记录
    path('index/identify_history/', views.identify_history),
    # 显示鉴定真伪结果页面
    path('index/judge_result/', views.judge_result),
    # 显示鉴定真伪结果的历史记录
    path('index/judge_history/', views.judge_history),
    # 个人信息
    path('index/userCenter/', views.usercenter),
    # 调用验证码
    path('captcha/', include('captcha.urls')),
]
# 读取静态文件
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

