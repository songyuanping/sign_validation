"""
/*
* wsgi class
* this class is for this project's WSGI config
* created by Su Linyu && Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
"""
WSGI config for signature project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'signature.settings')

application = get_wsgi_application()
