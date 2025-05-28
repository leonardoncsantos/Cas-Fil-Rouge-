# from django.conf.urls import url, include
from django.conf.urls import include
from django.urls import re_path as url
from . import views # import views so we can use them in urls.

app_name = 'costsim'

urlpatterns = [
    url(r'^insert/', views.index, name='insert'),
    url(r'^dropfile/', views.dropfile, name='dropfile'),
    url(r'^search/', views.search, name='search'),
    url(r'^calculus/$', views.calculus, name='calculus'),
]