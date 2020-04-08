from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('upload/', views.upload.as_view(), name='upload'),
    path('prepross/', views.prepross.as_view(), name='prepross'),
    path('profile/', views.profile, name='dash-profile'),
    path('profile/pwd/', views.change_password, name='dash-password'),
]
