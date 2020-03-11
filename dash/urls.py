from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('home/', views.home, name='dash-home'),
    path('profile/', views.profile, name='dash-profile'),
    path('profile/pwd/', views.change_password, name='dash-password'),
]
