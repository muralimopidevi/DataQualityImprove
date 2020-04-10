from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.UploadBookView.as_view(), name='upload'),
    path('index/', views.index.as_view(), name='index'),
    path('prepross/', views.prepross.as_view(), name='prepross'),
    path('profile/', views.profile, name='dash-profile'),
    path('profile/pwd/', views.change_password, name='dash-password'),
]
