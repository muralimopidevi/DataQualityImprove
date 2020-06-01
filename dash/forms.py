from django import forms
from django.contrib.auth.models import User
from .models import Profile
from django import forms
from .models import CSV


class UserUpdateForm(forms.ModelForm):
    first_name = forms.CharField()
    last_name = forms.CharField()
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']


class ProfileUpdateForm(forms.ModelForm):

    class Meta:
        model = Profile
        fields = ['image']


class CSVForm(forms.ModelForm):
    class Meta:
        model = CSV
        fields = ['title', 'pdf']



