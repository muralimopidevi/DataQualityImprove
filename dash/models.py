from django.contrib.auth.models import User
from PIL import Image
from django.db import models
from dash.csvstore import OverwriteStorage


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username}Profile'

    def save(self, *args, **kwargs):
        super(Profile,self).save(*args, **kwargs)

        img = Image.open(self.image.path)
        if img.height > 600 or img.width > 600:
            output_size = (600, 600)
            img.thumbnail(output_size)
            img.save(self.image.path)


class DownloadedFile(models.Model):
    docfile = models.FileField(storage=OverwriteStorage(), upload_to='CSV_FOLDER/')


class CurrentFile(models.Model):
    filename = models.CharField(max_length=300)


class Prepross(models.Model):
    filename = models.CharField(max_length=300)
    coltype = models.CharField(max_length=300)
    assvar = models.CharField(max_length=300)
    missingvalues = models.CharField(max_length=300)
    trainingset_size = models.IntegerField()
    featscaling = models.CharField(max_length=300)
    ordinal = models.CharField(max_length=300)