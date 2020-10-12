from django.db import models
from django.conf import settings

# Create your models here.
class ContentImg(models.Model):
    content_img_choice = models.ImageField(upload_to='input/')

