from django import forms
from django.conf import settings
from .models import ContentImg
import os



class NeuralInput(forms.Form):
    
    CONTENT_LAYERS = [('block5_conv2','block5_conv2'), ('block4_conv2','block4_conv2')] 

    STYLE_LAYERS = [('block1_conv1','block1_conv1'),
                ('block2_conv1', 'block2_conv1'),
                ('block3_conv1','block3_conv1'),
                ('block4_conv1','block4_conv1'),
                ('block5_conv1','block5_conv1')]

    MAIN_PATH =  os.path.dirname(__file__)
    MAIN_PATH =  os.path.join(MAIN_PATH, "static/utils/")
    STYLE_IMGS = [('style.jpg', 'style.jpg'),
                  ('style_1.jpg', 'style_1.jpg')]

    style_img_choice = forms.ChoiceField(required=True, 
    widget=forms.RadioSelect, choices=STYLE_IMGS)
    
    content_layer_choice = forms.ChoiceField(required=True, widget=forms.RadioSelect,
    choices=CONTENT_LAYERS)

    style_layer_choice = forms.MultipleChoiceField(required=True, 
    widget=forms.CheckboxSelectMultiple, choices=STYLE_LAYERS)

    alpha_choice = forms.FloatField(max_value=1e-1, min_value=1e-10)
    beta_choice = forms.FloatField(max_value=1e-1, min_value=1e-10)
    step = 1e-2*0.1
    alpha_choice.widget.attrs.update({'class':'custom-range'})
    beta_choice.widget.attrs.update({'class':'custom-range'})

class ContentFormImg(forms.ModelForm):
    
    class Meta:
        model = ContentImg
        fields = ['content_img_choice']

