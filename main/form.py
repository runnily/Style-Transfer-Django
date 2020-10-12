from django import forms
from django.conf import settings
from .models import ContentImg
from utils.style_transfer import Neural
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

    alpha_choice = forms.DecimalField(max_value=10, min_value=0.2)
    beta_choice = forms.DecimalField(max_value=10, min_value=0.2)

    def get_style_choice(self):
        pass

    def neural(self,**kwargs):
        return Neural(float(kwargs['alpha']), float(kwargs['beta']), [kwargs['content']], kwargs['style'], kwargs['s_img'], kwargs['c_img'])

class ContentFormImg(forms.ModelForm):
    
    class Meta:
        model = ContentImg
        fields = ['content_img_choice']

