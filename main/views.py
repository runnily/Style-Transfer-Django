from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.views import View
from django.shortcuts import render
from .form import NeuralInput, ContentFormImg

# Create your views here.
class homeView(View):
    template_name = "main/home.html"
    template_name_result = "main/result.html"
    initial = {'content_layer_choice': 'block5_conv2',
               'style_layer_choice' : ['block1_conv1', 'block2_conv1', 
               'block3_conv1','block4_conv1', 'block5_conv1'],
               'alpha_choice': 0.5,
               'beta_choice': 0.5,
               'style_img_choice': 'style.jpg',
               }
    

    def get(self, request, *args,  **kwargs):
        form_neural = NeuralInput(self.initial)
        form_content = ContentFormImg()

        content = {'title':'home',
                    'form_neural':form_neural,
                    'form_content': form_content}
        return render(request, self.template_name, content)
    
    def post(self, request, *args, **kwargs):
        form_neural = NeuralInput(request.POST, request.FILES)
        form_content = ContentFormImg(request.POST, request.FILES)

        content = {'title':'home',
                    'form_neural':form_neural,
                    'form_content': form_content}

        if form_neural.is_valid() & form_content.is_valid():
            content_layer = form_neural.cleaned_data.get('content_layer_choice')
            style_layer = form_neural.cleaned_data.get('style_layer_choice')
            alpha_choice = form_neural.cleaned_data.get('alpha_choice')
            beta_choice = form_neural.cleaned_data.get('beta_choice')
            content_img = str(form_content.cleaned_data.get('content_img_choice'))
            style_img = form_neural.cleaned_data.get('style_img_choice')
            form_content.save()
            transfer = form_neural.neural(alpha=alpha_choice, beta=beta_choice, content=content_layer, style=style_layer, c_img=content_img, s_img=style_img)
            return redirect('result'), transfer.regression_total_loss()
        else:
            messages.warning(request, 'One or more inputs were invalid')
        return render(request, self.template_name, content)

