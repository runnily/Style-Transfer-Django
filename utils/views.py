from django.shortcuts import render
from django.views import View
from .style_transfer import Neural

# Create your views here.
class resultView(View):
    template_name = "utils/result.html"
    def get(self, request , *args, **kwargs):
        n = Neural(alpha=request.session.get('alpha_choice'), beta=request.session.get('beta_choice'), 
                content_layers=[request.session.get('content_layer')], style_layers=request.session.get('style_layer'), 
                style_path=request.session.get('style_img'), content_path=request.session.get('content_img'),)
        return render(request, self.template_name, {"n" : n})