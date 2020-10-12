from django.shortcuts import render
from django.views import View

# Create your views here.
class resultView(View):
    template_name = "utils/result.html"
    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, {})