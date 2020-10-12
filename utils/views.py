from django.shortcuts import render
from django.views import View
from main.views import transfer

# Create your views here.
class resultView(View):
    template_name = "utils/result.html"
    if transfer:
        print("here")
    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, {})