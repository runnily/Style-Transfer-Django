from django.shortcuts import render
from django.views import View
from django.shortcuts import render

# Create your views here.
class homeView(View):
    template_name = "main/home.html"
    content = { "title" : "Home"}

    def get(self, request):
        return render(request, self.template_name, self.content)
