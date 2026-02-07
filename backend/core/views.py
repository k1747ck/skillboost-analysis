from django.shortcuts import render

def home(request):
    context = {
        "name": "Khushi",
        "role": "Learning Full Stack with Django "
    }
    return render(request, "home.html", context)
