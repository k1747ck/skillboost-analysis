from django.urls import path
from django.contrib import admin
from . import views

urlpatterns = [
    path('', views.home),
    path('add/', views.add_student),
    path('students/', views.student_list),

]

