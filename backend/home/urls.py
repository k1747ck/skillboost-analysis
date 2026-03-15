from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('add/', views.add_student),
    path('delete/<int:id>/', views.delete_student),
    path('edit/<int:id>/', views.edit_student),
    path('predict/', views.predict),
]

