from django.urls import path
from . import views

urlpatterns = [
    path('',              views.home,           name='home'),
    path('add/',          views.add_student,    name='add_student'),
    path('edit/<int:pk>/', views.edit_student,  name='edit_student'),
    path('delete/<int:pk>/', views.delete_student, name='delete_student'),
    path('students/',     views.student_list,   name='student_list'),
    path('predict/',      views.predict,        name='predict'),
    path('login/',        views.student_login,  name='login'),
    path('logout/',       views.student_logout, name='logout'),
    path('mentor/',       views.mentor,         name='mentor'),
]