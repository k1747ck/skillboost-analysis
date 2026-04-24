from django.contrib import admin
from .models import Student, StudentUser

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display  = ['student_id', 'name', 'course', 'aggregate_academic_score', 'performance_label', 'training_attendance']
    search_fields = ['name', 'student_id', 'course']
    list_filter   = ['course', 'performance_label', 'training_attendance']

@admin.register(StudentUser)
class StudentUserAdmin(admin.ModelAdmin):
    list_display = ['roll_number', 'student']