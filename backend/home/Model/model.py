from django.db import models

class Student(models.Model):
    student_id = models.CharField(max_length=20, unique=True, default='S000')
    name = models.CharField(max_length=100)
    email = models.EmailField()
    course = models.CharField(max_length=100)
    attendance_percentage = models.FloatField(default=0)
    mid_term_marks = models.FloatField(default=0)
    assignment_score = models.FloatField(default=0)
    class_participation = models.FloatField(default=0)
    activity_participation = models.CharField(max_length=10, choices=[('yes','Yes'),('no','No')], default='no')
    aggregate_academic_score = models.FloatField(default=0)
    training_attendance = models.CharField(max_length=10, choices=[('yes','Yes'),('no','No')], default='no')
    training_score = models.FloatField(default=0)
    feedback_rating = models.FloatField(default=0)
    pre_training_score = models.FloatField(default=0)
    post_training_score = models.FloatField(default=0)
    improvement = models.FloatField(default=0)
    performance_label = models.CharField(max_length=20, blank=True, null=True)
    date_added = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self):
        return f"{self.name} ({self.student_id})"


class StudentUser(models.Model):
    roll_number = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=100)
    student = models.OneToOneField(Student, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.roll_number