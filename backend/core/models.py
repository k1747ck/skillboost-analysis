from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    course = models.CharField(max_length=100)
    score = models.FloatField()
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

# Create your models here.
