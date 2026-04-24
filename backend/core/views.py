from django.shortcuts import render, redirect
from .models import Student
from django.db.models import Avg
import matplotlib.pyplot as plt
import io
import base64



def home(request):
    students = Student.objects.all()

    total_students = students.count()
    average_score = students.aggregate(Avg('score'))['score__avg']
    top_student = students.order_by('-score').first()
    top_students = students.order_by('-score')[:5]

    scores = [s.score for s in students]

    graph = None
    if scores:
        plt.figure()
        plt.hist(scores, bins=5)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

    context = {
        'total_students': total_students,
        'average_score': average_score,
        'top_student': top_student,
        'top_students': top_students,
        'graph': graph
    }

    return render(request, 'home.html', context)



def add_student(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        course = request.POST['course']
        score = request.POST['score']

        Student.objects.create(
            name=name,
            email=email,
            course=course,
            score=score
        )

        return redirect('/')

    return render(request, 'add_student.html')

def student_list(request):
    students = Student.objects.all()
    return render(request, 'student_list.html', {'students': students})

