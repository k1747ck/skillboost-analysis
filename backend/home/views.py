from django.shortcuts import render, redirect, get_object_or_404
from .models import Student
import json
from collections import Counter


def home(request):

    students = Student.objects.all()

    total_students = students.count()

    # Average score
    if total_students > 0:
        average_score = sum([s.score for s in students]) / total_students
    else:
        average_score = 0

    # Top performer
    top_student = None
    if students:
        top_student = max(students, key=lambda x: x.score).name

    # Graph data
    names = [s.name for s in students]
    scores = [s.score for s in students]

    courses = [s.course for s in students]
    course_count = Counter(courses)

    course_labels = list(course_count.keys())
    course_values = list(course_count.values())

    context = {
        'students': students,
        'total_students': total_students,
        'average_score': average_score,
        'names': json.dumps(names),
        'scores': json.dumps(scores),
        'course_labels': json.dumps(course_labels),
        'course_values': json.dumps(course_values),
        'top_student': top_student
    }

    return render(request, 'home.html', context)


def add_student(request):

    if request.method == 'POST':

        name = request.POST.get('name')
        email = request.POST.get('email')
        course = request.POST.get('course')
        score = request.POST.get('score')

        Student.objects.create(
            name=name,
            email=email,
            course=course,
            score=score
        )

        return redirect('/')

    return render(request, 'add_student.html')


def delete_student(request, id):

    student = get_object_or_404(Student, id=id)
    student.delete()

    return redirect('/')


def edit_student(request, id):

    student = get_object_or_404(Student, id=id)

    if request.method == "POST":

        student.name = request.POST['name']
        student.email = request.POST['email']
        student.course = request.POST['course']
        student.score = request.POST['score']

        student.save()

        return redirect('/')

    return render(request, 'edit_student.html', {'student': student})

def predict(request):

    prediction = None

    if request.method == "POST":

        score = float(request.POST.get('score'))

        # Temporary logic (replace with ML model later)
        if score >= 80:
            prediction = "High Performer"
        elif score >= 50:
            prediction = "Average Performer"
        else:
            prediction = "Needs Improvement"

    return render(request, 'predict.html', {'prediction': prediction})