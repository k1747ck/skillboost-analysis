from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Avg
from django.utils.safestring import mark_safe
from .models import Student, StudentUser
import json
import joblib
import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Model', 'outputs')

try:
    model     = joblib.load(os.path.join(MODEL_DIR, 'student_model.pkl'))
    scaler    = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    encoders  = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    ML_LOADED = True
except Exception as e:
    print(f"ML model not loaded: {e}")
    ML_LOADED = False


def student_login(request):
    error = None
    if request.method == 'POST':
        roll   = request.POST['roll_number']
        passwd = request.POST['password']
        try:
            user = StudentUser.objects.get(roll_number=roll, password=passwd)
            request.session['roll_number'] = user.roll_number
            request.session['student_name'] = user.student.name if user.student else roll
            return redirect('/')
        except StudentUser.DoesNotExist:
            error = "Invalid roll number or password."
    return render(request, 'home/stud_login.html', {'error': error})


def student_logout(request):
    request.session.flush()
    return redirect('/login/')


def mentor(request):
    return render(request, 'home/mentor.html')


def login_required_check(request):
    return 'roll_number' not in request.session


def home(request):
    if login_required_check(request):
        return redirect('/login/')

    students      = Student.objects.all()
    total_students = students.count()
    avg_score      = students.aggregate(Avg('aggregate_academic_score'))['aggregate_academic_score__avg']
    average_score  = round(avg_score, 2) if avg_score else 0
    top_student    = students.order_by('-aggregate_academic_score').first()

    names  = mark_safe(json.dumps([s.name for s in students]))
    scores = mark_safe(json.dumps([float(s.aggregate_academic_score) for s in students]))

    course_counts = {}
    for s in students:
        course_counts[s.course] = course_counts.get(s.course, 0) + 1
    course_labels = mark_safe(json.dumps(list(course_counts.keys())))
    course_values = mark_safe(json.dumps(list(course_counts.values())))

    return render(request, 'home/home.html', {
        'students':        students,
        'total_students':  total_students,
        'average_score':   average_score,
        'top_student':     top_student,
        'names':           names,
        'scores':          scores,
        'course_labels':   course_labels,
        'course_values':   course_values,
        'student_name':    request.session.get('student_name', ''),
    })


def add_student(request):
    if login_required_check(request):
        return redirect('/login/')
    if request.method == 'POST':
        student = Student.objects.create(
            student_id               = request.POST['student_id'],
            name                     = request.POST['name'],
            email                    = request.POST['email'],
            course                   = request.POST['course'],
            attendance_percentage    = request.POST['attendance_percentage'],
            mid_term_marks           = request.POST['mid_term_marks'],
            assignment_score         = request.POST['assignment_score'],
            class_participation      = request.POST['class_participation'],
            activity_participation   = request.POST['activity_participation'],
            aggregate_academic_score = request.POST['aggregate_academic_score'],
            training_attendance      = request.POST['training_attendance'],
            training_score           = request.POST['training_score'],
            feedback_rating          = request.POST['feedback_rating'],
            pre_training_score       = request.POST['pre_training_score'],
            post_training_score      = request.POST['post_training_score'],
            improvement              = request.POST['improvement'],
        )
        if ML_LOADED:
            try:
                act_enc   = int(encoders['Activity_Participation'].transform([student.activity_participation])[0])
                train_enc = int(encoders['Training_Attendance'].transform([student.training_attendance])[0])
                features  = np.array([[
                    student.attendance_percentage, student.mid_term_marks,
                    student.assignment_score, student.class_participation,
                    act_enc, student.aggregate_academic_score,
                    train_enc, student.training_score, student.feedback_rating,
                    student.pre_training_score, student.post_training_score,
                    student.improvement,
                ]])
                pred = model.predict(scaler.transform(features))[0]
                student.performance_label = encoders['Performance_Label'].inverse_transform([pred])[0]
                student.save()
            except Exception as e:
                print(f"Auto-predict failed: {e}")
        return redirect('/')
    return render(request, 'home/add_student.html')


def edit_student(request, pk):
    if login_required_check(request):
        return redirect('/login/')
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        student.student_id               = request.POST['student_id']
        student.name                     = request.POST['name']
        student.email                    = request.POST['email']
        student.course                   = request.POST['course']
        student.attendance_percentage    = request.POST['attendance_percentage']
        student.mid_term_marks           = request.POST['mid_term_marks']
        student.assignment_score         = request.POST['assignment_score']
        student.class_participation      = request.POST['class_participation']
        student.activity_participation   = request.POST['activity_participation']
        student.aggregate_academic_score = request.POST['aggregate_academic_score']
        student.training_attendance      = request.POST['training_attendance']
        student.training_score           = request.POST['training_score']
        student.feedback_rating          = request.POST['feedback_rating']
        student.pre_training_score       = request.POST['pre_training_score']
        student.post_training_score      = request.POST['post_training_score']
        student.improvement              = request.POST['improvement']
        student.save()
        return redirect('/')
    return render(request, 'home/edit_student.html', {'student': student})


def delete_student(request, pk):
    if login_required_check(request):
        return redirect('/login/')
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        student.delete()
        return redirect('/')
    return render(request, 'home/confirm_delete.html', {'student': student})


def student_list(request):
    if login_required_check(request):
        return redirect('/login/')
    students = Student.objects.all().order_by('-aggregate_academic_score')
    return render(request, 'home/student_list.html', {'students': students})


def predict(request):
    if login_required_check(request):
        return redirect('/login/')
    prediction  = None
    probability = None
    error       = None

    if not ML_LOADED:
        error = "ML model not loaded. Please run training script first."
        return render(request, 'home/predict.html', {'error': error})

    if request.method == 'POST':
        try:
            act_part  = request.POST['activity_participation'].strip().lower()
            train_att = request.POST['training_attendance'].strip().lower()
            act_enc   = int(encoders['Activity_Participation'].transform([act_part])[0])
            train_enc = int(encoders['Training_Attendance'].transform([train_att])[0])
            features  = np.array([[
                float(request.POST['attendance_percentage']),
                float(request.POST['mid_term_marks']),
                float(request.POST['assignment_score']),
                float(request.POST['class_participation']),
                act_enc,
                float(request.POST['aggregate_academic_score']),
                train_enc,
                float(request.POST['training_score']),
                float(request.POST['feedback_rating']),
                float(request.POST['pre_training_score']),
                float(request.POST['post_training_score']),
                float(request.POST['improvement']),
            ]])
            features_scaled = scaler.transform(features)
            pred_encoded    = model.predict(features_scaled)[0]
            pred_label      = encoders['Performance_Label'].inverse_transform([pred_encoded])[0]
            proba           = model.predict_proba(features_scaled)[0]
            prediction      = pred_label.capitalize()
            probability     = {
                encoders['Performance_Label'].classes_[i].capitalize(): round(float(p) * 100, 1)
                for i, p in enumerate(proba)
            }
        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render(request, 'home/predict.html', {
        'prediction':  prediction,
        'probability': probability,
        'error':       error,
    })