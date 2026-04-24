import pandas as pd
import os
from django.core.management.base import BaseCommand
from home.models import Student

class Command(BaseCommand):
    help = 'Import students from CSV'

    def handle(self, *args, **kwargs):
        BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'Model')
        students_df = pd.read_csv(os.path.join(BASE, 'students_clean.csv'))
        training_df = pd.read_csv(os.path.join(BASE, 'training.csv'))
        df = pd.merge(students_df, training_df, on='Student_ID', how='left')
        df['Training_Attendance'] = df['Training_Attendance'].fillna('no')
        df['Training_Score'] = df['Training_Score'].fillna(0)
        df['Feedback_Rating'] = df['Feedback_Rating'].fillna(0)
        df['Pre_Training_Score'] = df['Pre_Training_Score'].fillna(0)
        df['Post_Training_Score'] = df['Post_Training_Score'].fillna(0)
        df['Improvement'] = df['Improvement'].fillna(0)
        created = 0
        skipped = 0
        for _, row in df.iterrows():
            sid = str(row['Student_ID']).strip()
            if Student.objects.filter(student_id=sid).exists():
                skipped += 1
                continue
            Student.objects.create(
                student_id=sid,
                name=str(row.get('Student_Name', f'Student {sid}')),
                email=str(row.get('Email', f'{sid}@skillboost.com')),
                course=str(row.get('Course_Name', 'Unknown')),
                attendance_percentage=float(row.get('Attendance_Percentage', 0)),
                mid_term_marks=float(row.get('Mid_Term_Marks', 0)),
                assignment_score=float(row.get('Assignment_Score', 0)),
                class_participation=float(row.get('Class_Participation', 0)),
                activity_participation=str(row.get('Activity_Participation', 'no')).strip().lower(),
                aggregate_academic_score=float(row.get('Aggregate_Academic_Score', 0)),
                training_attendance=str(row.get('Training_Attendance', 'no')).strip().lower(),
                training_score=float(row.get('Training_Score', 0)),
                feedback_rating=float(row.get('Feedback_Rating', 0)),
                pre_training_score=float(row.get('Pre_Training_Score', 0)),
                post_training_score=float(row.get('Post_Training_Score', 0)),
                improvement=float(row.get('Improvement', 0)),
                performance_label=str(row.get('Performance_Label', '')).strip().lower(),
            )
            created += 1
        self.stdout.write(f'Done! Created: {created}, Skipped: {skipped}')
