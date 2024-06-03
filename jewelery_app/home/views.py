# home/views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def home(request):
    return render(request, 'home/index.html')

