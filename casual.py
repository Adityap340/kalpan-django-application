from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.core.mail import EmailMessage
from django.conf import settings
import cv2
import mediapipe as mp
import logging
import os
from io import BytesIO
from django.core.files.base import ContentFile

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the last frame
last_frame = None

def handpose_video_feed():
    global last_frame
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processing the frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Add your processing logic here (for overlaying rings, etc.)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame = buffer.tobytes()
        last_frame = frame_bgr  # Update the global variable with the last processed frame
        logging.debug("Updated last_frame")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    hands.close()

def capture_frame_and_send_email(request):
    global last_frame  # Ensure we use the global last_frame variable
    if last_frame is not None:
        # Convert the frame to bytes
        _, buffer = cv2.imencode('.jpg', last_frame)
        frame_bytes = buffer.tobytes()

        # Send the email
        email = EmailMessage(
            'Captured Frame',
            'Please find the attached frame captured from the jewelry app.',
            settings.DEFAULT_FROM_EMAIL,
            ['recipient@example.com'],
        )

        email.attach('captured_frame.jpg', frame_bytes, 'image/jpeg')
        email.send()

        logging.debug("Captured frame and sent email successfully")
        return JsonResponse({'status': 'success', 'message': 'Frame captured and email sent successfully'})
    else:
        logging.debug("No frame available to capture")
        return JsonResponse({'status': 'error', 'message': 'No frame available'})

def video_feed(request):
    return StreamingHttpResponse(handpose_video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'handestimation/index.html')
