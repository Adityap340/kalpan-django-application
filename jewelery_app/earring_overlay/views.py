import cv2
import dlib
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from imutils import face_utils
from django.core.mail import EmailMessage
from django.conf import settings
import logging
import os
BASE_DIR = settings.BASE_DIR

last_frame = None

earrings_array = []
for a,b,c in os.walk(os.path.join(BASE_DIR,"earring_overlay","static","earrings",)):
    earrings_array = c

context = {
    "data":earrings_array,
}

selected_earring = 'ring1_single.png'

p = os.path.join('assests','shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def generate_frames():
    global last_frame
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            earring1 = cv2.imread(os.path.join(BASE_DIR,"earring_overlay","static","earrings",selected_earring),cv2.IMREAD_UNCHANGED)
            earring1 = cv2.resize(earring1, (25, 65))

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rects = detector(gray, 0)

            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for i, (x, y) in enumerate(shape):
                    if i == 2 or i == 14:
                        x_offset = x - 20 if i == 2 else x - 10
                        y_offset = y
                        y1, y2 = y_offset, y_offset + earring1.shape[0]
                        x1, x2 = x_offset, x_offset + earring1.shape[1]
                        alpha_s = earring1[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            frame[y1:y2, x1:x2, c] = (alpha_s * earring1[:, :, c] +
                                                      alpha_l * frame[y1:y2, x1:x2, c])

            ret, buffer = cv2.imencode('.jpg', frame)
            last_frame = frame
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()

def set_selected_earring(request):
    global selected_earring
    if request.method == 'POST':
        selected_earring = request.POST['selected_earring']
    return JsonResponse({'status': 'success', 'message': 'Image changed successfully'})


def capture_frame_and_send_email(request):
    global last_frame  # Ensure we use the global last_frame variable
    if request.method == 'POST':
        email_address = request.POST.get('email')
        if not email_address:
            return JsonResponse({'status': 'error', 'message': 'Email Address is required'})
    if last_frame is not None:
        # Convert the frame to bytes
        _, buffer = cv2.imencode('.jpg', last_frame)
        frame_bytes = buffer.tobytes()

        # Send the email
        email = EmailMessage(
            'Captured Frame',
            'Please find the attached frame captured from the jewelry app.',
            settings.DEFAULT_FROM_EMAIL,
            [email_address],
        )

        email.attach('captured_frame.jpg', frame_bytes, 'image/jpeg')
        email.send()

        logging.debug("Captured frame and sent email successfully")
        return JsonResponse({'status': 'success', 'message': 'Frame captured and email sent successfully'})
    else:
        logging.debug("No frame available to capture")
        return JsonResponse({'status': 'error', 'message': 'No frame available'})


def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'earring_overlay/index.html',context)
