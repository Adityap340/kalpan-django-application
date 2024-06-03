import cv2
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.core.mail import EmailMessage
from django.conf import settings
import logging

last_frame = None

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image of the necklace
necklace = cv2.imread("D:\\Internship\\necklace6.png", cv2.IMREAD_UNCHANGED)

def necklace_overlay_video_feed():
    global last_frame
    cap = cv2.VideoCapture(0)

    # Initialize variables to store the previous position of the necklace
    prev_necklace_x = 0
    prev_necklace_y = 0

    # Smoothing factor
    alpha = 0.2

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop over the detected faces
        for (x, y, w, h) in faces:
            # Calculate the position of the neck
            neck_x = x + w // 2
            neck_y = y + h + int(h * 0.1)  # Place the necklace slightly below the face

            # Update the position of the necklace using smoothing factor
            necklace_x = int((1 - alpha) * prev_necklace_x + alpha * neck_x)
            necklace_y = int((1 - alpha) * prev_necklace_y + alpha * neck_y)

            # Update the previous position of the necklace
            prev_necklace_x = necklace_x
            prev_necklace_y = necklace_y

            # Resize the necklace image to fit the neck
            scale_factor = h / necklace.shape[0]  # Adjust the scale factor as needed
            necklace_resized = cv2.resize(necklace, (int(necklace.shape[1] * scale_factor), h))

            # Calculate the position to overlay the necklace
            overlay_x = necklace_x - necklace_resized.shape[1] // 2
            overlay_y = necklace_y

            # Ensure that the overlay region stays within the bounds of the input image
            overlay_x = max(0, overlay_x)
            overlay_y = max(0, overlay_y)
            overlay_y_end = min(overlay_y + necklace_resized.shape[0], frame.shape[0])
            overlay_x_end = min(overlay_x + necklace_resized.shape[1], frame.shape[1])

            # Ensure the overlay dimensions are valid
            if overlay_x_end > overlay_x and overlay_y_end > overlay_y:
                for c in range(0, 3):
                    frame[overlay_y:overlay_y_end, overlay_x:overlay_x_end, c] = (
                        necklace_resized[0:overlay_y_end - overlay_y, 0:overlay_x_end - overlay_x, c] *
                        (necklace_resized[0:overlay_y_end - overlay_y, 0:overlay_x_end - overlay_x, 3] / 255.0) +
                        frame[overlay_y:overlay_y_end, overlay_x:overlay_x_end, c] *
                        (1.0 - necklace_resized[0:overlay_y_end - overlay_y, 0:overlay_x_end - overlay_x, 3] / 255.0)
                    )

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        last_frame = frame
        frame = buffer.tobytes()

        # Yield the frame in a byte format suitable for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

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
    return StreamingHttpResponse(necklace_overlay_video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'necklace_overlay/index.html')
