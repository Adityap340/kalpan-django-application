from django.shortcuts import render
import cv2
import mediapipe as mp
import cvzone
from google.protobuf.json_format import MessageToDict
from django.http import StreamingHttpResponse, JsonResponse
from django.core.mail import EmailMessage
from django.conf import settings
import logging

import os
BASE_DIR = settings.BASE_DIR

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
last_frame = None


rings_array = []
for a,b,c in os.walk(os.path.join(BASE_DIR,"handestimation","static","rings",)): 
    rings_array = c

context = {
    "data":rings_array,
}
selected_ring = 'ring1.png'
def handpose_video_feed():    
    # if request.method == 'GET':
    global last_frame
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_handedness:
                    label = MessageToDict(hand)['classification'][0]['label']
                    for hand_landmarks in results.multi_hand_landmarks:
                        if hand_landmarks.landmark:
                            ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                            ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                            middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

                            h, w, _ = frame_rgb.shape
                            ring_finger_pip_x = int(ring_finger_pip.x * w)
                            ring_finger_pip_y = int(ring_finger_pip.y * h)
                            ring_finger_mcp_x = int(ring_finger_mcp.x * w)
                            ring_finger_mcp_y = int(ring_finger_mcp.y * h)
                            middle_finger_pip_x = int(middle_finger_pip.x * w)
                            middle_finger_pip_y = int(middle_finger_pip.y * h)

                            distance = ((ring_finger_pip_x - ring_finger_mcp_x) ** 2 + (
                                        ring_finger_pip_y - ring_finger_mcp_y) ** 2) ** 0.45
                            scaling_factor = int(distance)

                            if label == 'Left' and ring_finger_pip_x > middle_finger_pip_x:
                                average_x = (ring_finger_mcp_x + ring_finger_pip_x) // 2
                                average_y = (int(ring_finger_mcp_y) + ring_finger_pip_y) // 2

                                overlay_image = cv2.imread(os.path.join(BASE_DIR,"handestimation","static","rings",selected_ring),
                                                        cv2.IMREAD_UNCHANGED)
                                overlay_image = cv2.resize(overlay_image, (scaling_factor, scaling_factor))
                                frame_rgb = cvzone.overlayPNG(frame_rgb, overlay_image,
                                                            [average_x - scaling_factor // 2,
                                                            average_y - scaling_factor // 2])
                            elif label == 'Right' and ring_finger_pip_x < middle_finger_pip_x:
                                average_x = (ring_finger_mcp_x + ring_finger_pip_x) // 2
                                average_y = (int(ring_finger_mcp_y) + ring_finger_pip_y) // 2

                                overlay_image = cv2.imread(os.path.join(BASE_DIR,"handestimation","static","rings",selected_ring),
                                                        cv2.IMREAD_UNCHANGED)
                                overlay_image = cv2.resize(overlay_image, (scaling_factor, scaling_factor))
                                frame_rgb = cvzone.overlayPNG(frame_rgb, overlay_image,
                                                            [average_x - scaling_factor // 2, average_y - scaling_factor // 2])
                            else:
                                cv2.putText(frame_rgb, "Please show backside of your hand", (50, 50),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
                        else:
                            cv2.putText(frame_rgb, "Please show your hands (No Hands Detected)", (50, 50),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()
            last_frame = frame_bgr
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()
        hands.close()

def set_selected_ring(request):
    global selected_ring
    if request.method == 'POST':
        selected_ring = request.POST['selected_ring']
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
    return StreamingHttpResponse(handpose_video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'handestimation/index.html',context)




