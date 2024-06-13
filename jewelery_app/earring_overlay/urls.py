from django.urls import path
from . import views

app_name = 'earring_overlay'

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('capture_frame_and_send_email/', views.capture_frame_and_send_email, name='capture_frame_and_send_email'),
    path('set_selected_earring/',views.set_selected_earring, name='set_selected_earring'),
]
