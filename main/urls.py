from django.urls import path

from . import views

urlpatterns = [
    path("", view=views.index, name="index"),
    path("get_increase_prob/", views.predict_tomorrow)]
