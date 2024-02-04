from django.urls import path

from . import views

urlpatterns = [path("get_increase_prob/", views.get_increase_prob)]
