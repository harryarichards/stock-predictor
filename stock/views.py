from django.http import HttpResponse

from predictor import predict_tomorrow

from .models import predictor


def get_increase_prob(request):
    prob = predict_tomorrow(predictor=predictor)
    return HttpResponse(prob)
