from django.http import JsonResponse

from predictor import predict_tomorrow

from .models import predictor


def get_increase_prob(request):
    prob, data = predict_tomorrow(predictor=predictor)
    data = data.to_dict(orient="records")
    object = {"prob": prob, "data": data}
    return JsonResponse(object)
