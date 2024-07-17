from django.http import HttpResponse, JsonResponse

from model.predictor import predict_tomorrow

from .models import predictor

def index(response):
    return HttpResponse("stock predictor root url")

def predict_tomorrow(request):
    prob, data = predict_tomorrow(predictor=predictor)
    data = data.to_dict(orient="records")
    object = {"prob": prob, "data": data}
    return JsonResponse(object)
