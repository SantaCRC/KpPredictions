from django.shortcuts import render
from .models import KpData
import random

def home(request):
    real = random.sample(range(1, 100), 10)
    CNN = random.sample(range(1, 100), 10)
    LSTM = random.sample(range(1, 100), 10)
    return render(request, 'charts.html', {'real': real, 'CNN': CNN, 'LSTM': LSTM})
