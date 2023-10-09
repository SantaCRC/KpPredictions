from django.db import models

class KpData(models.Model):
    date = models.DateField()
    predicted_value = models.FloatField()
