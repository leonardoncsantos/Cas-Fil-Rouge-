from django import forms

class ProductionPlaceForm(forms.Form):
    production_place = forms.CharField(label='Production place')
