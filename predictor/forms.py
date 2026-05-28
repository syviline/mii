from django import forms

from .ml_config import get_model_choices


class PredictionForm(forms.Form):
    model_key = forms.ChoiceField(
        label='Модель',
        choices=get_model_choices,
    )
    image = forms.ImageField(label='Изображение')
