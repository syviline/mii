import base64

from django.shortcuts import render

from .forms import PredictionForm
from .services import predict_image


def predict_view(request):
    result = None
    error = None
    preview_image = None
    uploaded_file_name = None
    uploaded_file_size = None

    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            uploaded_file_name = image.name
            uploaded_file_size = _format_file_size(image.size)
            preview_image = _build_preview_data_url(image)
            try:
                result = predict_image(
                    form.cleaned_data['model_key'],
                    image,
                )
            except Exception as exc:
                error = str(exc)
    else:
        form = PredictionForm()

    return render(
        request,
        'predictor/predict.html',
        {
            'form': form,
            'result': result,
            'error': error,
            'preview_image': preview_image,
            'uploaded_file_name': uploaded_file_name,
            'uploaded_file_size': uploaded_file_size,
        },
    )


def _build_preview_data_url(image):
    image.seek(0)
    encoded = base64.b64encode(image.read()).decode('ascii')
    image.seek(0)
    return f'data:{image.content_type};base64,{encoded}'


def _format_file_size(size):
    if size >= 1024 * 1024:
        return f'{size / (1024 * 1024):.2f} MB'
    if size >= 1024:
        return f'{size / 1024:.1f} KB'
    return f'{size} B'
