from django.conf import settings

from . import preprocessing


MODEL_CONFIGS = {
    # Replace this sample entry with your real model file and labels.
    'vehicle': {
        'title': 'Vehicle',
        'path': settings.MODEL_STORAGE_DIR / 'baseline_vehicle_model.keras',
        'preprocess': preprocessing.preprocess_image_vehicle,
        'labels': [
            'Auto Rickshaws',
            'Bikes',
            'Cars',
            'Motorcycles',
            'Planes',
            'Ships',
            'Trains'
        ]
    },
    'cat-dog-dense-128': {
        'title': 'Cat Dog Dense 128x128',
        'path': settings.MODEL_STORAGE_DIR / 'cats_dogs.keras',
        'preprocess': preprocessing.dense_rgb_128_raw,
        'labels': ['cats', 'dogs'],
        'from_logits': True,
    }
}


def get_model_choices():
    return [
        (model_key, config['title'])
        for model_key, config in MODEL_CONFIGS.items()
    ]


def get_model_config(model_key):
    return MODEL_CONFIGS[model_key]
