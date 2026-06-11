from django.conf import settings

from . import preprocessing


MODEL_CONFIGS = {
    # Replace this sample entry with your real model file and labels.
    'vehicle': {
        'title': 'Vehicle CNN',
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
    'vehicle-dense': {
        'title': 'Vehicle Dense',
        'path': settings.MODEL_STORAGE_DIR / 'vehicle_mobilenetv2_dynamic.tflite',
        'runtime': 'tflite',
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
        'title': 'Cat Dog',
        'path': settings.MODEL_STORAGE_DIR / 'cats_dogs.keras',
        'preprocess': preprocessing.dense_rgb_128_raw,
        'labels': ['cats', 'dogs'],
        'from_logits': True,
    },
    'cat-dog-dense': {
        'title': 'Cat Dog Dense',
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
