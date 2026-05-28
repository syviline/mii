from functools import lru_cache

from django.core.exceptions import ValidationError

from .ml_config import get_model_config


@lru_cache(maxsize=None)
def load_model(model_key):
    config = get_model_config(model_key)
    model_path = config['path']

    if not model_path.exists():
        raise FileNotFoundError(
            f'Файл модели не найден: {model_path}. '
            'Положите .keras файл в папку ml_models или поправьте путь в predictor/ml_config.py.'
        )

    from tensorflow import keras

    return keras.models.load_model(model_path)


def predict_image(model_key, image_file):
    config = get_model_config(model_key)
    model = load_model(model_key)
    batch = config['preprocess'](image_file)
    predictions = model.predict(batch)

    if len(predictions) == 0:
        raise ValidationError('Модель вернула пустой результат.')

    scores = predictions[0]
    if config.get('from_logits'):
        import numpy as np

        exp_scores = np.exp(scores - np.max(scores))
        scores = exp_scores / exp_scores.sum()

    best_index = int(scores.argmax())
    labels = config.get('labels') or []
    label = labels[best_index] if best_index < len(labels) else f'class_{best_index}'
    confidence = float(scores[best_index])
    score_items = [
        {
            'label': labels[index] if index < len(labels) else f'class_{index}',
            'confidence': float(score),
            'confidence_percent': float(score) * 100,
        }
        for index, score in enumerate(scores)
    ]
    top_scores = sorted(
        score_items,
        key=lambda item: item['confidence'],
        reverse=True,
    )[:5]

    return {
        'label': label,
        'confidence': confidence,
        'confidence_percent': confidence * 100,
        'scores': score_items,
        'top_scores': top_scores,
    }
