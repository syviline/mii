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
            'Положите файл модели в папку ml_models или поправьте путь в predictor/ml_config.py.'
        )

    if config.get('runtime') == 'tflite':
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter

    from tensorflow import keras
    return keras.models.load_model(model_path)


def predict_image(model_key, image_file):
    import numpy as np

    config = get_model_config(model_key)
    model = load_model(model_key)
    batch = config['preprocess'](image_file)

    if config.get('runtime') == 'tflite':
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], batch.astype(np.float32))
        model.invoke()
        scores = model.get_tensor(output_details[0]['index'])[0]
    else:
        predictions = model.predict(batch)
        if len(predictions) == 0:
            raise ValidationError('Модель вернула пустой результат.')
        scores = predictions[0]

    if config.get('from_logits'):
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
