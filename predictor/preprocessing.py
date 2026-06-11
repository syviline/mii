import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image_vehicle(image_file):
    """
    Предобработка изображения для модели MobileNetV2 (224x224).
    Использует стандартную функцию preprocess_input из Keras.
    """
    # 1. Загрузка
    image_file.seek(0)
    img = Image.open(BytesIO(image_file.read())).convert('RGB')
    
    # 2. Изменение размера (как в переменной IMG_SIZE)
    img = img.resize((224, 224))
    
    # 3. Превращение в массив
    img_array = np.array(img).astype('float32')
    
    # 4. Добавление размерности батча (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Специфичная для MobileNetV2 нормализация (входные данные должны быть [-1, 1])
    # В ноутбуке использовалась функция preprocess_input
    # img_preprocessed = preprocess_input(img_array)
    
    return img_preprocessed


def dense_rgb_128_raw(image_file):
    """
    Для модели, обученной через image_dataset_from_directory(image_size=(128, 128))
    и содержащей layers.Rescaling(1./255) внутри самой модели.
    """
    import numpy as np
    from PIL import Image

    image_file.seek(0)
    image = Image.open(image_file).convert('RGB')
    image = image.resize((128, 128))
    array = np.asarray(image, dtype='float32')
    return np.expand_dims(array, axis=0)