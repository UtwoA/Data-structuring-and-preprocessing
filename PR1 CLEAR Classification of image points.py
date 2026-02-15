import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image_paths = [
    'data/2026-02-13-00_00_2026-02-13-23_59_Sentinel-2_L2A_NDWI.jpg',
    'data/2026-02-13-00_00_2026-02-13-23_59_Sentinel-2_L2A_SWIR.jpg',
    'data/2026-02-13-00_00_2026-02-13-23_59_Sentinel-2_L2A_NDVI.jpg',
    'data/2026-02-13-00_00_2026-02-13-23_59_Sentinel-2_L2A_Highlight_Optimized_Natural_Color-2.jpg'
]


channels = []
for path in image_paths:
    img = Image.open(path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]
    
    channels.append(red)
    channels.append(green)
    channels.append(blue)


channels = np.array(channels)
height, width = channels.shape[1], channels.shape[2]

water = [
    (1041, 243, 1045, 247),
    (338, 689, 346, 697),
    (1088, 352, 1092, 358),
]

words = [
    (557, 14, 558, 19),
    (557, 68, 561, 69),
    (555, 117, 558, 118),
    (548, 814, 549, 823),
    (523, 746, 524, 750),
    (1125, 467, 1126, 480),
]

urban = [
    (978, 617, 989, 626),
    (997, 380, 1022, 400),
    (924, 657, 937, 670),
]

vegetation = [
    (369, 425, 392, 440),
    (316, 579, 336, 596),
    (384, 400, 416, 415),
    (14, 429, 27, 444,)
]

trees = [
    (4, 688, 17, 739),
    (667, 716, 676, 730),
    (951, 178, 964, 185),
    (1381, 377, 1409, 415),
]


all_interests = {
    'Вода': water,
    'Слова': words,
    'Застройка': urban,
    'Растительность': vegetation,
    'Деревья': trees
}

# Цвета для визуализации
class_colors = {
    'Вода': (0, 0, 255),
    'Слова': (0, 0, 0),
    'Застройка': (255, 255, 255),
    'Растительность': (0, 128, 0),
    'Деревья': (1, 50, 32)
}

class_names = ['Вода', 'Слова', 'Застройка', 'Растительность', 'Деревья']

class_params = {}

for class_name in class_names:
    all_pixels = []
    
    for interest in all_interests[class_name]:
        x1, y1, x2, y2 = interest
        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
        y1, y2 = int(min(y1, y2)), int(max(y1, y2))
        
        interest_pixels = channels[:, y1:y2, x1:x2]
        interest_pixels = interest_pixels.reshape(len(channels), -1).T 
        all_pixels.append(interest_pixels)

    all_pixels = np.vstack(all_pixels)
    mean_vector = np.mean(all_pixels, axis=0)
    
    # Вычисление матрицы ковариации
    centered = all_pixels - mean_vector
    N = all_pixels.shape[0]
    cov_matrix = (centered.T @ centered) / (N - 1)
    
    # Сохранение параметров класса
    class_params[class_name] = {
        'mean': mean_vector,
        'cov': cov_matrix,
        'num_pixels': len(all_pixels)
    }
    

    
classification_map = np.zeros((height, width), dtype=np.int32)

total_pixels = height * width
processed = 0

# Обход всех пикселей изображения
for y in range(height):
    for x in range(width):
        # Получение вектора признаков текущего пикселя (из всех каналов)
        pixel = channels[:, y, x]
        
        # Вычисление расстояния до каждого класса
        min_distance = float('inf')
        assigned_class = 0
        
        for class_id, class_name in enumerate(class_names):
            # Получение параметров класса
            mean = class_params[class_name]['mean']
            cov = class_params[class_name]['cov']
            
            # Добавление единичной матрицы E для устранения сингулярности
            E = np.eye(cov.shape[0])
            S_plus_E = cov + E
            
            # Вычисление обратной матрицы
            try:
                inv_matrix = np.linalg.inv(S_plus_E)
            except np.linalg.LinAlgError:
                inv_matrix = np.linalg.pinv(S_plus_E)
            
            # Разность векторов
            diff = pixel - mean
            
            # Расстояние Евклида-Махаланобиса
            # d_E-M(x, y) = sqrt((x - y)^T * (S + E)^(-1) * (x - y))
            distance = np.sqrt(diff.T @ inv_matrix @ diff)
            
            # Выбор класса с минимальным расстоянием
            if distance < min_distance:
                min_distance = distance
                assigned_class = class_id
        
        # Присвоение пикселя классу
        classification_map[y, x] = assigned_class
        
        # Вывод прогресса
        processed += 1
        if processed % 10000 == 0:
            progress = (processed / total_pixels) * 100
            print(f"   Обработано: {progress:.1f}%")

# Создание цветовой карты классификации
colored_map = np.zeros((height, width, 3))
for class_id, class_name in enumerate(class_names):
    color = np.array(class_colors[class_name]) / 255.0
    mask = classification_map == class_id
    colored_map[mask] = color

plt.figure(figsize=(16, 10))
plt.imshow(colored_map)
plt.axis('off')
plt.tight_layout()
plt.savefig('results/classification_result.png', dpi=300, bbox_inches='tight')
