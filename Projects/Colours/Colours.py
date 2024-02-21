import cv2
import numpy as np
import time

# Открытие изображения
img = cv2.imread('lena2.png')

# Преобразование изображения в формат BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# старт таймера
start_time = time.time()

# Преобразование изображения в массив numpy
pixels = np.array(img)

# Подсчет уникальных цветов
unique_colors, counts = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0, return_counts=True)

# конец таймера + вычисление времени
end_time = time.time()
calculated_time = end_time - start_time

# Вывод количества уникальных цветов и времени выполнения
print(f"Время выполнения: {calculated_time} секунд")
print(f"Уникальных цветов: {len(unique_colors)}")
