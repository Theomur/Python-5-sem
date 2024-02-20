import time

from PIL import Image

# старт таймера
start_time = time.time()

# Открытие изображения
img = Image.open("lena.png")

# Преобразование изображения в массив пикселей
pixels = img.load()

# Создание словаря для подсчета цветов
color_count = {}

# Проход по всем пикселям изображения
for y in range(img.height):
    for x in range(img.width):
        # Получение цвета пикселя
        color = pixels[x, y]
        # Добавление цвета в словарь или увеличение счетчика
        color_count[color] = color_count.get(color, 0) + 1

# конец таймера + вычисление времени
end_time = time.time()
calculated_time = end_time - start_time

# Вывод количества уникальных цветов и времени выполнения
print(f"Время выполнения: {calculated_time} секунд")
print(f"Уникальных цветов: {len(color_count)}")
