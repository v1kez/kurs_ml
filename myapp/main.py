import os
import numpy as np
from flask import Flask, render_template, url_for, request, send_from_directory
import tensorflow as tf
import torch
from PIL import Image

app = Flask(__name__)

menu = [
    {"name": "Классификация COVID", "url": 'covid_cnn'},
    {"name": "Детектирование объектов", "url": 'object_detection'},
]

# Путь к вашей модели CNN
model_path = os.path.join(os.path.dirname(__file__), 'model', 'covid_cnn.h5')

# Загрузка модели CNN
try:
    cnn_model = tf.keras.models.load_model(model_path)
except FileNotFoundError as e:
    print(f"Ошибка: {e}")
    cnn_model = None

# Путь к вашей модели YOLOv5
yolo_model_path = os.path.join(os.path.dirname(__file__), 'model', 'best.pt')

# Загрузка модели YOLOv5
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
except Exception as e:
    print(f"Ошибка при загрузке модели YOLOv5: {e}")
    yolo_model = None

@app.route("/")
def index():
    return render_template('index.html', title="X-RAY", menu=menu)

@app.route('/covid_cnn', methods=['GET', 'POST'])
def covid_cnn():
    if request.method == 'GET':
        return render_template('lab19.html', menu=menu, title="Нейронная сеть", post_url=url_for('covid_cnn'))

    elif request.method == 'POST':
        if cnn_model is None:
            return render_template('lab19.html', error="Модель не загружена. Проверьте путь к файлу.", menu=menu)

        file = request.files.get('image')
        img_height = 180
        img_width = 180

        if file and file.filename:
            class_names = ['covid', 'нормальный']
            filename = file.filename
            file_path = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)
            file.save(file_path)

            img = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = cnn_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]

            os.remove(file_path)

            return render_template('lab19.html', predicted_class=predicted_class, menu=menu, title="Нейронная сеть",
                                   percent=round(np.max(score) * 100, 1))
        else:
            return render_template('lab19.html', error="Пожалуйста, загрузите изображение.", menu=menu)


@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'GET':
        return render_template('object_detection.html', menu=menu, title="Детектирование объектов",
                               post_url=url_for('object_detection'))

    elif request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename:
            filename = file.filename
            file_path = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)
            file.save(file_path)

            # Используем YOLOv5 для детектирования объектов
            results = yolo_model(file_path)  # Результаты в формате YOLOv5

            # Сохраняем результат
            results.save()  # Сохранение изображений с аннотациями в папке 'runs/detect/exp'

            # Получаем список папок exp
            exp_folders = [d for d in os.listdir('runs/detect') if d.startswith('exp')]
            valid_exp_folders = []

            # Проверяем, что папки имеют правильный формат
            for folder in exp_folders:
                try:
                    exp_number = int(folder[3:])  # Преобразуем номер exp в целое число
                    valid_exp_folders.append((exp_number, folder))
                except ValueError:
                    continue  # Игнорируем папки с некорректными именами

            if valid_exp_folders:
                # Находим последнюю папку exp
                last_exp_folder = max(valid_exp_folders, key=lambda x: x[0])[1]

                # Логируем содержимое папки
                saved_files_path = os.path.join('runs', 'detect', last_exp_folder)
                saved_files = os.listdir(saved_files_path)
                print(f"Сохраненные файлы в {saved_files_path}: {saved_files}")  # Отладочная информация

                # Получаем имя сохраненного изображения (можно изменить логику, если нужно)
                annotated_image_name = filename  # Или используйте другое имя в зависимости от сохранения
                annotated_image_path = os.path.join(saved_files_path, annotated_image_name)

                # Проверяем, существует ли файл
                if os.path.exists(annotated_image_path):
                    annotated_image_url = url_for('send_runs_file',
                                                  filename=f'detect/{last_exp_folder}/{annotated_image_name}')
                else:
                    return render_template('object_detection.html',
                                           error=f"Изображение '{annotated_image_name}' не найдено.", menu=menu)
            else:
                return render_template('object_detection.html', error="Не удалось найти папки с результатами.",
                                       menu=menu)

            os.remove(file_path)  # Удаляем временный файл

            return render_template('object_detection.html', image_path=annotated_image_url, menu=menu,
                                   title="Детектирование объектов")
        else:
            return render_template('object_detection.html', error="Пожалуйста, загрузите изображение.", menu=menu)


@app.route('/runs/<path:filename>')
def send_runs_file(filename):
    return send_from_directory('runs', filename)

if __name__ == "__main__":
    app.run(debug=True)
