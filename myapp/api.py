import os
import requests

url = 'http://127.0.0.1:5000/api/covid_cnn'
# image_path = 'C:/Users/bordy/PycharmProjects/kurs_ml/myapp/uploads/02.jpeg'
image_path = os.path.join(os.path.dirname(__file__), 'uploads', 'Newitems_node_203_Large.png')


# Проверьте, существует ли файл
if not os.path.isfile(image_path):
    print(f"Файл не найден: {image_path}")
else:
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}

            # Отправка POST-запроса
            response = requests.post(url, files=files)

            # Проверка статуса ответа
            print("Status Code:", response.status_code)  # Выводим статус код

            if response.status_code == 200:
                try:
                    # Пытаемся распечатать JSON-ответ
                    json_response = response.json()
                    print("Ответ от сервера:", json_response)  # Выводим считанный JSON
                except ValueError:
                    print("Ошибка: Невозможно декодировать ответ как JSON.")
                    print("Текст ответа:", response.text)  # Выводим текст ответа для отладки
            else:
                print("Ошибка! Код ответа:", response.status_code)
                print("Текст ответа:", response.text)  # Выводим текст ответа для отладки
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")