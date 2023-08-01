import requests

if __name__ == '__main__':
    # features = {'features': [12, 3.1, 0.9, 8]}
    features = [3.5, 4, 7, 2.8]

    r = requests.post('http://localhost:5000/predict', json=features)
    print(r.status_code)
    # реализуем обработку результата
    if r.status_code == 200:
        # если запрос выполнен успешно (код обработки=200),
        # выводим результат на экран
        print(r.json()['prediction'])
    else:
        # если запрос завершён с кодом, отличным от 200,
        # выводим содержимое ответа
        print(r.text)