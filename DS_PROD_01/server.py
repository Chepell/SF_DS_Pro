from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

@app.route('/hi')
def hello_func():
    name = request.args.get('name')
    return f"HELLO {name}!!!"
@app.route('/')
def index():
    return 'Test message. The server is running'

@app.route('/time')
def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route('/add', methods=['POST'])
def add():
    # Параметры тела доступны в поле data.
    # Но если тело — это JSON-строка, то можно использовать поле json.
    num = request.json.get('num')
    if num > 10:
        return 'too much', 400
    return jsonify({'result': num + 1})


if __name__ == '__main__':
    app.run('localhost', 5000)
