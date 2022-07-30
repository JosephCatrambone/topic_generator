from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
	return 'Ohai!'


@app.route('/api/v1/', method=['POST'])
def make_topics():
	return 'Hello World!'


if __name__ == '__main__':
	app.run()
