from flask import Flask, jsonify, request

from modeling import predict

app = Flask(__name__)


@app.route('/')
def index():
	return 'Ohai!'


@app.route('/api/v1/', methods=['POST'])
def make_topics():
	json_headers = ("Content-Type" in request.headers and request.headers["Content-Type"] == "application/json") \
		or request.content_type == "application/json"
	if not json_headers:
		return 400, 'Request Content-Type must be application/json with a "text" field in the body.'
	request_data = request.json
	if "text" not in request_data:
		return 400, "Request must contain 'text' key."
	request_text = request_data["text"]
	return jsonify({
		"input_text": request_text,
		"keywords": predict([request_text,]),
	})


if __name__ == '__main__':
	app.run()
