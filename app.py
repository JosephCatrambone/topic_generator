from flask import Flask, jsonify, request, render_template

from modeling import predict

app = Flask(__name__)


# Serve the site index:
@app.route('/')
def index():
	return app.send_static_file("index.html")


@app.route('/api/v1/', methods=['POST'])
def make_topics():
	# Check for appropriate request type:
	json_headers = ("Content-Type" in request.headers and request.headers["Content-Type"] == "application/json") \
		or request.content_type == "application/json"
	if not json_headers:
		return 'Request Content-Type must be application/json with a "text" field in the body.', 400

	# Check for appropriate request schema:
	request_data = request.json
	if "text" not in request_data:
		return "Request must contain 'text' key.", 400

	# Check for appropriate contents:
	request_text = request_data["text"]
	if not request_text:
		return "Request contained an empty 'text' field.", 400

	return jsonify({
		"input_text": request_text,
		"keywords": predict([request_text,]),
	})


if __name__ == '__main__':
	app.run()
