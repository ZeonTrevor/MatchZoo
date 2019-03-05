from flask import Flask, Response

app = Flask(__name__)


@app.route("/")
def index():
    return Response("It works!"), 200


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
