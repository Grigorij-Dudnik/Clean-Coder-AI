from flask import Flask

app = Flask(__name__)


@app.get("/")
async def home():
    return "Welcome to the Flask application!"


@app.get("/hello/<name>")
async def hello_name(name):
    return "Hello " + name


if __name__ == "__main__":
    app.run(debug=True)
