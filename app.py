from flask import session

from server import init_app

app = init_app()


@app.before_first_request
def clear_session_vars():
    session.clear()


if __name__ == "__main__":
    app.run()
