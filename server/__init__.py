from flask import Flask

# from flask_sqlalchemy import SQLAlchemy

# db = SQLAlchemy()

# def init_test_app():
#     app = Flask(__name__)
#     app.config['TESTING'] = True
#     app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///dbtest.db"
#     # Dynamically bind SQLAlchemy to application
#     db.init_app(app)
#     app.app_context().push()  # this does the binding
#     return app


def init_app():
    app = Flask(__name__, instance_relative_config=False)

    app.config.from_object('server.config.Config')

    # db.init_app(app)
    # with app.app_context():

    #
    #   DROP TABLES
    #

    #
    #   CREATE ALL TABLES
    #

    # db.create_all()

    #
    #   CREATE TABLES
    #

    from server.routes.audio import bp as audio_bp
    app.register_blueprint(audio_bp)

    return app
