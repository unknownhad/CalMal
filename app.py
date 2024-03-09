#!/usr/bin/env python3
import os
from flask_restful import Api
from flask import Flask, render_template
import routes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('.', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
api = Api()
routes.add_routes_to_resource(api)
api.init_app(app)


@app.route('/')
def RenderMainIndex():
    return render_template('register.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=1234)
