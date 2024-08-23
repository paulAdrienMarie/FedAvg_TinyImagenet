from aiohttp import web
import os
from views import index, style, update_model, set_num_user
HERE = os.path.dirname(__file__)


def setup_routes(app):
    app.add_routes([
    web.static("/script", os.path.join(HERE,"static")),
    web.static("/artifacts", os.path.join(HERE,'artifacts')),
    web.static("/onnx", os.path.join(HERE,"onnx")),
    web.static("/dist", os.path.join(HERE,"dist"))
    ])
    app.router.add_get('/', index)
    app.router.add_get('/style.css', style)
    app.router.add_post('/update_model', update_model)
    app.router.add_post('/set_nb_users', set_num_user)

    