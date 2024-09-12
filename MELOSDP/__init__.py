from flask import Flask
from flask_cors import CORS
from MELOSDP.util import APIResponse

app = Flask('flask_MELOSDP', template_folder="../templates", static_folder="../static")

# 解决跨域问题，允许所有域名访问
CORS(app, resources={r'/*': {'origins': '*'}})
