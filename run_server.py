import sys
sys.path.insert(0, r'C:\Games\Indore\Indore')
from server.app.main import app
import uvicorn
uvicorn.run(app, host='127.0.0.1', port=8000)