import socketio
import uvicorn
import threading
import os
import sys
from starlette.routing import Route
from starlette.responses import HTMLResponse
from starlette.applications import Starlette


async def serve_index(request):
    # Read the index.html file directly
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(index_path, "r") as f:
        content = f.read()
    return HTMLResponse(content)


# Create global socketio server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Register Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def message(sid, data):
    print(f"Message received from {sid}: {data}")
    await sio.emit('message', {'response': 'Server received your message'}, room=sid)

# Create Starlette app with route for root path
routes = [Route("/", serve_index)]
starlette_app = Starlette(routes=routes)

# Create socketio app with the Starlette app
# Add a mount point for static files
from starlette.staticfiles import StaticFiles

static_dir = os.path.join(os.path.dirname(__file__), "static")
starlette_app.mount("/", StaticFiles(directory=static_dir), name="static")

# Create the ASGI app
app = socketio.ASGIApp(sio, starlette_app)


class WebsocketVis:
    def __init__(self, port=7778, use_reload=False):
        self.port = port
        self.server = None
        self.server_thread = None
        self.sio = sio  # Use the global sio instance
        self.use_reload = use_reload

    def start(self):
        # If reload is requested, run in main thread
        if self.use_reload:
            print("Starting server with hot reload in main thread")
            uvicorn.run(
                "server:app",  # Use import string for reload to work
                host="0.0.0.0",
                port=self.port,
                reload=True,
                reload_dirs=[os.path.dirname(__file__)],
            )
            return self

        # Otherwise, run in background thread
        else:
            print("Starting server in background thread")
            self.server_thread = threading.Thread(
                target=uvicorn.run,
                kwargs={
                    "app": app,  # Use direct app object for thread mode
                    "host": "0.0.0.0",
                    "port": self.port,
                },
                daemon=True,
            )
            self.server_thread.start()
            return self

    def stop(self):
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join()
        self.sio.disconnect()


# For direct execution with uvicorn CLI
if __name__ == "__main__":
    # Check if --reload flag is passed
    use_reload = "--reload" in sys.argv
    server = WebsocketVis(port=7778, use_reload=use_reload)
    server.start()
