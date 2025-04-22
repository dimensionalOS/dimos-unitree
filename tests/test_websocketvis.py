#!/usr/bin/env python3

from dimos.web.websocket_vis.server import WebsocketVis


def main():
    # Start the WebSocket server
    websocket_vis = WebsocketVis(use_reload=True)
    websocket_vis.start()

    print(f"WebSocket server started on port {websocket_vis.port}")
    try:
        # Keep the server running
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping WebSocket server...")
        websocket_vis.stop()
        print("WebSocket server stopped")


if __name__ == "__main__":
    main()
