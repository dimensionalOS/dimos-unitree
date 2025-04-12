#!/usr/bin/env python3
import requests
import json
import asyncio
import websockets
from nano_llm import Plugin

# Server URLs
SERVER_URL = "http://192.168.100.221:8000/"
WEBSOCKET_URL = "ws://192.168.100.221:8000/ws"

class RobotDogControl(Plugin):
    """
    Plugin to control the robot dog via REST API and WebSocket.
    """
    def __init__(self, **kwargs):
        super().__init__(outputs=None, threaded=False, **kwargs)

        # Add available tools
        self.add_tool(self.run_action)
        self.add_tool(self.get_status)
        self.add_tool(self.ws_control)

        # Create a new event loop for WebSocket handling
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None

        # Start WebSocket connection in a background task
        self.loop.create_task(self.connect_websocket())

    async def connect_websocket(self):
        """
        Establishes a persistent WebSocket connection to the robot dog server.
        """
        while True:
            try:
                self.websocket = await websockets.connect(WEBSOCKET_URL)
                print("[INFO] WebSocket connected to robot dog server.")
                break
            except Exception as e:
                print(f"[ERROR] WebSocket connection failed: {e}. Retrying...")
                await asyncio.sleep(5)  # Retry after 5 seconds

    async def send_ws_command(self, command: dict):
        """
        Sends a command over the WebSocket connection and waits for a response.
        """
        if not self.websocket or self.websocket.closed:
            await self.connect_websocket()

        try:
            await self.websocket.send(json.dumps(command))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            print(f"[ERROR] WebSocket error: {e}")
            return {"status": "error", "message": str(e)}

    def run_action(self, action_name: str) -> str:
        """
        Executes a predefined action on the robot dog using the REST API.
        """
        response = requests.post(f"{SERVER_URL}/run_action/{action_name}")
        return response.json()

    def get_status(self) -> str:
        """
        Fetches the current status of the robot dog.
        """
        response = requests.get(f"{SERVER_URL}/status")
        return response.json()

    def ws_control(self, command: str) -> str:
        """
        Controls the robot dog using WebSocket commands.
        Supported commands: 'fw', 'bw', 'left', 'right', 'cw', 'ccw', 'stop'
        these are to move forward, backward, left, right, clockwise, counter clockwise and stop moving respectively. each command will continue to move until stopped
        """
        valid_commands = {"fw", "bw", "left", "right", "cw", "ccw", "stop"}

        if command not in valid_commands:
            return {"status": "error", "message": "Invalid command"}

        return self.loop.run_until_complete(self.send_ws_command({"command": command}))
