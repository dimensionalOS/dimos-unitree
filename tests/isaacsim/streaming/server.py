from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import jwt
import time
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

app = FastAPI()

# Add CORS middleware with more permissive settings for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load .env from streaming directory
load_dotenv(Path(__file__).parent / '.env')

# LiveKit configuration
LIVEKIT_HOST = 'http://localhost:7880'  # Internal connection to LiveKit
API_KEY = os.getenv('LIVEKIT_API_KEY')
API_SECRET = os.getenv('LIVEKIT_SECRET')

if not all([API_KEY, API_SECRET]):
    raise ValueError("API_KEY and API_SECRET must be set in .env file")

class WHIPRequest(BaseModel):
    sdp: str

def create_token(room_name: str, identity: str, is_publisher: bool) -> str:
    at = int(time.time())
    exp = at + 86400  # 24 hours
    
    grant = {
        "room": room_name,
        "identity": identity,
        "video": {"publish": is_publisher, "subscribe": True},
        "nbf": at,
        "exp": exp,
    }
    
    return jwt.encode(
        grant,
        API_SECRET,
        algorithm="HS256",
        headers={"kid": API_KEY}
    )

@app.post("/whip/{room_name}")
async def whip_endpoint(room_name: str, request: WHIPRequest):
    """Handle WHIP requests from the simulator"""
    print(f"[WHIP] Received request for room: {room_name}")
    print(f"[WHIP] SDP: {request.sdp[:100]}...")  # Print first 100 chars of SDP
    
    token = create_token(room_name, f"publisher_{int(time.time())}", True)
    print(f"[WHIP] Created token: {token[:30]}...")

    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            print("[WHIP] Creating room...")
            response = await client.post(
                f"{LIVEKIT_HOST}/room/create",
                json={"name": room_name},
                headers=headers
            )
            print(f"[WHIP] Room creation response: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[WHIP] Room creation failed: {response.text}")
                raise HTTPException(status_code=500, detail="Failed to create room")
            
            print("[WHIP] Connecting publisher...")
            response = await client.post(
                f"{LIVEKIT_HOST}/rtc/connect",
                json={"sdp": request.sdp},
                headers=headers
            )
            print(f"[WHIP] Publisher connection response: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[WHIP] Publisher connection failed: {response.text}")
                raise HTTPException(status_code=500, detail="Failed to connect")
                
            return response.json()
    except Exception as e:
        print(f"[WHIP] Error: {str(e)}")
        raise

@app.post("/watch/{room_name}")
async def watch_endpoint(room_name: str):
    """Handle viewer connections"""
    token = create_token(room_name, f"viewer_{int(time.time())}", False)
    
    return {
        "token": token,
        "ws_url": f"ws://18.189.249.222:7880?token={token}"  # Updated to use EC2 IP
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 