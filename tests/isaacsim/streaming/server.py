from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import jwt
import time
import os
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LiveKit configuration
LIVEKIT_HOST = "http://localhost:7880"
API_KEY = "your-api-key"
API_SECRET = "your-api-secret"

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
    token = create_token(room_name, f"publisher_{int(time.time())}", True)
    
    async with httpx.AsyncClient() as client:
        # Create or join room
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(
            f"{LIVEKIT_HOST}/room/create",
            json={"name": room_name},
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to create room")
        
        # Connect as publisher
        response = await client.post(
            f"{LIVEKIT_HOST}/rtc/connect",
            json={"sdp": request.sdp},
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to connect")
            
        return response.json()

@app.post("/watch/{room_name}")
async def watch_endpoint(room_name: str):
    """Handle viewer connections"""
    token = create_token(room_name, f"viewer_{int(time.time())}", False)
    
    return {
        "token": token,
        "ws_url": f"ws://localhost:7880?token={token}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 