import asyncio
import json
import logging
import struct
import sys
import threading

from reactivex import create
from .msgs.pub_sub import WebRTCDataChannelPubSub
from .lidar.lidar_decoder import LidarDecoder
from .msgs.heartbeat import WebRTCDataChannelHeartBeat
from .msgs.validation import WebRTCDataChannelValidaton
from .msgs.rtc_inner_req import WebRTCDataChannelRTCInnerReq
from .util import print_status
from .msgs.error_handler import handle_error

from .constants import DATA_CHANNEL_TYPE, RTC_TOPIC, SPORT_CMD

decoder = LidarDecoder()

class WebRTCDataChannel:
    def __init__(self, conn, pc) -> None:
        self.channel = pc.createDataChannel("data")
        self.data_channel_opened = False
        self.conn = conn

        self.pub_sub = WebRTCDataChannelPubSub(self.channel)

        self.heartbeat = WebRTCDataChannelHeartBeat(self.channel, self.pub_sub)
        self.validaton = WebRTCDataChannelValidaton(self.channel, self.pub_sub)
        self.rtc_inner_req = WebRTCDataChannelRTCInnerReq(self.conn, self.channel, self.pub_sub)

        #Event handler for Validation succeed
        def on_validate():
            self.data_channel_opened = True
            self.heartbeat.start_heartbeat()
            self.rtc_inner_req.network_status.start_network_status_fetch()
            print_status("Data Channel Verification", "✅ OK")
            

        self.validaton.set_on_validate_callback(on_validate)

        #Event handler for Network status Update
        def on_network_status(mode):
            print(f"Go2 connection mode: {mode}")

        self.rtc_inner_req.network_status.set_on_network_status_callback(on_network_status)

        # Event handler for data channel open
        @self.channel.on("open")
        def on_open():
            logging.info("Data channel opened")

        # Event handler for data channel close
        @self.channel.on("close")
        def on_close():
            logging.info("Data channel closed")
            self.data_channel_opened = False
            self.heartbeat.stop_heartbeat()
            self.rtc_inner_req.network_status.stop_network_status_fetch()
            
        # Event handler for data channel messages
        @self.channel.on("message")
        async def on_message(message):
            logging.info("Received message on data channel: %s", message)
            try:
            
                # Check if the message is not empty
                if not message:
                    return

                # Determine how to parse the 'data' field
                if isinstance(message, str):
                    parsed_data = json.loads(message)
                elif isinstance(message, bytes):
                    parsed_data = WebRTCDataChannel.deal_array_buffer(message)
                
                # Resolve any pending futures or callbacks associated with this message
                self.pub_sub.run_resolve(parsed_data)

                # Handle the response
                await self.handle_response(parsed_data)
        
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON message: %s", message, exc_info=True)
            except Exception as error:
                logging.error("Error processing WebRTC data", exc_info=True)


    async def handle_response(self, msg: dict):
        msg_type = msg["type"]

        if msg_type == DATA_CHANNEL_TYPE["VALIDATION"]:
            await self.validaton.handle_response(msg)
        elif msg_type == DATA_CHANNEL_TYPE["RTC_INNER_REQ"]:
            self.rtc_inner_req.handle_response(msg)
        elif msg_type == DATA_CHANNEL_TYPE["HEARTBEAT"]:
            self.heartbeat.handle_response(msg)
        elif msg_type in {DATA_CHANNEL_TYPE["ERRORS"], DATA_CHANNEL_TYPE["ADD_ERROR"], DATA_CHANNEL_TYPE["RM_ERROR"]}:
            handle_error(msg)
        elif msg_type == DATA_CHANNEL_TYPE["ERR"]:
            await self.validaton.handle_err_response(msg)
        

    async def wait_datachannel_open(self, timeout=5):
        """Waits for the data channel to open asynchronously."""
        try:
            await asyncio.wait_for(self._wait_for_open(), timeout)
        except asyncio.TimeoutError:
            print("Data channel did not open in time")
            sys.exit(1)

    async def _wait_for_open(self):
        """Internal function that waits for the data channel to be opened."""
        while not self.data_channel_opened:
            await asyncio.sleep(0.1)
    
    @staticmethod
    def deal_array_buffer(buffer):
        header_1, header_2 = struct.unpack_from('<HH', buffer, 0)
        if header_1 == 2 and header_2 == 0:
            return WebRTCDataChannel.deal_array_buffer_for_lidar(buffer[4:])
        else:
            return WebRTCDataChannel.deal_array_buffer_for_normal(buffer)
    @staticmethod
    def deal_array_buffer_for_normal(buffer):
        header_length, = struct.unpack_from('<H', buffer, 0)
        json_data = buffer[4:4 + header_length]
        binary_data = buffer[4 + header_length:]

        decoded_json = json.loads(json_data.decode('utf-8'))

        decoded_data = decoder.decode(binary_data, decoded_json['data'])

        decoded_json['data']['data'] = decoded_data
        return decoded_json
    @staticmethod
    def deal_array_buffer_for_lidar(buffer):
        header_length, = struct.unpack_from('<I', buffer, 0)
        json_data = buffer[8:8 + header_length]
        binary_data = buffer[8 + header_length:]

        decoded_json = json.loads(json_data.decode('utf-8'))

        decoded_data = decoder.decode(binary_data, decoded_json['data'])

        decoded_json['data']['data'] = decoded_data
        return decoded_json

    
    #Should turn it on when subscribed to ulidar topic
    async def disableTrafficSaving(self, switch: bool):
        data = {
            "req_type": "disable_traffic_saving",
            "instruction": "on" if switch else "off"
        }
        response = await self.pub_sub.publish(
            "",
            data,
            DATA_CHANNEL_TYPE["RTC_INNER_REQ"],
        )
        if response['info']['execution'] == "ok":
            print(f"DisableTrafficSavings: {data['instruction']}")
            return True
        return False

    
    #Enable/Disable video channel
    def switchVideoChannel(self, switch: bool):
        self.pub_sub.publish_without_callback(
            "",
            "on" if switch else "off",
            DATA_CHANNEL_TYPE["VID"],
        )
        print(f"Video channel: {'on' if switch else 'off'}")
    

    #Enable/Disable audio channel
    def switchAudioChannel(self, switch: bool):
        self.pub_sub.publish_without_callback(
            "",
            "on" if switch else "off",
            DATA_CHANNEL_TYPE["AUD"],
        )
        print(f"Audio channel: {'on' if switch else 'off'}")
    

    async def switchToNormalMode(self):
        # Get the current motion_switcher status
        response = await self.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], 
            {"api_id": 1001}
        )
        if response['data']['header']['status']['code'] == 0:
            data = json.loads(response['data']['data'])
            current_motion_switcher_mode = data['name']
            print(f"Current motion mode: {current_motion_switcher_mode}")

        # Switch to "normal" mode if not already
        if current_motion_switcher_mode != "normal":
            print(f"Switching motion mode from {current_motion_switcher_mode} to 'normal'...")
            await self.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], 
                {
                    "api_id": 1002,
                    "parameter": {"name": "normal"}
                }
            )
            await asyncio.sleep(5)  # Wait while it stands up

    # Wiggle the robot
    def sendWiggle(self):
        print(f"Trying to Wiggle.")

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(self.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {"api_id": SPORT_CMD["WiggleHips"]}
            ))

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        print(f"Done trying to Wiggle.")
        return

    # Stretch the robot
    def sendStretch(self):
        def on_subscribe(observer, scheduler):
            print(f"Trying to Stretch.")
            observer.on_next(None)
            observer.on_completed()

        observable = create(on_subscribe)

        observable.subscribe(
            on_next=lambda _: self.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {"api_id": SPORT_CMD["Stretch"]}
            ),
            on_error=lambda e: print(f"Error: {e}"),
            on_completed=lambda: print("Stretch command sent.")
        )

    async def sendDamp(self):
        await self.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], 
            {"api_id": SPORT_CMD["Damp"]}
        )

    async def sendStandUp(self):
        await self.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], 
            {"api_id": SPORT_CMD["StandUp"]}
        )

    def sendStandUp(self):
        print(f"Trying to Stand Up.")

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(self.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], 
                {"api_id": SPORT_CMD["StandUp"]}
            ))

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        print(f"Done trying to Stand Up.")
        return
    
    async def sendSit(self):
        await self.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], 
            {"api_id": SPORT_CMD["Sit"]}
        )

