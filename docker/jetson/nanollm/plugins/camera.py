#!/usr/bin/env python3
from nano_llm import bot_function
from nano_llm.web import WebServer


@bot_function
def Camera():
    """
    returns camera ip
    """
    return "rtsp://admin:admin@192.168.11.60:554/2"