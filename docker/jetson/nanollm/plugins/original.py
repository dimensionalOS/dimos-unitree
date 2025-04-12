#!/usr/bin/env python3
#gst-launch-1.0 -v videotestsrc ! video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! x264enc ! rtph264pay pt=96 ! udpsink host=192.168.11.101 port=5600
#gst-launch-1.0 udpsrc port=5600 ! application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! autovideosink
#gst-launch-1.0 -v udpsrc address=192.168.2.1 port=5600 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! queue ! rtph264depay ! queue ! rtph264pay pt=96 ! queue ! udpsink host=192.168.11.101 port=5600


#python3 original.py --model Efficient-Large-Model/VILA1.5-3b --prompt "Describe the image." --video-input rtsp://admin:admin@192.168.11.60:554/2

# This multimodal example is a simplified version of the 'Live Llava' demo,
# wherein the same prompt (or set of prompts) is applied to a stream of images.
#
# You can run it like this (these options will replicate the defaults)
#
#    python3 -m nano_llm.vision.example \
#      --model Efficient-Large-Model/VILA1.5-3b \
#      --video-input "/data/images/*.jpg" \
#      --prompt "Describe the image."--prompt "Describe the image." \
#      --prompt "Are there people in the image?"
#
# You can specify multiple prompts (or a text file) to be applied to each image,
# and the video inputs can be sequences of files, camera devices, or network streams.
#
# For example, `--video-input /dev/video0` will capture from a V4L2 webcam. See here:
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
#
import time
import termcolor

from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
from nano_llm.plugins import VideoSource

from jetson_utils import cudaMemcpy, cudaToNumpy


# parse args and set some defaults
args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
prompts = load_prompts(args.prompt)

if not prompts:
  prompts = ["Describe the image.", "Are there people in the image?"]
    
if not args.model:
  args.model = "Efficient-Large-Model/VILA1.5-3b"

if not args.video_input:
  success = False
else:
  success = True
    
print("HEERERERERERERRE")
print(args)

# load vision/language model
model = NanoLLM.from_pretrained(
    args.model, 
    api=args.api,
    quantization=args.quantization, 
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

assert(model.has_vision)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# open the video stream
video_source = VideoSource(**vars(args), cuda_stream=0, return_copy=False)

# apply the prompts to each frame
while success:
  img = video_source.capture()
  
  if img is None:
      continue

  chat_history.append('user', image=img)
  time_begin = time.perf_counter()
  
  for prompt in prompts:
      chat_history.append('user', prompt, use_cache=True)
      embedding, _ = chat_history.embed_chat()
      
      print('>>', prompt)
      
      reply = model.generate(
          embedding,
          kv_cache=chat_history.kv_cache,
          max_new_tokens=args.max_new_tokens,
          min_new_tokens=args.min_new_tokens,
          do_sample=args.do_sample,
          repetition_penalty=args.repetition_penalty,
          temperature=args.temperature,
          top_p=args.top_p,
      )
      
      for token in reply:
          termcolor.cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)

      chat_history.append('bot', reply)
    
  time_elapsed = time.perf_counter() - time_begin
  print(f"time:  {time_elapsed*1000:.2f} ms  rate:  {1.0/time_elapsed:.2f} FPS")
  
  chat_history.reset()
  
  if video_source.eos:
      break