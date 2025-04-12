#!/usr/bin/env python3
#
# This multimodal example is a simplified version of the 'Live Llava' demo,
# wherein the same prompt (or set of prompts) is applied to a stream of images.
#
# You can run it like this (these options will replicate the defaults)
#
#    python3 -m nano_llm.vision.example \
#      --model Efficient-Large-Model/VILA1.5-3b \
#      --video-input "/data/images/*.jpg" \
#      --prompt "Describe the image." \
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


class VLLM():
    def __init__(self, model="Efficient-Large-Model/VILA1.5-3b", 
                    video_input, 
                    prompts=["Describe the image.", "Are there people in the image?"],
                    api=None,
                    vision_model=None,
                    vision_scaling=None,
                    max_context_len=None,
                    quantization=None,
                    system_prompt=None,
                    chat_template=None,
                    max_new_tokens=128,
                    min_new_tokens=-1,
                    do_sample=True,
                    repetition_penalty=1,
                    top_p=0.95,
                    temperature=0.7
                    ):
        
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.min_new_tokens = min_new_tokens
        self.prompts = prompts
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.quantization = quantization
        self.system_prompt = system_prompt
        self.model = model
        self.api = api
        self.vision_model = vision_model
        self.vision_scaling = vision_scaling
        self.max_context_len = max_context_len
        self.chat_template = chat_template

        # load vision/language model
        self.model = NanoLLM.from_pretrained(
            self.model, 
            api=self.api,
            quantization=self.quantization, 
            max_context_len=self.max_context_len,
            vision_model=self.vision_model,
            vision_scaling=self.vision_scaling)

        assert(model.has_vision)

        self.success = True

        # create the chat history
        self.chat_history = ChatHistory(model, self.chat_template, self.system_prompt)

    def run(self, video):
        # video_source = VideoSource(**vars(args), cuda_stream=0, return_copy=False)
        self.video_source = cv2.VideoCapture(video)

        while self.success:
            self.success, img = self.video_source.read()

            self.chat_history.append('user', image=img)
            time_begin = time.perf_counter()
    
            for prompt in self.prompts:
                self.chat_history.append('user', prompt, use_cache=True)
                embedding, _ = self.chat_history.embed_chat()
        
                print('>>', prompt)
            
                reply = model.generate(
                    embedding,
                    kv_cache=self.chat_history.kv_cache,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    do_sample=self.do_sample,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    )
        
                for token in reply:
                    termcolor.cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)

                chat_history.append('bot', reply)
      
            time_elapsed = time.perf_counter() - time_begin
            print(f"time:  {time_elapsed*1000:.2f} ms  rate:  {1.0/time_elapsed:.2f} FPS")
            
            chat_history.reset()
            
            if video_source.eos:
                break
        self.success = True