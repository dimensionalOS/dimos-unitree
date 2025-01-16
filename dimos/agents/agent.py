import base64
import os
import threading
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from reactivex import create, empty, Observable
from reactivex import operators as ops
from reactivex.disposable import CompositeDisposable, Disposable
import multiprocessing
from reactivex.scheduler import ThreadPoolScheduler

from dimos.agents.memory.base import AbstractAgentSemanticMemory
from dimos.agents.memory.chroma_impl import AgentSemanticMemory

# Initialize environment variables
load_dotenv()

# Scheduler for thread pool
optimal_thread_count = multiprocessing.cpu_count()
pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

class Agent:
    def __init__(self, dev_name:str="NA", agent_type:str="Base", agent_memory:AbstractAgentSemanticMemory=AgentSemanticMemory()):
        """
        Initializes a new instance of the Agent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent (e.g., 'Base', 'Vision').
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
        """
        self.dev_name = dev_name
        self.agent_type = agent_type
        self.agent_memory:AbstractAgentSemanticMemory = agent_memory  # TODO: Change to AbstractAgentMemory when we have those superclasses
        self.disposables = CompositeDisposable()

    def dispose_all(self):
        """
        Disposes of all active subscriptions managed by this agent.
        """
        if self.disposables:
            self.disposables.dispose()
        else:
            print("No disposables to dispose.")


class OpenAI_Agent(Agent):
    memory_file_lock = threading.Lock()

    def __init__(self, dev_name: str, agent_type:str="Vision", query="What do you see?", output_dir='/app/assets/agent', agent_memory:AbstractAgentSemanticMemory=AgentSemanticMemory()):
        """
        Initializes a new instance of the OpenAI_Agent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent, typically 'Vision' for this class.
            query (str): The default query to send along with images to OpenAI.
            output_dir (str): Directory where output files are stored.
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
        """
        super().__init__(dev_name, agent_type, agent_memory)
        self.client = OpenAI()
        self.query = query
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_processing = threading.Event()

    def _encode_image(self, image):
        """
        Encodes an image to JPEG format and converts it to a Base64 string.

        Args:
            image (numpy.ndarray): The image to encode.

        Returns:
            Observable: An Observable that emits the Base64 string of the encoded image.
        """
        return create(lambda observer, _: self._observable_encode(observer, image))

    def _observable_encode(self, observer, image):
        """
        Helper method to encode image and emit to an observer.

        Args:
            observer (Observer): The observer to emit to.
            image (numpy.ndarray): The image to encode.

        Raises:
            ValueError: If the image cannot be encoded.
        """
        try:
            _, buffer = cv2.imencode('.jpg', image)
            if buffer is None:
                observer.on_error(ValueError("Failed to encode image"))
            observer.on_next(base64.b64encode(buffer).decode('utf-8'))
            observer.on_completed()
        except Exception as e:
            observer.on_error(e)

    def _query_openai_with_image(self, base64_image):
        """
        Sends an encoded image to OpenAI and gets a response.

        Args:
            base64_image (str): The Base64-encoded image to send.

        Returns:
            Observable: An Observable that emits the response from OpenAI.
        """
        return create(lambda observer, _: self._observable_query(observer, base64_image))

    def _observable_query(self, observer, base64_image):
        """
        Helper method to query OpenAI with an encoded image and emit to an observer.

        Args:
            observer (Observer): The observer to emit to.
            base64_image (str): The Base64-encoded image to send.

        Raises:
            Exception: If the query to OpenAI fails.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "text", "text": self.query}, 
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}]}],
                max_tokens=300
            )
            observer.on_next(response.choices[0].message.content)
            observer.on_completed()
        except Exception as e:
            observer.on_error(e)

    def _process_if_idle(self, image):
        """
        Processes an image if the agent is idle.

        Args:
            image (numpy.ndarray): The image to process.

        Returns:
            Observable: An Observable that handles the processing of the image.
        """
        if not self.is_processing.is_set():
            print("Processing Frame.")
            self.is_processing.set()
            return self._encode_image(image).pipe(
                ops.flat_map(self._query_openai_with_image),
                ops.do_action(on_next=lambda _: None, on_completed=self._reset_processing_flag)
            )
        else:
            print("Skipping Frame.")
            return empty()
    
    def _reset_processing_flag(self):
        """
        Resets the processing flag to allow new image processing.
        """
        self.is_processing.clear()

    def _process_image_stream(self, image_stream):
        """
        Processes a stream of images.

        Args:
            image_stream (Observable): The stream of images to process.

        Returns:
            Observable: An Observable that processes each image in the stream.
        """
        return image_stream.pipe(
            ops.observe_on(pool_scheduler),
            ops.flat_map(self._process_if_idle),
            ops.filter(lambda x: x is not None)
        )

    def subscribe_to_image_processing(
        self, 
        frame_observable: Observable
    ) -> Disposable:
        """Subscribes to and processes a stream of video frames.

        Sets up a subscription to process incoming video frames through OpenAI's
        vision model. Each frame is processed only when the agent is idle to prevent
        overwhelming the API. Responses are logged to a file and the subscription
        is tracked for cleanup.

        Args:
            frame_observable: An Observable emitting video frames.
                Each frame should be a numpy array in BGR format with shape
                (height, width, 3).

        Returns:
            A Disposable representing the subscription. Can be used for external
            resource management while still being tracked internally.

        Raises:
            TypeError: If frame_observable is not an Observable.
            ValueError: If frames have invalid format or dimensions.

        Example:
            >>> agent = OpenAI_Agent("camera_1")
            >>> disposable = agent.subscribe_to_image_processing(frame_stream)
            >>> # Later cleanup
            >>> disposable.dispose()

        Note:
            The subscription is automatically added to the agent's internal
            CompositeDisposable for cleanup. The returned Disposable provides
            additional control if needed.
        """
        disposable = self._process_image_stream(frame_observable).subscribe(
            on_next=self._log_response_to_file,
            on_error=lambda e: print(f"Error in {self.dev_name}: {e}"),
            on_completed=lambda: print(f"Stream processing completed for {self.dev_name}")
        )
        self.disposables.add(disposable)
        return disposable
    
    def _log_response_to_file(self, response):
        """
        Logs the response from OpenAI to a file.

        Args:
            response (str): The response from OpenAI to log.
        """
        with self.memory_file_lock:
            with open(os.path.join(self.output_dir, 'memory.txt'), 'a') as file:
                file.write(f"{self.dev_name}: {response}\n")
                print(f"OpenAI Response [{self.dev_name}]:", response)
