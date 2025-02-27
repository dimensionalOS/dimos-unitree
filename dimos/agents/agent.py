from datetime import datetime
import json
import os
import threading
from typing import Tuple
from dotenv import load_dotenv
import numpy as np
from openai import NOT_GIVEN, OpenAI
from reactivex import Observer, create, Observable, operators as ops
from reactivex.disposable import CompositeDisposable, Disposable
from reactivex.scheduler import ThreadPoolScheduler

from dimos.agents.memory.base import AbstractAgentSemanticMemory
from dimos.agents.memory.chroma_impl import AgentSemanticMemory
from dimos.agents.prompt_builder.impl import PromptBuilder
from dimos.agents.tokenizer.openai_impl import AbstractTokenizer, OpenAI_Tokenizer
from pydantic import BaseModel

from dimos.robot.skills import AbstractSkill
from dimos.stream.frame_processor import FrameProcessor
from dimos.stream.video_operators import Operators, VideoOperators

# Initialize environment variables
load_dotenv()

AGENT_PRINT_COLOR = "\033[33m"
AGENT_RESET_COLOR = "\033[0m"

# region Agent
class Agent:
    def __init__(self, dev_name: str = "NA", agent_type: str = "Base", agent_memory: AbstractAgentSemanticMemory = None):
        """
        Initializes a new instance of the Agent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent (e.g., 'Base', 'Vision'). Currently unused.
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
        """
        self.dev_name = dev_name
        self.agent_type = agent_type
        self.agent_memory = agent_memory or AgentSemanticMemory()
        self.disposables = CompositeDisposable()

    def dispose_all(self):
        """
        Disposes of all active subscriptions managed by this agent.
        """
        if self.disposables:
            self.disposables.dispose()
        else:
            print(f"{AGENT_PRINT_COLOR}No disposables to dispose.{AGENT_RESET_COLOR}")

# endregion Agent



class LLMAgent(Agent):
    logging_file_memory_lock = threading.Lock()

    def __init__(self, dev_name: str = "NA", agent_type: str = "Base", agent_memory: AbstractAgentSemanticMemory = None):
        super().__init__(dev_name, agent_type, agent_memory or AgentSemanticMemory())


    # def __init__(self, 
    #             dev_name: str, 
    #             agent_type: str = "LLM",
    #             query: str = "What do you see?", 
    #             input_video_stream: Observable = None,
    #             output_dir: str = '/app/assets/agent', 
    #             agent_memory: AbstractAgentSemanticMemory = None,
    #             system_query: str = None, 
    #             system_query_without_documents: str = None,
    #             max_input_tokens_per_request: int = 128000,
    #             max_output_tokens_per_request: int = 16384,
    #             model_name: str = "gpt-4o",
    #             prompt_builder: PromptBuilder = None,
    #             tokenizer: AbstractTokenizer = OpenAI_Tokenizer(),
    #             rag_query_n: int = 4,
    #             rag_similarity_threshold: float = 0.45,
    #             list_of_skills: list[AbstractSkill] = None,
    #             response_model: BaseModel = None,
    #             image_detail: str = "low",
    #             pool_scheduler: ThreadPoolScheduler = None):

    def _observable_query(self, observer: Observer, base64_image=None, dimensions=None, override_token_limit=False, full_image=None):
        pass

    def _query_openai_with_image(self, base64_image: str, dimensions: Tuple[int, int]) -> Observable:
        pass

    def subscribe_to_image_processing(
        self, 
        frame_observable: Observable
    ) -> Disposable:
        pass


    # @abstractmethod
    # def run(self, query: str, base64_image: str, dimensions: Tuple[int, int]) -> Observable:
    #     pass 

    

    # region Logging
    def _log_response_to_file(self, response = None, output_dir: str = '/app/assets/agent'):
        """
        Logs the response from LLM to a file.

        Args:
            response (str): The response from LLM to log.
            output_dir (str): The directory to log the response to.
        """
        if response is not None:
            with self.logging_file_memory_lock:
                with open(os.path.join(output_dir, 'memory.txt'), 'a') as file:
                    file.write(f"{self.dev_name}: {response}\n")
                    print(f"{AGENT_PRINT_COLOR}[INFO] LLM Response [{self.dev_name}]: {response}{AGENT_RESET_COLOR}")
    # endregion Logging


# region OpenAI Agent
class OpenAIAgent(LLMAgent):
    
    def __init__(self, 
                 dev_name: str, 
                 agent_type: str = "Vision",
                 query: str = "What do you see?", 
                 input_query_stream: Observable = None,
                 input_video_stream: Observable = None,
                 output_dir: str = '/app/assets/agent', 
                 agent_memory: AbstractAgentSemanticMemory = None,
                 system_query: str = None, 
                 system_query_without_documents: str = None,
                 max_input_tokens_per_request: int = 128000,
                 max_output_tokens_per_request: int = 16384,
                 model_name: str = "gpt-4o",
                 prompt_builder: PromptBuilder = None,
                 tokenizer: AbstractTokenizer = OpenAI_Tokenizer(),
                 rag_query_n: int = 4,
                 rag_similarity_threshold: float = 0.45,
                 skills: AbstractSkill = None,
                 response_model: BaseModel = None,
                 frame_processor: FrameProcessor = None,
                 image_detail: str = "low",
                 pool_scheduler: ThreadPoolScheduler = None, 
                 # Pool scheduler must be set to 2 or more for threading to work.
                 # This should a shared resource across all your consumers / agents 
                 # and should be set to the number of cores available on the machine 
                 # or more in case of hyperthreading.
                 ): 
        """
        Initializes a new instance of the OpenAIAgent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent, defaulting to 'Vision'. Currently unused.
            query (str): The default query to send along with images to OpenAI.
            input_query_stream (Observable): The input query stream to use for the agent. When provided, the agent will automatically subscribe to the stream and process queries.
            input_video_stream (Observable): The input video stream to use for the agent. When provided, the agent will automatically subscribe to the stream and process frames.
            output_dir (str): Directory where output files are stored.
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
            system_query (str): The system query template when documents are available.
            system_query_without_documents (str): The system query template when no documents are available.
            max_input_tokens_per_request (int): The maximum number of input tokens allowed per request.
            max_output_tokens_per_request (int): The maximum number of output tokens allowed per request.
            model_name (str): The model name to be used for the prompt builder.
            prompt_builder (PromptBuilder): An instance of the PromptBuilder to create prompts.
            tokenizer (AbstractTokenizer): The tokenizer to use for tokenization.
            rag_query_n (int): The number of results to retrieve for the RAG query.
            rag_similarity_threshold (float): The similarity threshold for RAG queries.
            list_of_skills (list[AbstractSkill]): The list of skills to use for the agent.
            response_model (BaseModel): The response model to use for the agent.
        """

        # TODO: Make input_video_stream not be type specific. Have it take in an observable stream of a few given types internally.
        # TODO: Have an input type parameter for the input, which may be redundant, but useful for now.
        # TODO: Have rate limiter of calls per minute, and buffer(?), and later approach maybe having that on a per stream/observable basis.

        super().__init__(dev_name, agent_type, agent_memory or AgentSemanticMemory())
        self.client = OpenAI()
        self.query = query
        self.output_dir = output_dir
        self.system_query = system_query
        self.system_query_without_documents = system_query_without_documents
        os.makedirs(self.output_dir, exist_ok=True)

        # Scheduler for thread pool
        import multiprocessing
        self.pool_scheduler = pool_scheduler or ThreadPoolScheduler(multiprocessing.cpu_count())

        # Skills Library
        # TODO: Cleanup this skills segment (var setting when none provided)
        self.skills = skills
        if self.skills is None:
            self.skills = AbstractSkill()
            self.skills.set_tools(NOT_GIVEN)
        
        # Response Model
        self.response_model = response_model if response_model is not None else NOT_GIVEN

        # Prompt Builder
        self.model_name = model_name
        self.prompt_builder = prompt_builder or PromptBuilder(self.model_name)

        # Tokenizer
        self.tokenizer: AbstractTokenizer = tokenizer or OpenAI_Tokenizer(model_name=self.model_name)

        # Depth of RAG Query
        self.rag_query_n = rag_query_n
        self.rag_similarity_threshold = rag_similarity_threshold

        # Image Detail
        self.image_detail = image_detail

        # Allocated tokens to each api call of this agent.
        self.max_output_tokens_per_request = max_output_tokens_per_request
        self.max_input_tokens_per_request = max_input_tokens_per_request
        self.max_tokens_per_request = max_input_tokens_per_request + max_output_tokens_per_request     

        # Add to agent memory (TODO: Remove/Restructure)
        # Context should be able to be added, but should not be placed here statically as such.
        def add_context_to_memory():
            self.agent_memory.add_vector("id0", "Optical Flow is a technique used to track the movement of objects in a video sequence.")
            self.agent_memory.add_vector("id1", "Edge Detection is a technique used to identify the boundaries of objects in an image.")
            self.agent_memory.add_vector("id2", "Video is a sequence of frames captured at regular intervals.")
            self.agent_memory.add_vector("id3", "Colors in Optical Flow are determined by the movement of light, and can be used to track the movement of objects.")
            self.agent_memory.add_vector("id4", "Json is a data interchange format that is easy for humans to read and write, and easy for machines to parse and generate.")
        
        add_context_to_memory()

        # Frame Processor
        self.frame_processor = frame_processor

        # Streams
        self.input_video_stream = input_video_stream
        self.input_query_stream = input_query_stream

        # Error if more than one input stream is provided.
        if (self.input_video_stream is not None) and (self.input_query_stream is not None):
            raise ValueError("More than one input stream provided. Please provide only one input stream.")

        # Input Video Stream
        if self.input_video_stream is not None:
            print(f"{AGENT_PRINT_COLOR}Subscribing to input video stream...{AGENT_RESET_COLOR}")
            self.disposables.add(
                self.subscribe_to_image_processing(self.input_video_stream) 
            )
        
        # Input Query Stream
        if self.input_query_stream is not None:
            print(f"{AGENT_PRINT_COLOR}Subscribing to input query stream...{AGENT_RESET_COLOR}")
            self.disposables.add(
                self.subscribe_to_query_processing(self.input_query_stream)
            )
        
        print(f"{AGENT_PRINT_COLOR}OpenAI Agent Initialized.{AGENT_RESET_COLOR}")

    def _observable_query(self, observer: Observer, base64_image=None, dimensions=None, override_token_limit=False, full_image=None, incoming_query: str = None):
        """
        Helper method to query OpenAI with an optional encoded image and emit to an observer.

        Args:
            observer (Observer): The observer to emit to.
            base64_image (str): The Base64-encoded image to send.

        Raises:
            Exception: If the query to OpenAI fails.
        """
        try:
            # region Query
            if incoming_query is not None:
                self.query = incoming_query
            # endregion Query

            # region RAG Context
            # Get the RAG Context
            results = self.agent_memory.query(
                query_texts=self.query,
                n_results=self.rag_query_n,
                similarity_threshold=self.rag_similarity_threshold
            )

            # Pretty format the query results
            formatted_results = "\n".join(
                f"Document ID: {doc.id}\nMetadata: {doc.metadata}\nContent: {doc.page_content}\nScore: {score}\n"
                for (doc, score) in results
            )
            print(f"{AGENT_PRINT_COLOR}Agent Memory Query Results:\n{formatted_results}{AGENT_RESET_COLOR}")
            print(f"{AGENT_PRINT_COLOR}=== Results End ==={AGENT_RESET_COLOR}")

            # Condensed results as a single string
            condensed_results = " | ".join(
                f"{doc.page_content}"
                for (doc, _) in results
            )
            # endregion RAG Context

            # region Dynamic Prompt Builder
            # Define Budgets and Policies  
            budgets = {
                "system_prompt": self.max_input_tokens_per_request // 4,
                "user_query": self.max_input_tokens_per_request // 4,
                "image": self.max_input_tokens_per_request // 4,
                "rag": self.max_input_tokens_per_request // 4,
            }
            policies = {
                "system_prompt": "truncate_end",
                "user_query": "truncate_middle",
                "image": "do_not_truncate",
                "rag": "truncate_end",
            }

            # Prompt Builder          
            messages = self.prompt_builder.build(
                user_query=self.query, 
                override_token_limit=override_token_limit,
                base64_image=base64_image, 
                image_width=dimensions[0] if dimensions is not None else None,
                image_height=dimensions[1] if dimensions is not None else None,
                image_detail=self.image_detail,
                rag_context=condensed_results, 
                fallback_system_prompt=self.system_query_without_documents,
                system_prompt=self.system_query,
                budgets=budgets,
                policies=policies,
            )
            # endregion Dynamic Prompt Builder
            
            # region OpenAI API Call
            def _tooling_callback(message, messages, response_message, skills: AbstractSkill):
                has_called_tools = False
                new_messages = []
                for tool_call in message.tool_calls:
                    has_called_tools = True
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    result = skills.call_function(name, **args)
                    print(f"{AGENT_PRINT_COLOR}Function Call Results: {result}{AGENT_RESET_COLOR}")
                    
                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                        "name": name
                    })
                
                # Complete the second call, after the functions have completed.
                if has_called_tools:
                    print(f"{AGENT_PRINT_COLOR}Sending Another Query.{AGENT_RESET_COLOR}")

                    # Send another query
                    messages.append(response_message)
                    messages.extend(new_messages)

                    # OpenAI API Call
                    response_2 = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        response_format=self.response_model,
                        tools=(skills.get_tools() if skills is not None else NOT_GIVEN),
                        max_tokens=self.max_output_tokens_per_request,
                    )
                    response_message_2 = response_2.choices[0].message
                    print(f"Message: {response_message_2}")
                    return response_message_2
                else:
                    print(f"{AGENT_PRINT_COLOR}No Need for Another Query.{AGENT_RESET_COLOR}")
                    return None

            if self.response_model is not NOT_GIVEN:
                # OpenAI API Call
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages, 
                    response_format=self.response_model,
                    tools=(self.skills.get_tools() if self.skills is not None else NOT_GIVEN),
                    max_tokens=self.max_output_tokens_per_request,
                )
                response_message = response.choices[0].message

                # Check if the response message exists.
                if response_message is None:
                    print(f"{AGENT_PRINT_COLOR}Response message does not exist.{AGENT_RESET_COLOR}")
                    observer.on_error(Exception("Response message does not exist.")) # TODO: Check this is correct syntax.
                    observer.on_completed()
                    return

                # If no skills are provided, emit the response directly.
                if (self.skills is None) or (self.skills.get_tools() is None) or (self.skills.get_tools() is NOT_GIVEN):
                    print(f"{AGENT_PRINT_COLOR}No skills provided, emitting response directly.{AGENT_RESET_COLOR}")
                    if response_message.parsed:
                        print(f"{AGENT_PRINT_COLOR}Response message parsed: {response_message.parsed}{AGENT_RESET_COLOR}")
                        observer.on_next(response_message.parsed)
                    else:
                        print(f"{AGENT_PRINT_COLOR}Response message does not have parsed data: {response_message}{AGENT_RESET_COLOR}")
                        observer.on_next(response_message.content)
                    observer.on_completed()
                    return
                
                # If the response message does not have tool calls, emit the response directly.
                if response_message.tool_calls is None:
                    print(f"{AGENT_PRINT_COLOR}Response message does not have tool calls, emitting response directly.{AGENT_RESET_COLOR}")
                    if response_message.parsed:
                        print(f"{AGENT_PRINT_COLOR}Response message parsed: {response_message.parsed}{AGENT_RESET_COLOR}")
                        observer.on_next(response_message.parsed)
                    else:
                        print(f"{AGENT_PRINT_COLOR}Response message does not have parsed data: {response_message}{AGENT_RESET_COLOR}")
                        observer.on_next(response_message.content)
                    observer.on_completed()
                    return
                
                # OpenAI API Call
                response_message_2 = _tooling_callback(
                    response_message, 
                    messages, 
                    response_message, 
                    self.skills)    

                # endregion OpenAI API Call

                # region Emit Response
                observer.on_next(response_message_2 or response_message)
                observer.on_completed()
                # endregion Emit Response
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_output_tokens_per_request,
                    tools=(self.skills.get_tools() if self.skills is not None else NOT_GIVEN),
                )
                response_message = response.choices[0].message

                # Check if the response message exists.
                if response_message is None:
                    print(f"{AGENT_PRINT_COLOR}Response message does not exist.{AGENT_RESET_COLOR}")
                    observer.on_error(Exception("Response message does not exist.")) # TODO: Check this is correct syntax.
                    observer.on_completed()
                    return
                
                # If no skills are provided, emit the response directly.
                if self.skills.get_tools() is NOT_GIVEN:
                    print(f"{AGENT_PRINT_COLOR}No skills provided, emitting response directly: {response_message}{AGENT_RESET_COLOR}")
                    observer.on_next(response_message.content)
                    observer.on_completed()
                    return
                
                # If the response message does not have tool calls, emit the response directly.
                if response_message.tool_calls is None:
                    print(f"{AGENT_PRINT_COLOR}Response message does not have tool calls, emitting response directly: {response_message}{AGENT_RESET_COLOR}")
                    observer.on_next(response_message.content)
                    observer.on_completed()
                    return

                # OpenAI API Call
                response_message_2 = _tooling_callback(
                    response_message, 
                    messages, 
                    response_message,
                    self.skills)    
                # endregion OpenAI API Call

                # region Emit Response
                observer.on_next(response_message_2 or response_message)
                observer.on_completed()
                # endregion Emit Response
            # endregion OpenAI API Call
        
        except Exception as e:
            print(f"{AGENT_PRINT_COLOR}[ERROR] OpenAI query failed in {self.dev_name}: {e}{AGENT_RESET_COLOR}")
            observer.on_error(e)

    # region Image Encoding / Decoding / Processing
    def _query_openai_with_image(self, base64_image: str, dimensions: Tuple[int, int]) -> Observable:
        """
        Sends an encoded image to OpenAI and gets a response.

        Args:
            base64_image (str): The Base64-encoded image to send.
            dimensions (Tuple[int, int]): A tuple containing the width and height of the image.

        Returns:
            Observable: An Observable that emits the response from OpenAI.
        """
        return create(lambda observer, _: self._observable_query(observer, base64_image, dimensions))

    def _query_openai_with_query(self, incoming_query: str) -> Observable:
        """
        Sends an encoded image to OpenAI and gets a response.

        Args:
            incoming_query (str): The query to send to OpenAI.

        Returns:
            Observable: An Observable that emits the response from OpenAI.
        """
        return create(lambda observer, _: self._observable_query(observer, incoming_query=incoming_query))

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
            >>> agent = OpenAIAgent("camera_1")
            >>> disposable = agent.subscribe_to_image_processing(frame_stream)
            >>> # Later cleanup
            >>> disposable.dispose()

        Note:
            The subscription is automatically added to the agent's internal
            CompositeDisposable for cleanup. The returned Disposable provides
            additional control if needed.
        """
        emission_counts = {}  # Dictionary to store counts for each id

        def print_emission_count(id, color="red"):
            color_options = {
                "red": "\033[31m",
                "green": "\033[32m",
                "blue": "\033[34m",
                "yellow": "\033[33m",
                "magenta": "\033[35m",
                "cyan": "\033[36m",
                "white": "\033[37m",
                "reset": "\033[0m"
            }
            THIS_PRINT_COLOR = color_options.get(color, "\033[31m")  # Default to red if color not found
            THIS_RESET_COLOR = color_options["reset"]

            # Initialize the count for this id if it doesn't exist
            if id not in emission_counts:
                emission_counts[id] = 0

            # Increment the count
            emission_counts[id] += 1

            # Print the current count for this id
            print(f"{THIS_PRINT_COLOR}({self.dev_name} - {id}) Emission Count - {emission_counts[id]} {datetime.now()}{THIS_RESET_COLOR}")

        processing_lock = threading.Lock()

        # Frame Processor
        if self.frame_processor is None:
            self.frame_processor = FrameProcessor(
                output_dir="/app/assets/output/frames", 
                delete_on_init=True
            )

        disposable = frame_observable.pipe(
            # ops.do_action(on_next=lambda _: print_emission_count('A')),
            ops.filter(lambda _: not processing_lock.locked()),  # Check the lock
            # ops.do_action(on_next=lambda _: print_emission_count('B')),
            ops.do_action(on_next=lambda _: processing_lock.acquire(blocking=False)),  # Acquire the lock
            # ops.do_action(on_next=lambda _: print_emission_count('C')),
            ops.observe_on(self.pool_scheduler),
            # ops.do_action(on_next=lambda _: print_emission_count('D')),
            # Process the item:
            # ==========================
            VideoOperators.with_jpeg_export(self.frame_processor, suffix=f"{self.dev_name}_frame_", save_limit=100),
            # ops.do_action(on_next=lambda _: print_emission_count('E')),
            Operators.encode_image(),
            # ops.do_action(on_next=lambda _: print_emission_count('F')),
            ops.flat_map(lambda base64_and_dims: self._query_openai_with_image(*base64_and_dims)),
            # ==========================
            # ops.do_action(on_next=lambda _: print_emission_count('G')),
            ops.do_action(on_next=lambda _: processing_lock.release()),  # Release the lock
            # ops.do_action(on_next=lambda _: print_emission_count('H')),
            ops.subscribe_on(self.pool_scheduler) # Allows the lock to be released on the main thread.
        ).subscribe(
            on_next=lambda response: self._log_response_to_file(response, self.output_dir),
            on_error=lambda e: print(f"{AGENT_PRINT_COLOR}Error in {self.dev_name}: {e}{AGENT_RESET_COLOR}"),
            on_completed=lambda: print(f"{AGENT_PRINT_COLOR}Stream processing completed for {self.dev_name}{AGENT_RESET_COLOR}")
        )
        self.disposables.add(disposable)
        return disposable
    
    def subscribe_to_query_processing(
        self, 
        query_observable: Observable
    ) -> Disposable:
        """Subscribes to and processes a stream of queries.

        Args:
            query_observable: An Observable emitting queries.

        Returns:
            A Disposable representing the subscription. Can be used for external
            resource management while still being tracked internally.
        """
        emission_counts = {}  # Dictionary to store counts for each id

        def print_emission_count(id, color="red"):
            color_options = {
                "red": "\033[31m",
                "green": "\033[32m",
                "blue": "\033[34m",
                "yellow": "\033[33m",
                "magenta": "\033[35m",
                "cyan": "\033[36m",
                "white": "\033[37m",
                "reset": "\033[0m"
            }
            THIS_PRINT_COLOR = color_options.get(color, "\033[31m")  # Default to red if color not found
            THIS_RESET_COLOR = color_options["reset"]

            # Initialize the count for this id if it doesn't exist
            if id not in emission_counts:
                emission_counts[id] = 0

            # Increment the count
            emission_counts[id] += 1

            # Print the current count for this id
            print(f"{THIS_PRINT_COLOR}({self.dev_name} - {id}) Emission Count - {emission_counts[id]} {datetime.now()}{THIS_RESET_COLOR}")

        processing_lock = threading.Lock()

        # Frame Processor
        if self.frame_processor is None:
            self.frame_processor = FrameProcessor(
                output_dir="/app/assets/output/frames", 
                delete_on_init=True
            )

        disposable = query_observable.pipe(
            # ops.do_action(on_next=lambda _: print_emission_count('A')),
            ops.filter(lambda _: not processing_lock.locked()),  # Check the lock
            # ops.do_action(on_next=lambda _: print_emission_count('B')),
            ops.do_action(on_next=lambda _: processing_lock.acquire(blocking=False)),  # Acquire the lock
            # ops.do_action(on_next=lambda _: print_emission_count('C')),
            ops.observe_on(self.pool_scheduler),
            # ops.do_action(on_next=lambda _: print_emission_count('D')),
            # Process the item:
            # ==========================
            ops.flat_map(lambda incoming_query: self._query_openai_with_query(incoming_query)),
            # ==========================
            # ops.do_action(on_next=lambda _: print_emission_count('E')),
            ops.do_action(on_next=lambda _: processing_lock.release()),  # Release the lock
            # ops.do_action(on_next=lambda _: print_emission_count('F')),
            ops.subscribe_on(self.pool_scheduler) # Allows the lock to be released on the main thread.
        ).subscribe(
            on_next=lambda response: self._log_response_to_file(response, self.output_dir),
            on_error=lambda e: print(f"{AGENT_PRINT_COLOR}Error in {self.dev_name}: {e}{AGENT_RESET_COLOR}"),
            on_completed=lambda: print(f"{AGENT_PRINT_COLOR}Stream processing completed for {self.dev_name}{AGENT_RESET_COLOR}")
        )
        self.disposables.add(disposable)
        return disposable

    # endregion Image Encoding / Decoding / Processing



# endregion OpenAI Agent
