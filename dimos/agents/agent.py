import json
import os
import threading
from typing import Tuple, Optional
from dotenv import load_dotenv
from openai import NOT_GIVEN, OpenAI
from reactivex import Observer, create, Observable, empty, operators as RxOps, throw
import reactivex
from reactivex.disposable import CompositeDisposable, Disposable
from reactivex.scheduler import ThreadPoolScheduler
from pydantic import BaseModel

from dimos.agents.memory.base import AbstractAgentSemanticMemory
from dimos.agents.memory.chroma_impl import AgentSemanticMemory
from dimos.agents.prompt_builder.impl import PromptBuilder
from dimos.agents.tokenizer.openai_impl import AbstractTokenizer, OpenAI_Tokenizer
from dimos.robot.skills import AbstractSkill
from dimos.stream.frame_processor import FrameProcessor
from dimos.stream.video_operators import Operators as MyOps, VideoOperators as MyVidOps

# Initialize environment variables
load_dotenv()

AGENT_PRINT_COLOR = "\033[33m"
AGENT_RESET_COLOR = "\033[0m"

# -----------------------------------------------------------------------------
# region Agent Base Class
# -----------------------------------------------------------------------------
class Agent:
    """Base agent that manages memory and subscriptions."""

    def __init__(self, 
                 dev_name: str = "NA", 
                 agent_type: str = "Base",
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None):
        """
        Initializes a new instance of the Agent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent (e.g., 'Base', 'Vision').
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
        """
        self.dev_name = dev_name
        self.agent_type = agent_type
        self.agent_memory = agent_memory or AgentSemanticMemory()
        self.disposables = CompositeDisposable()

    def dispose_all(self):
        """Disposes of all active subscriptions managed by this agent."""
        if self.disposables:
            self.disposables.dispose()
        else:
            print(f"{AGENT_PRINT_COLOR}No disposables to dispose.{AGENT_RESET_COLOR}")
# endregion Agent Base Class

# -----------------------------------------------------------------------------
# region LLMAgent Base Class (Generic LLM Agent)
# -----------------------------------------------------------------------------
class LLMAgent(Agent):
    """Generic LLM agent containing common logic for LLM-based agents.

    This class implements functionality for:
      - Updating the query
      - Querying the agent’s memory (for RAG)
      - Building prompts via a prompt builder
      - Handling tooling callbacks in responses
      - Subscribing to image and query streams

    Subclasses must implement the `_send_query` method, which is responsible
    for sending the prompt to a specific LLM API.
    """
    logging_file_memory_lock = threading.Lock()

    def __init__(self, 
                 dev_name: str = "NA", 
                 agent_type: str = "LLM",
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None):
        """
        Initializes a new instance of the LLMAgent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent.
            agent_memory (AbstractAgentSemanticMemory): The memory system for the agent.
        """
        super().__init__(dev_name, agent_type, agent_memory or AgentSemanticMemory())
        # These attributes can be configured by a subclass if needed.
        self.query: Optional[str] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.skills: Optional[AbstractSkill] = None
        self.system_query: Optional[str] = None
        self.system_query_without_documents: Optional[str] = None
        self.image_detail: str = "low"
        self.max_input_tokens_per_request: int = 128000
        self.max_output_tokens_per_request: int = 16384
        self.max_tokens_per_request: int = (self.max_input_tokens_per_request +
                                            self.max_output_tokens_per_request)
        self.rag_query_n: int = 4
        self.rag_similarity_threshold: float = 0.45
        self.pool_scheduler: Optional[ThreadPoolScheduler] = None
        self.frame_processor: Optional[FrameProcessor] = None
        self.output_dir: str = os.path.join(os.getcwd(), "assets", "agent")
        os.makedirs(self.output_dir, exist_ok=True)

    def _update_query(self, incoming_query: Optional[str]) -> None:
        """Updates the query if an incoming query is provided.

        Args:
            incoming_query (str): The new query text.
        """
        if incoming_query is not None:
            self.query = incoming_query

    def _get_rag_context(self) -> Tuple[str, str]:
        """Queries the agent memory to retrieve RAG context.

        Returns:
            Tuple[str, str]: A tuple containing the formatted results (for logging)
            and condensed results (for use in the prompt).
        """
        results = self.agent_memory.query(
            query_texts=self.query,
            n_results=self.rag_query_n,
            similarity_threshold=self.rag_similarity_threshold
        )
        formatted_results = "\n".join(
            f"Document ID: {doc.id}\nMetadata: {doc.metadata}\nContent: {doc.page_content}\nScore: {score}\n"
            for (doc, score) in results
        )
        condensed_results = " | ".join(
            f"{doc.page_content}" for (doc, _) in results
        )
        print(f"{AGENT_PRINT_COLOR}Agent Memory Query Results:\n{formatted_results}{AGENT_RESET_COLOR}")
        print(f"{AGENT_PRINT_COLOR}=== Results End ==={AGENT_RESET_COLOR}")
        return formatted_results, condensed_results

    def _build_prompt(self, base64_image: Optional[str], 
                      dimensions: Optional[Tuple[int, int]],
                      override_token_limit: bool,
                      condensed_results: str) -> list:
        """Builds a prompt message using the prompt builder.

        Args:
            base64_image (str): Optional Base64-encoded image.
            dimensions (Tuple[int, int]): Optional image dimensions.
            override_token_limit (bool): Whether to override token limits.
            condensed_results (str): The condensed RAG context.

        Returns:
            list: A list of message dictionaries to be sent to the LLM.
        """
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
        return self.prompt_builder.build(
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

    def _handle_tooling(self, response_message, messages):
        """Handles tooling callbacks in the response message.

        If tool calls are present, the corresponding functions are executed and
        a follow-up query is sent.

        Args:
            response_message: The response message containing tool calls.
            messages (list): The original list of messages sent.

        Returns:
            The final response message after processing tool calls, if any.
        """
        # TODO: Make this more generic or move implementation to OpenAIAgent. This is presently OpenAI-specific.
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
            if has_called_tools:
                print(f"{AGENT_PRINT_COLOR}Sending Another Query.{AGENT_RESET_COLOR}")
                messages.append(response_message)
                messages.extend(new_messages)
                # Delegate to sending the query again.
                return self._send_query(messages)
            else:
                print(f"{AGENT_PRINT_COLOR}No Need for Another Query.{AGENT_RESET_COLOR}")
                return None

        if response_message.tool_calls is not None:
            return _tooling_callback(response_message, messages, response_message, self.skills)
        return None

    def _observable_query(self,
                          observer: Observer,
                          base64_image: Optional[str] = None,
                          dimensions: Optional[Tuple[int, int]] = None,
                          override_token_limit: bool = False,
                          incoming_query: Optional[str] = None):
        """Prepares and sends a query to the LLM, emitting the response to the observer.

        Args:
            observer (Observer): The observer to emit responses to.
            base64_image (str): Optional Base64-encoded image.
            dimensions (Tuple[int, int]): Optional image dimensions.
            override_token_limit (bool): Whether to override token limits.
            incoming_query (str): Optional query to update the agent's query.

        Raises:
            Exception: Propagates any exceptions encountered during processing.
        """
        try:
            self._update_query(incoming_query)
            _, condensed_results = self._get_rag_context()
            messages = self._build_prompt(base64_image, dimensions, override_token_limit,
                                          condensed_results)
            # print(f"{AGENT_PRINT_COLOR}Sending Query: {messages}{AGENT_RESET_COLOR}")
            print(f"{AGENT_PRINT_COLOR}Sending Query.{AGENT_RESET_COLOR}")
            response_message = self._send_query(messages)
            print(f"{AGENT_PRINT_COLOR}Received Response: {response_message}{AGENT_RESET_COLOR}")
            if response_message is None:
                raise Exception("Response message does not exist.")

            # TODO: Make this more generic. The parsed tag and tooling handling may be OpenAI-specific.
            # If no skills are provided or there are no tool calls, emit the response directly.
            if (self.skills is None or self.skills.get_tools() in (None, NOT_GIVEN) or
                    response_message.tool_calls is None):
                final_msg = (response_message.parsed
                             if hasattr(response_message, 'parsed') and response_message.parsed
                             else response_message.content)
                observer.on_next(final_msg)
            else:
                response_message_2 = self._handle_tooling(response_message, messages)
                final_msg = response_message_2 if response_message_2 is not None else response_message
                observer.on_next(final_msg)
            observer.on_completed()
        except Exception as e:
            print(f"{AGENT_PRINT_COLOR}[ERROR] Query failed in {self.dev_name}: {e}{AGENT_RESET_COLOR}")
            observer.on_error(e)

    def _send_query(self, messages: list):
        """Sends the query to the LLM API.

        This method must be implemented by subclasses with specifics of the LLM API.

        Args:
            messages (list): The prompt messages to be sent.

        Returns:
            The response message from the LLM.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError("Subclasses must implement _send_query method.")

    def _log_response_to_file(self, response, output_dir: str = None):
        """Logs the LLM response to a file.

        Args:
            response: The response message to log.
            output_dir (str): The directory where the log file is stored.
        """
        if output_dir is None:
            output_dir = self.output_dir
        if response is not None:
            with self.logging_file_memory_lock:
                log_path = os.path.join(output_dir, 'memory.txt')
                with open(log_path, 'a') as file:
                    file.write(f"{self.dev_name}: {response}\n")
                print(f"{AGENT_PRINT_COLOR}[INFO] LLM Response [{self.dev_name}]: {response}{AGENT_RESET_COLOR}")

    def subscribe_to_image_processing(self, frame_observable: Observable) -> Disposable:
        """Subscribes to a stream of video frames for processing.

        This method sets up a subscription to process incoming video frames.
        Each frame is encoded and then sent to the LLM by directly calling the
        _observable_query method. The response is then logged to a file.

        Args:
            frame_observable (Observable): An observable emitting video frames.

        Returns:
            Disposable: A disposable representing the subscription.
        """
        processing_lock = threading.Lock()
        if self.frame_processor is None:
            self.frame_processor = FrameProcessor(delete_on_init=True)

        print_emission_args = {"enabled": True, "dev_name": self.dev_name, "counts": {}}
        
        def release_lock_on_error(e):
            print(f"\033[91mReleasing lock for {self.dev_name} on error: {e}\033[0m")
            safe_release()

        def safe_release():
            if processing_lock.locked():
                try:
                    processing_lock.release()
                except RuntimeError as e:
                    print("Lock not held, can't release")

        observable: Observable = frame_observable.pipe(
            MyOps.print_emission(id='A', **print_emission_args),
            RxOps.filter(lambda _: not processing_lock.locked()),
            MyOps.print_emission(id='B', **print_emission_args),
            RxOps.do_action(on_next=lambda _: processing_lock.acquire(blocking=False)),
            MyOps.print_emission(id='C', **print_emission_args),
            RxOps.observe_on(self.pool_scheduler),
            MyOps.print_emission(id='D', **print_emission_args),
            MyVidOps.with_jpeg_export(self.frame_processor, suffix=f"{self.dev_name}_frame_", save_limit=100),
            MyOps.print_emission(id='E', **print_emission_args),
            # Process the item:
            # ==========================
            MyVidOps.encode_image(),
            MyOps.print_emission(id='F', **print_emission_args),
            RxOps.filter(lambda base64_and_dims: base64_and_dims is not None 
                         and base64_and_dims[0] is not None and base64_and_dims[1] is not None),
            RxOps.flat_map(lambda base64_and_dims: create(
                lambda observer, _: self._observable_query(
                    observer, base64_image=base64_and_dims[0], dimensions=base64_and_dims[1]
                ).pipe(
                    RxOps.catch(lambda e, _: release_lock_on_error(e)),
                )
            )),
            # ==========================
            MyOps.print_emission(id='G', **print_emission_args),
            RxOps.do_action(on_next=safe_release()),
            MyOps.print_emission(id='H', **print_emission_args),
            RxOps.subscribe_on(self.pool_scheduler)
        )
        disposable = observable.subscribe(
            on_next=lambda response: self._log_response_to_file(response, self.output_dir),
            on_error=release_lock_on_error,
            on_completed=lambda: print(f"{AGENT_PRINT_COLOR}Stream processing completed for {self.dev_name}{AGENT_RESET_COLOR}")
        )
        self.disposables.add(disposable)
        return disposable

    def subscribe_to_query_processing(self, query_observable: Observable) -> Disposable:
        """Subscribes to a stream of queries for processing.

        This method sets up a subscription to process incoming queries by directly
        calling the _observable_query method. The responses are logged to a file.

        Args:
            query_observable (Observable): An observable emitting queries.

        Returns:
            Disposable: A disposable representing the subscription.
        """
        processing_lock = threading.Lock()
        print_emission_args = {"enabled": True, "dev_name": self.dev_name, "counts": {}}

        def release_lock_on_error(e):
            print(f"\033[91mReleasing lock for {self.dev_name} on error: {e}\033[0m")
            safe_release()

        def safe_release():
            if processing_lock.locked():
                try:
                    processing_lock.release()
                except RuntimeError as e:
                    print("Lock not held, can't release")

        observable: Observable = query_observable.pipe(
            MyOps.print_emission(id='A', **print_emission_args),
            RxOps.filter(lambda _: not processing_lock.locked()),
            MyOps.print_emission(id='B', **print_emission_args),
            RxOps.do_action(on_next=lambda _: processing_lock.acquire(blocking=False)),
            MyOps.print_emission(id='C', **print_emission_args),
            RxOps.observe_on(self.pool_scheduler),
            MyOps.print_emission(id='D', **print_emission_args),
            # Process the item:
            # ==========================
            RxOps.flat_map(lambda incoming_query: create(
                lambda observer, _: self._observable_query(observer, incoming_query=incoming_query)
            )),
            # ==========================
            MyOps.print_emission(id='E', **print_emission_args),
            RxOps.do_action(on_next=lambda _: processing_lock.release()),
            MyOps.print_emission(id='F', **print_emission_args),
            RxOps.subscribe_on(self.pool_scheduler)
        )
        disposable = observable.subscribe(
            on_next=lambda response: self._log_response_to_file(response, self.output_dir),
            on_error=release_lock_on_error,
            on_completed=lambda: print(f"{AGENT_PRINT_COLOR}Stream processing completed for {self.dev_name}{AGENT_RESET_COLOR}")
        )
        self.disposables.add(disposable)
        return disposable
# endregion LLMAgent Base Class (Generic LLM Agent)

# -----------------------------------------------------------------------------
# region OpenAIAgent Subclass (OpenAI-Specific Implementation)
# -----------------------------------------------------------------------------
class OpenAIAgent(LLMAgent):
    """OpenAI agent implementation that uses OpenAI's API for processing.

    This class implements the _send_query method to interact with OpenAI's API.
    It also sets up OpenAI-specific parameters, such as the client, model name,
    tokenizer, and response model.
    """

    def __init__(self,
                 dev_name: str,
                 agent_type: str = "Vision",
                 query: str = "What do you see?",
                 input_query_stream: Optional[Observable] = None,
                 input_video_stream: Optional[Observable] = None,
                 output_dir: str = os.path.join(os.getcwd(), "assets", "agent"),
                 agent_memory: Optional[AbstractAgentSemanticMemory] = None,
                 system_query: Optional[str] = None,
                 system_query_without_documents: Optional[str] = None,
                 max_input_tokens_per_request: int = 128000,
                 max_output_tokens_per_request: int = 16384,
                 model_name: str = "gpt-4o",
                 prompt_builder: Optional[PromptBuilder] = None,
                 tokenizer: Optional[AbstractTokenizer] = None,
                 rag_query_n: int = 4,
                 rag_similarity_threshold: float = 0.45,
                 skills: Optional[AbstractSkill] = None,
                 response_model: Optional[BaseModel] = None,
                 frame_processor: Optional[FrameProcessor] = None,
                 image_detail: str = "low",
                 pool_scheduler: Optional[ThreadPoolScheduler] = None):
        """
        Initializes a new instance of the OpenAIAgent.

        Args:
            dev_name (str): The device name of the agent.
            agent_type (str): The type of the agent.
            query (str): The default query text.
            input_query_stream (Observable): An observable for query input.
            input_video_stream (Observable): An observable for video frames.
            output_dir (str): Directory for output files.
            agent_memory (AbstractAgentSemanticMemory): The memory system.
            system_query (str): The system prompt template.
            system_query_without_documents (str): Fallback system prompt.
            max_input_tokens_per_request (int): Maximum input tokens.
            max_output_tokens_per_request (int): Maximum output tokens.
            model_name (str): The OpenAI model name.
            prompt_builder (PromptBuilder): The prompt builder instance.
            tokenizer (AbstractTokenizer): The tokenizer.
            rag_query_n (int): Number of results for RAG.
            rag_similarity_threshold (float): Similarity threshold for RAG.
            skills (AbstractSkill): Skills available to the agent.
            response_model (BaseModel): The response model.
            frame_processor (FrameProcessor): The frame processor.
            image_detail (str): Level of image detail.
            pool_scheduler (ThreadPoolScheduler): Scheduler for threading.
        """
        super().__init__(dev_name, agent_type, agent_memory or AgentSemanticMemory())
        self.client = OpenAI()
        self.query = query
        self.output_dir = output_dir
        self.system_query = system_query
        self.system_query_without_documents = system_query_without_documents
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up a thread pool scheduler.
        import multiprocessing
        self.pool_scheduler = pool_scheduler or ThreadPoolScheduler(multiprocessing.cpu_count())

        # Configure skills.
        self.skills = skills if skills is not None else AbstractSkill()
        if skills is None:
            self.skills.set_tools(NOT_GIVEN)

        self.response_model = response_model if response_model is not None else NOT_GIVEN
        self.model_name = model_name
        self.prompt_builder = prompt_builder or PromptBuilder(self.model_name)
        self.tokenizer = tokenizer or OpenAI_Tokenizer(model_name=self.model_name)
        self.rag_query_n = rag_query_n
        self.rag_similarity_threshold = rag_similarity_threshold
        self.image_detail = image_detail
        self.max_output_tokens_per_request = max_output_tokens_per_request
        self.max_input_tokens_per_request = max_input_tokens_per_request
        self.max_tokens_per_request = max_input_tokens_per_request + max_output_tokens_per_request

        # Add static context to memory.
        self._add_context_to_memory()

        self.frame_processor = frame_processor or FrameProcessor(delete_on_init=True)
        self.input_video_stream = input_video_stream
        self.input_query_stream = input_query_stream

        # Ensure only one input stream is provided.
        if self.input_video_stream is not None and self.input_query_stream is not None:
            raise ValueError("More than one input stream provided. Please provide only one input stream.")

        if self.input_video_stream is not None:
            print(f"{AGENT_PRINT_COLOR}Subscribing to input video stream...{AGENT_RESET_COLOR}")
            self.disposables.add(self.subscribe_to_image_processing(self.input_video_stream))
        if self.input_query_stream is not None:
            print(f"{AGENT_PRINT_COLOR}Subscribing to input query stream...{AGENT_RESET_COLOR}")
            self.disposables.add(self.subscribe_to_query_processing(self.input_query_stream))

        print(f"{AGENT_PRINT_COLOR}OpenAI Agent Initialized.{AGENT_RESET_COLOR}")

    def _add_context_to_memory(self):
        """Adds initial context to the agent's memory."""
        context_data = [
            ("id0", "Optical Flow is a technique used to track the movement of objects in a video sequence."),
            ("id1", "Edge Detection is a technique used to identify the boundaries of objects in an image."),
            ("id2", "Video is a sequence of frames captured at regular intervals."),
            ("id3", "Colors in Optical Flow are determined by the movement of light, and can be used to track the movement of objects."),
            ("id4", "Json is a data interchange format that is easy for humans to read and write, and easy for machines to parse and generate."),
        ]
        for doc_id, text in context_data:
            self.agent_memory.add_vector(doc_id, text)

    def _send_query(self, messages: list):
        """Sends the query to OpenAI's API.

        Depending on whether a response model is provided, the appropriate API
        call is made.

        Args:
            messages (list): The prompt messages to send.

        Returns:
            The response message from OpenAI.

        Raises:
            Exception: If no response message is returned.
        """
        if self.response_model is not NOT_GIVEN:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=self.response_model,
                tools=(self.skills.get_tools() if self.skills is not None else NOT_GIVEN),
                max_tokens=self.max_output_tokens_per_request,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_output_tokens_per_request,
                tools=(self.skills.get_tools() if self.skills is not None else NOT_GIVEN),
            )
        response_message = response.choices[0].message
        if response_message is None:
            print(f"{AGENT_PRINT_COLOR}Response message does not exist.{AGENT_RESET_COLOR}")
            raise Exception("Response message does not exist.")
        return response_message
# endregion OpenAIAgent Subclass (OpenAI-Specific Implementation)
