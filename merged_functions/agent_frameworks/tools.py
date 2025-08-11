# Merged file for agent_frameworks/tools
# This file contains code merged from multiple repositories

import json
import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
from typing import cast
from typing import runtime_checkable
import jsonref
from pydantic import BaseModel
from typing_extensions import NotRequired
from typing_extensions import TypedDict
from  import EVENT_LOGGER_NAME
from  import CancellationToken
from _component_config import ComponentBase
from _function_utils import normalize_annotated_type
from _telemetry import trace_tool_span
from logging import ToolCallEvent

# From tools/_base.py
class ParametersSchema(TypedDict):
    type: str
    properties: Dict[str, Any]
    required: NotRequired[Sequence[str]]
    additionalProperties: NotRequired[bool]

# From tools/_base.py
class ToolSchema(TypedDict):
    parameters: NotRequired[ParametersSchema]
    name: str
    description: NotRequired[str]
    strict: NotRequired[bool]

# From tools/_base.py
class ToolOverride(BaseModel):
    """Override configuration for a tool's name and/or description."""

    name: Optional[str] = None
    description: Optional[str] = None

# From tools/_base.py
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def schema(self) -> ToolSchema: ...

    def args_type(self) -> Type[BaseModel]: ...

    def return_type(self) -> Type[Any]: ...

    def state_type(self) -> Type[BaseModel] | None: ...

    def return_value_as_string(self, value: Any) -> str: ...

    async def run_json(
        self, args: Mapping[str, Any], cancellation_token: CancellationToken, call_id: str | None = None
    ) -> Any: ...

    async def save_state_json(self) -> Mapping[str, Any]: ...

    async def load_state_json(self, state: Mapping[str, Any]) -> None: ...

# From tools/_base.py
class StreamTool(Tool, Protocol):
    def run_json_stream(
        self, args: Mapping[str, Any], cancellation_token: CancellationToken, call_id: str | None = None
    ) -> AsyncGenerator[Any, None]: ...

# From tools/_base.py
class BaseTool(ABC, Tool, Generic[ArgsT, ReturnT], ComponentBase[BaseModel]):
    component_type = "tool"

    def __init__(
        self,
        args_type: Type[ArgsT],
        return_type: Type[ReturnT],
        name: str,
        description: str,
        strict: bool = False,
    ) -> None:
        self._args_type = args_type
        # Normalize Annotated to the base type.
        self._return_type = normalize_annotated_type(return_type)
        self._name = name
        self._description = description
        self._strict = strict

    @property
    def schema(self) -> ToolSchema:
        model_schema: Dict[str, Any] = self._args_type.model_json_schema()

        if "$defs" in model_schema:
            model_schema = cast(Dict[str, Any], jsonref.replace_refs(obj=model_schema, proxies=False))  # type: ignore
            del model_schema["$defs"]

        parameters = ParametersSchema(
            type="object",
            properties=model_schema["properties"],
            required=model_schema.get("required", []),
            additionalProperties=model_schema.get("additionalProperties", False),
        )

        # If strict is enabled, the tool schema should list all properties as required.
        assert "required" in parameters
        if self._strict and set(parameters["required"]) != set(parameters["properties"].keys()):
            raise ValueError(
                "Strict mode is enabled, but not all input arguments are marked as required. Default arguments are not allowed in strict mode."
            )

        assert "additionalProperties" in parameters
        if self._strict and parameters["additionalProperties"]:
            raise ValueError(
                "Strict mode is enabled but additional argument is also enabled. This is not allowed in strict mode."
            )

        tool_schema = ToolSchema(
            name=self._name,
            description=self._description,
            parameters=parameters,
            strict=self._strict,
        )
        return tool_schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def args_type(self) -> Type[BaseModel]:
        return self._args_type

    def return_type(self) -> Type[Any]:
        return self._return_type

    def state_type(self) -> Type[BaseModel] | None:
        return None

    def return_value_as_string(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return json.dumps(dumped)
            return str(dumped)

        return str(value)

    @abstractmethod
    async def run(self, args: ArgsT, cancellation_token: CancellationToken) -> ReturnT: ...

    async def run_json(
        self, args: Mapping[str, Any], cancellation_token: CancellationToken, call_id: str | None = None
    ) -> Any:
        """Run the tool with the provided arguments in a dictionary.

        Args:
            args (Mapping[str, Any]): The arguments to pass to the tool.
            cancellation_token (CancellationToken): A token to cancel the operation if needed.
            call_id (str | None): An optional identifier for the tool call, used for tracing.

        Returns:
            Any: The return value of the tool's run method.
        """
        with trace_tool_span(
            tool_name=self._name,
            tool_description=self._description,
            tool_call_id=call_id,
        ):
            # Execute the tool's run method
            return_value = await self.run(self._args_type.model_validate(args), cancellation_token)

        # Log the tool call event
        event = ToolCallEvent(
            tool_name=self.name,
            arguments=dict(args),  # Using the raw args passed to run_json
            result=self.return_value_as_string(return_value),
        )
        logger.info(event)

        return return_value

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass

# From tools/_base.py
class BaseStreamTool(
    BaseTool[ArgsT, ReturnT], StreamTool, ABC, Generic[ArgsT, StreamT, ReturnT], ComponentBase[BaseModel]
):
    component_type = "tool"

    @abstractmethod
    def run_stream(self, args: ArgsT, cancellation_token: CancellationToken) -> AsyncGenerator[StreamT | ReturnT, None]:
        """Run the tool with the provided arguments and return a stream of data and end with the final return value."""
        ...

    async def run_json_stream(
        self,
        args: Mapping[str, Any],
        cancellation_token: CancellationToken,
        call_id: str | None = None,
    ) -> AsyncGenerator[StreamT | ReturnT, None]:
        """Run the tool with the provided arguments in a dictionary and return a stream of data
        from the tool's :meth:`run_stream` method and end with the final return value.

        Args:
            args (Mapping[str, Any]): The arguments to pass to the tool.
            cancellation_token (CancellationToken): A token to cancel the operation if needed.
            call_id (str | None): An optional identifier for the tool call, used for tracing.

        Returns:
            AsyncGenerator[StreamT | ReturnT, None]: A generator yielding results from the tool's :meth:`run_stream` method.
        """
        return_value: ReturnT | StreamT | None = None
        with trace_tool_span(
            tool_name=self._name,
            tool_description=self._description,
            tool_call_id=call_id,
        ):
            # Execute the tool's run_stream method
            async for result in self.run_stream(self._args_type.model_validate(args), cancellation_token):
                return_value = result
                yield result

        assert return_value is not None, "The tool must yield a final return value at the end of the stream."
        if not isinstance(return_value, self._return_type):
            raise TypeError(
                f"Expected return value of type {self._return_type.__name__}, but got {type(return_value).__name__}"
            )

        # Log the tool call event
        event = ToolCallEvent(
            tool_name=self.name,
            arguments=dict(args),  # Using the raw args passed to run_json
            result=self.return_value_as_string(return_value),
        )
        logger.info(event)

# From tools/_base.py
class BaseToolWithState(BaseTool[ArgsT, ReturnT], ABC, Generic[ArgsT, ReturnT, StateT], ComponentBase[BaseModel]):
    def __init__(
        self,
        args_type: Type[ArgsT],
        return_type: Type[ReturnT],
        state_type: Type[StateT],
        name: str,
        description: str,
    ) -> None:
        super().__init__(args_type, return_type, name, description)
        self._state_type = state_type

    component_type = "tool"

    @abstractmethod
    def save_state(self) -> StateT: ...

    @abstractmethod
    def load_state(self, state: StateT) -> None: ...

    async def save_state_json(self) -> Mapping[str, Any]:
        return self.save_state().model_dump()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        self.load_state(self._state_type.model_validate(state))

# From tools/_base.py
def name(self) -> str: ...

# From tools/_base.py
def description(self) -> str: ...

# From tools/_base.py
def schema(self) -> ToolSchema: ...

# From tools/_base.py
def args_type(self) -> Type[BaseModel]: ...

# From tools/_base.py
def return_type(self) -> Type[Any]: ...

# From tools/_base.py
def state_type(self) -> Type[BaseModel] | None: ...

# From tools/_base.py
def return_value_as_string(self, value: Any) -> str: ...

# From tools/_base.py
def run_json_stream(
        self, args: Mapping[str, Any], cancellation_token: CancellationToken, call_id: str | None = None
    ) -> AsyncGenerator[Any, None]: ...

# From tools/_base.py
def run_stream(self, args: ArgsT, cancellation_token: CancellationToken) -> AsyncGenerator[StreamT | ReturnT, None]:
        """Run the tool with the provided arguments and return a stream of data and end with the final return value."""
        ...

# From tools/_base.py
def save_state(self) -> StateT: ...

# From tools/_base.py
def load_state(self, state: StateT) -> None: ...

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from autogen_core.tools import ToolSchema
from semantic_kernel.functions import KernelFunctionFromMethod
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

# From semantic_kernel/_kernel_function_from_tool.py
class KernelFunctionFromTool(KernelFunctionFromMethod):
    def __init__(self, tool: BaseTool[InputT, OutputT], plugin_name: str | None = None):
        # Get the pydantic model types from the tool
        args_type = tool.args_type()
        return_type = tool.return_type()

        # 1) Define an async function that calls the tool
        @kernel_function(name=tool.name, description=tool.description)
        async def tool_method(**kwargs: dict[str, Any]) -> Any:
            return await tool.run_json(kwargs, cancellation_token=CancellationToken())

        # Parse schema for parameters
        parameters_meta: list[KernelParameterMetadata] = []
        properties = tool.schema.get("parameters", {}).get("properties", {})

        # Get the field types from the pydantic model
        field_types = args_type.model_fields

        for prop_name, prop_info in properties.items():
            assert prop_name in field_types, f"Property {prop_name} not found in Tool {tool.name}"
            assert isinstance(prop_info, dict), f"Property {prop_name} is not a dict in Tool {tool.name}"

            # Get the actual type from the pydantic model field
            field_type = field_types[prop_name]
            parameters_meta.append(
                KernelParameterMetadata(
                    name=prop_name,
                    description=field_type.description or "",
                    default_value=field_type.get_default(),
                    type=prop_info.get("type", "string"),  # type: ignore
                    type_object=field_type.annotation,
                    is_required=field_type.is_required(),
                )
            )

        # Create return parameter metadata
        return_parameter = KernelParameterMetadata(
            name="return",
            description=f"Result from '{tool.name}' tool",
            default_value=None,
            type="object" if issubclass(return_type, BaseModel) else "string",
            type_object=return_type,
            is_required=True,
        )

        # Initialize the parent class
        super().__init__(
            method=tool_method,
            plugin_name=plugin_name,
            parameters=parameters_meta,
            return_parameter=return_parameter,
            additional_metadata=None,
        )

        self._tool = tool

# From semantic_kernel/_kernel_function_from_tool.py
class KernelFunctionFromToolSchema(KernelFunctionFromPrompt):
    def __init__(self, tool_schema: ToolSchema, plugin_name: str | None = None):
        properties = tool_schema.get("parameters", {}).get("properties", {})
        required = properties.get("required", [])

        prompt_template_config = PromptTemplateConfig(
            name=tool_schema.get("name", ""),
            description=tool_schema.get("description", ""),
            input_variables=[
                InputVariable(
                    name=prop_name, description=prop_info.get("description", ""), is_required=prop_name in required
                )
                for prop_name, prop_info in properties.items()
            ],
        )

        super().__init__(
            function_name=tool_schema.get("name", ""),
            plugin_name=plugin_name,
            description=tool_schema.get("description", ""),
            prompt_template_config=prompt_template_config,
        )

import base64
from typing import List
from typing import Tuple
import cv2
import ffmpeg
import numpy
import whisper
from autogen_core import Image
from autogen_core.models import ChatCompletionClient
from autogen_core.models import UserMessage

# From video_surfer/tools.py
def extract_audio(video_path: str, audio_output_path: str) -> str:
    """
    Extracts audio from a video file and saves it as an MP3 file.

    :param video_path: Path to the video file.
    :param audio_output_path: Path to save the extracted audio file.
    :return: Confirmation message with the path to the saved audio file.
    """
    (ffmpeg.input(video_path).output(audio_output_path, format="mp3").run(quiet=True, overwrite_output=True))  # type: ignore
    return f"Audio extracted and saved to {audio_output_path}."

# From video_surfer/tools.py
def transcribe_audio_with_timestamps(audio_path: str) -> str:
    """
    Transcribes the audio file with timestamps using the Whisper model.

    :param audio_path: Path to the audio file.
    :return: Transcription with timestamps.
    """
    model = whisper.load_model("base")  # type: ignore
    result: Dict[str, Any] = model.transcribe(audio_path, task="transcribe", language="en", verbose=False)  # type: ignore

    segments: List[Dict[str, Any]] = result["segments"]
    transcription_with_timestamps = ""

    for segment in segments:
        start: float = segment["start"]
        end: float = segment["end"]
        text: str = segment["text"]
        transcription_with_timestamps += f"[{start:.2f} - {end:.2f}] {text}\n"

    return transcription_with_timestamps

# From video_surfer/tools.py
def get_video_length(video_path: str) -> str:
    """
    Returns the length of the video in seconds.

    :param video_path: Path to the video file.
    :return: Duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    return f"The video is {duration:.2f} seconds long."

# From video_surfer/tools.py
def save_screenshot(video_path: str, timestamp: float, output_path: str) -> None:
    """
    Captures a screenshot at the specified timestamp and saves it to the output path.

    :param video_path: Path to the video file.
    :param timestamp: Timestamp in seconds.
    :param output_path: Path to save the screenshot. The file format is determined by the extension in the path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    else:
        raise IOError(f"Failed to capture frame at {timestamp:.2f}s")
    cap.release()

# From video_surfer/tools.py
def get_screenshot_at(video_path: str, timestamps: List[float]) -> List[Tuple[float, np.ndarray[Any, Any]]]:
    """
    Captures screenshots at the specified timestamps and returns them as Python objects.

    :param video_path: Path to the video file.
    :param timestamps: List of timestamps in seconds.
    :return: List of tuples containing timestamp and the corresponding frame (image).
             Each frame is a NumPy array (height x width x channels).
    """
    screenshots: List[Tuple[float, np.ndarray[Any, Any]]] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    for timestamp in timestamps:
        if 0 <= timestamp <= duration:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                # Append the timestamp and frame to the list
                screenshots.append((timestamp, frame))
            else:
                raise IOError(f"Failed to capture frame at {timestamp:.2f}s")
        else:
            raise ValueError(f"Timestamp {timestamp:.2f}s is out of range [0s, {duration:.2f}s]")

    cap.release()
    return screenshots

import asyncio
import re
import warnings
from typing import Literal
from typing import Sequence
from typing import TypedDict
from typing import Union
from autogen_core import EVENT_LOGGER_NAME
from autogen_core import FunctionCall
from autogen_core import MessageHandlerContext
from autogen_core.logging import LLMCallEvent
from autogen_core.models import AssistantMessage
from autogen_core.models import CreateResult
from autogen_core.models import FinishReasons
from autogen_core.models import FunctionExecutionResultMessage
from autogen_core.models import LLMMessage
from autogen_core.models import ModelFamily
from autogen_core.models import ModelInfo
from autogen_core.models import RequestUsage
from autogen_core.models import SystemMessage
from autogen_core.models import validate_model_info
from autogen_core.tools import Tool
from llama_cpp import ChatCompletionFunctionParameters
from llama_cpp import ChatCompletionRequestAssistantMessage
from llama_cpp import ChatCompletionRequestFunctionMessage
from llama_cpp import ChatCompletionRequestSystemMessage
from llama_cpp import ChatCompletionRequestToolMessage
from llama_cpp import ChatCompletionRequestUserMessage
from llama_cpp import ChatCompletionTool
from llama_cpp import ChatCompletionToolFunction
from llama_cpp import Llama
from llama_cpp import llama_chat_format
from typing_extensions import Unpack

# From llama_cpp/_llama_cpp_completion_client.py
class LlamaCppParams(TypedDict, total=False):
    # from_pretrained parameters:
    repo_id: Optional[str]
    filename: Optional[str]
    additional_files: Optional[List[Any]]
    local_dir: Optional[str]
    local_dir_use_symlinks: Union[bool, Literal["auto"]]
    cache_dir: Optional[str]
    # __init__ parameters:
    model_path: str
    n_gpu_layers: int
    split_mode: int
    main_gpu: int
    tensor_split: Optional[List[float]]
    rpc_servers: Optional[str]
    vocab_only: bool
    use_mmap: bool
    use_mlock: bool
    kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]]
    seed: int
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_threads: Optional[int]
    n_threads_batch: Optional[int]
    rope_scaling_type: Optional[int]
    pooling_type: int
    rope_freq_base: float
    rope_freq_scale: float
    yarn_ext_factor: float
    yarn_attn_factor: float
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_orig_ctx: int
    logits_all: bool
    embedding: bool
    offload_kqv: bool
    flash_attn: bool
    no_perf: bool
    last_n_tokens_size: int
    lora_base: Optional[str]
    lora_scale: float
    lora_path: Optional[str]
    numa: Union[bool, int]
    chat_format: Optional[str]
    chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler]
    draft_model: Optional[Any]  # LlamaDraftModel not exposed by llama_cpp
    tokenizer: Optional[Any]  # BaseLlamaTokenizer not exposed by llama_cpp
    type_k: Optional[int]
    type_v: Optional[int]
    spm_infill: bool
    verbose: bool

# From llama_cpp/_llama_cpp_completion_client.py
class LlamaCppChatCompletionClient(ChatCompletionClient):
    """Chat completion client for LlamaCpp models.
    To use this client, you must install the `llama-cpp` extra:

    .. code-block:: bash

        pip install "autogen-ext[llama-cpp]"

    This client allows you to interact with LlamaCpp models, either by specifying a local model path or by downloading a model from Hugging Face Hub.

    Args:
        model_info (optional, ModelInfo): The information about the model. Defaults to :attr:`~LlamaCppChatCompletionClient.DEFAULT_MODEL_INFO`.
        model_path (optional, str): The path to the LlamaCpp model file. Required if repo_id and filename are not provided.
        repo_id (optional, str): The Hugging Face Hub repository ID. Required if model_path is not provided.
        filename (optional, str): The filename of the model within the Hugging Face Hub repository. Required if model_path is not provided.
        n_gpu_layers (optional, int): The number of layers to put on the GPU.
        n_ctx (optional, int): The context size.
        n_batch (optional, int): The batch size.
        verbose (optional, bool): Whether to print verbose output.
        **kwargs: Additional parameters to pass to the Llama class.

    Examples:

        The following code snippet shows how to use the client with a local model file:

        .. code-block:: python

            import asyncio

            from autogen_core.models import UserMessage
            from autogen_ext.models.llama_cpp import LlamaCppChatCompletionClient


            async def main():
                llama_client = LlamaCppChatCompletionClient(model_path="/path/to/your/model.gguf")
                result = await llama_client.create([UserMessage(content="What is the capital of France?", source="user")])
                print(result)


            asyncio.run(main())

        The following code snippet shows how to use the client with a model from Hugging Face Hub:

        .. code-block:: python

            import asyncio

            from autogen_core.models import UserMessage
            from autogen_ext.models.llama_cpp import LlamaCppChatCompletionClient


            async def main():
                llama_client = LlamaCppChatCompletionClient(
                    repo_id="unsloth/phi-4-GGUF", filename="phi-4-Q2_K_L.gguf", n_gpu_layers=-1, seed=1337, n_ctx=5000
                )
                result = await llama_client.create([UserMessage(content="What is the capital of France?", source="user")])
                print(result)


            asyncio.run(main())
    """

    DEFAULT_MODEL_INFO: ModelInfo = ModelInfo(
        vision=False, json_output=True, family=ModelFamily.UNKNOWN, function_calling=True, structured_output=True
    )

    def __init__(
        self,
        model_info: Optional[ModelInfo] = None,
        **kwargs: Unpack[LlamaCppParams],
    ) -> None:
        """
        Initialize the LlamaCpp client.
        """

        if model_info:
            validate_model_info(model_info)
            self._model_info = model_info
        else:
            # Default model info.
            self._model_info = self.DEFAULT_MODEL_INFO

        if "repo_id" in kwargs and "filename" in kwargs and kwargs["repo_id"] and kwargs["filename"]:
            repo_id: str = cast(str, kwargs.pop("repo_id"))
            filename: str = cast(str, kwargs.pop("filename"))
            pretrained = Llama.from_pretrained(repo_id=repo_id, filename=filename, **kwargs)  # type: ignore
            assert isinstance(pretrained, Llama)
            self.llm = pretrained

        elif "model_path" in kwargs:
            self.llm = Llama(**kwargs)  # pyright: ignore[reportUnknownMemberType]
        else:
            raise ValueError("Please provide model_path if ... or provide repo_id and filename if ....")
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        # None means do not override the default
        # A value means to override the client default - often specified in the constructor
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        create_args = dict(extra_create_args)
        # Convert LLMMessage objects to dictionaries with 'role' and 'content'
        # converted_messages: List[Dict[str, str | Image | list[str | Image] | list[FunctionCall]]] = []
        converted_messages: list[
            ChatCompletionRequestSystemMessage
            | ChatCompletionRequestUserMessage
            | ChatCompletionRequestAssistantMessage
            | ChatCompletionRequestUserMessage
            | ChatCompletionRequestToolMessage
            | ChatCompletionRequestFunctionMessage
        ] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage) and isinstance(msg.content, str):
                converted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                converted_messages.append({"role": "assistant", "content": msg.content})
            elif (
                isinstance(msg, SystemMessage) or isinstance(msg, UserMessage) or isinstance(msg, AssistantMessage)
            ) and isinstance(msg.content, list):
                raise ValueError("Multi-part messages such as those containing images are currently not supported.")
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            create_args["response_format"] = {"type": "json_object", "schema": json_output.model_json_schema()}
        elif json_output is True:
            create_args["response_format"] = {"type": "json_object"}
        elif json_output is not False and json_output is not None:
            raise ValueError("json_output must be a boolean, a BaseModel subclass or None.")

        # Handle tool_choice parameter
        if tool_choice != "auto":
            warnings.warn(
                "tool_choice parameter is specified but LlamaCppChatCompletionClient does not support it. "
                "This parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if self.model_info["function_calling"]:
            # Run this in on the event loop to avoid blocking.
            response_future = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=converted_messages, tools=convert_tools(tools), stream=False, **create_args
                ),
            )
        else:
            response_future = asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.create_chat_completion(messages=converted_messages, stream=False, **create_args)
            )
        if cancellation_token:
            cancellation_token.link_future(response_future)
        response = await response_future

        if not isinstance(response, dict):
            raise ValueError("Unexpected response type from LlamaCpp model.")

        self._total_usage["prompt_tokens"] += response["usage"]["prompt_tokens"]
        self._total_usage["completion_tokens"] += response["usage"]["completion_tokens"]

        # Parse the response
        response_tool_calls: ChatCompletionTool | None = None
        response_text: str | None = None
        if "choices" in response and len(response["choices"]) > 0:
            if "message" in response["choices"][0]:
                response_text = response["choices"][0]["message"]["content"]
            if "tool_calls" in response["choices"][0]:
                response_tool_calls = response["choices"][0]["tool_calls"]  # type: ignore

        content: List[FunctionCall] | str = ""
        thought: str | None = None
        if response_tool_calls:
            content = []
            for tool_call in response_tool_calls:
                if not isinstance(tool_call, dict):
                    raise ValueError("Unexpected tool call type from LlamaCpp model.")
                content.append(
                    FunctionCall(
                        id=tool_call["id"],
                        arguments=tool_call["function"]["arguments"],
                        name=normalize_name(tool_call["function"]["name"]),
                    )
                )
            if response_text and len(response_text) > 0:
                thought = response_text
        else:
            if response_text:
                content = response_text

        # Detect tool usage in the response
        if not response_tool_calls and not response_text:
            logger.debug("DEBUG: No response text found. Returning empty response.")
            return CreateResult(
                content="", usage=RequestUsage(prompt_tokens=0, completion_tokens=0), finish_reason="stop", cached=False
            )

        # Create a CreateResult object
        if "finish_reason" in response["choices"][0]:
            finish_reason = response["choices"][0]["finish_reason"]
        else:
            finish_reason = "unknown"
        if finish_reason not in ("stop", "length", "function_calls", "content_filter", "unknown"):
            finish_reason = "unknown"
        create_result = CreateResult(
            content=content,
            thought=thought,
            usage=cast(RequestUsage, response["usage"]),
            finish_reason=normalize_stop_reason(finish_reason),  # type: ignore
            cached=False,
        )

        # If we are running in the context of a handler we can get the agent_id
        try:
            agent_id = MessageHandlerContext.agent_id()
        except RuntimeError:
            agent_id = None

        logger.info(
            LLMCallEvent(
                messages=cast(List[Dict[str, Any]], converted_messages),
                response=create_result.model_dump(),
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
                agent_id=agent_id,
            )
        )
        return create_result

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        # None means do not override the default
        # A value means to override the client default - often specified in the constructor
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        # Validate tool_choice parameter even though streaming is not implemented
        if tool_choice != "auto" and tool_choice != "none":
            if not self.model_info["function_calling"]:
                raise ValueError("tool_choice specified but model does not support function calling")
            if len(tools) == 0:
                raise ValueError("tool_choice specified but no tools provided")
            logger.warning("tool_choice parameter specified but may not be supported by llama-cpp-python")

        raise NotImplementedError("Stream not yet implemented for LlamaCppChatCompletionClient")
        yield ""

    # Implement abstract methods
    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_usage.get("prompt_tokens", 0),
            completion_tokens=self._total_usage.get("completion_tokens", 0),
        )

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    def count_tokens(
        self,
        messages: Sequence[SystemMessage | UserMessage | AssistantMessage | FunctionExecutionResultMessage],
        **kwargs: Any,
    ) -> int:
        total = 0
        for msg in messages:
            # Use the Llama model's tokenizer to encode the content
            tokens = self.llm.tokenize(str(msg.content).encode("utf-8"))
            total += len(tokens)
        return total

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    def remaining_tokens(
        self,
        messages: Sequence[SystemMessage | UserMessage | AssistantMessage | FunctionExecutionResultMessage],
        **kwargs: Any,
    ) -> int:
        used_tokens = self.count_tokens(messages)
        return max(self.llm.n_ctx() - used_tokens, 0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_usage.get("prompt_tokens", 0),
            completion_tokens=self._total_usage.get("completion_tokens", 0),
        )

    async def close(self) -> None:
        """
        Close the LlamaCpp client.
        """
        self.llm.close()

# From llama_cpp/_llama_cpp_completion_client.py
def normalize_stop_reason(stop_reason: str | None) -> FinishReasons:
    if stop_reason is None:
        return "unknown"

    # Convert to lower case
    stop_reason = stop_reason.lower()

    KNOWN_STOP_MAPPINGS: Dict[str, FinishReasons] = {
        "stop": "stop",
        "length": "length",
        "content_filter": "content_filter",
        "function_calls": "function_calls",
        "end_turn": "stop",
        "tool_calls": "function_calls",
    }

    return KNOWN_STOP_MAPPINGS.get(stop_reason, "unknown")

# From llama_cpp/_llama_cpp_completion_client.py
def normalize_name(name: str) -> str:
    """
    LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

    Prefer _assert_valid_name for validating user configuration or input
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]

# From llama_cpp/_llama_cpp_completion_client.py
def assert_valid_name(name: str) -> str:
    """
    Ensure that configured names are valid, raises ValueError if not.

    For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(f"Invalid name: {name}. Only letters, numbers, '_' and '-' are allowed.")
    if len(name) > 64:
        raise ValueError(f"Invalid name: {name}. Name must be less than 64 characters.")
    return name

# From llama_cpp/_llama_cpp_completion_client.py
def convert_tools(
    tools: Sequence[Tool | ToolSchema],
) -> List[ChatCompletionTool]:
    result: List[ChatCompletionTool] = []
    for tool in tools:
        if isinstance(tool, Tool):
            tool_schema = tool.schema
        else:
            assert isinstance(tool, dict)
            tool_schema = tool

        result.append(
            ChatCompletionTool(
                type="function",
                function=ChatCompletionToolFunction(
                    name=tool_schema["name"],
                    description=(tool_schema["description"] if "description" in tool_schema else ""),
                    parameters=(
                        cast(ChatCompletionFunctionParameters, tool_schema["parameters"])
                        if "parameters" in tool_schema
                        else {}
                    ),
                ),
            )
        )
    # Check if all tools have valid names.
    for tool_param in result:
        assert_valid_name(tool_param["function"]["name"])
    return result

# From llama_cpp/_llama_cpp_completion_client.py
def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_usage.get("prompt_tokens", 0),
            completion_tokens=self._total_usage.get("completion_tokens", 0),
        )

# From llama_cpp/_llama_cpp_completion_client.py
def capabilities(self) -> ModelInfo:
        return self.model_info

# From llama_cpp/_llama_cpp_completion_client.py
def count_tokens(
        self,
        messages: Sequence[SystemMessage | UserMessage | AssistantMessage | FunctionExecutionResultMessage],
        **kwargs: Any,
    ) -> int:
        total = 0
        for msg in messages:
            # Use the Llama model's tokenizer to encode the content
            tokens = self.llm.tokenize(str(msg.content).encode("utf-8"))
            total += len(tokens)
        return total

# From llama_cpp/_llama_cpp_completion_client.py
def model_info(self) -> ModelInfo:
        return self._model_info

# From llama_cpp/_llama_cpp_completion_client.py
def remaining_tokens(
        self,
        messages: Sequence[SystemMessage | UserMessage | AssistantMessage | FunctionExecutionResultMessage],
        **kwargs: Any,
    ) -> int:
        used_tokens = self.count_tokens(messages)
        return max(self.llm.n_ctx() - used_tokens, 0)

# From llama_cpp/_llama_cpp_completion_client.py
def total_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_usage.get("prompt_tokens", 0),
            completion_tokens=self._total_usage.get("completion_tokens", 0),
        )

import inspect
import math
import os
from asyncio import Task
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Callable
from typing import Set
import tiktoken
from autogen_core import TRACE_LOGGER_NAME
from autogen_core import Component
from autogen_core.logging import LLMStreamEndEvent
from autogen_core.logging import LLMStreamStartEvent
from autogen_core.models import ChatCompletionTokenLogprob
from autogen_core.models import ModelCapabilities
from autogen_core.models import TopLogprob
from openai import NOT_GIVEN
from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionRole
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat import ParsedChatCompletion
from openai.types.chat import ParsedChoice
from openai.types.chat import completion_create_params
from openai.types.chat.chat_completion import Choice
from openai.types.shared_params import FunctionDefinition
from openai.types.shared_params import FunctionParameters
from openai.types.shared_params import ResponseFormatJSONObject
from openai.types.shared_params import ResponseFormatText
from pydantic import SecretStr
from typing_extensions import Self
from _utils.normalize_stop_reason import normalize_stop_reason
from _utils.parse_r1_content import parse_r1_content
from  import _model_info
from _transformation import get_transformer
from _utils import assert_valid_name
from config import AzureOpenAIClientConfiguration
from config import AzureOpenAIClientConfigurationConfigModel
from config import OpenAIClientConfiguration
from config import OpenAIClientConfigurationConfigModel
from auth.azure import AzureTokenProvider

# From openai/_openai_client.py
class CreateParams:
    messages: List[ChatCompletionMessageParam]
    tools: List[ChatCompletionToolParam]
    response_format: Optional[Type[BaseModel]]
    create_args: Dict[str, Any]

# From openai/_openai_client.py
class BaseOpenAIChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        *,
        create_args: Dict[str, Any],
        model_capabilities: Optional[ModelCapabilities] = None,  # type: ignore
        model_info: Optional[ModelInfo] = None,
        add_name_prefixes: bool = False,
        include_name_in_message: bool = True,
    ):
        self._client = client
        self._add_name_prefixes = add_name_prefixes
        self._include_name_in_message = include_name_in_message
        if model_capabilities is None and model_info is None:
            try:
                self._model_info = _model_info.get_info(create_args["model"])
            except KeyError as err:
                raise ValueError("model_info is required when model name is not a valid OpenAI model") from err
        elif model_capabilities is not None and model_info is not None:
            raise ValueError("model_capabilities and model_info are mutually exclusive")
        elif model_capabilities is not None and model_info is None:
            warnings.warn(
                "model_capabilities is deprecated, use model_info instead",
                DeprecationWarning,
                stacklevel=2,
            )
            info = cast(ModelInfo, model_capabilities)
            info["family"] = ModelFamily.UNKNOWN
            self._model_info = info
        elif model_capabilities is None and model_info is not None:
            self._model_info = model_info

        # Validate model_info, check if all required fields are present
        validate_model_info(self._model_info)

        self._resolved_model: Optional[str] = None
        if "model" in create_args:
            self._resolved_model = _model_info.resolve_model(create_args["model"])

        if (
            not self._model_info["json_output"]
            and "response_format" in create_args
            and (
                isinstance(create_args["response_format"], dict)
                and create_args["response_format"]["type"] == "json_object"
            )
        ):
            raise ValueError("Model does not support JSON output.")

        self._create_args = create_args
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> ChatCompletionClient:
        return OpenAIChatCompletionClient(**config)

    def _rstrip_last_assistant_message(self, messages: Sequence[LLMMessage]) -> Sequence[LLMMessage]:
        """
        Remove the last assistant message if it is empty.
        """
        # When Claude models last message is AssistantMessage, It could not end with whitespace
        if isinstance(messages[-1], AssistantMessage):
            if isinstance(messages[-1].content, str):
                messages[-1].content = messages[-1].content.rstrip()

        return messages

    def _process_create_args(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        tool_choice: Tool | Literal["auto", "required", "none"],
        json_output: Optional[bool | type[BaseModel]],
        extra_create_args: Mapping[str, Any],
    ) -> CreateParams:
        # Make sure all extra_create_args are valid
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")

        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        # The response format value to use for the beta client.
        response_format_value: Optional[Type[BaseModel]] = None

        if "response_format" in create_args:
            # Legacy support for getting beta client mode from response_format.
            value = create_args["response_format"]
            if isinstance(value, type) and issubclass(value, BaseModel):
                if self.model_info["structured_output"] is False:
                    raise ValueError("Model does not support structured output.")
                warnings.warn(
                    "Using response_format to specify the BaseModel for structured output type will be deprecated. "
                    "Use json_output in create and create_stream instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                response_format_value = value
                # Remove response_format from create_args to prevent passing it twice.
                del create_args["response_format"]
            # In all other cases when response_format is set to something else, we will
            # use the regular client.

        if json_output is not None:
            if self.model_info["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output.")
            if json_output is True:
                # JSON mode.
                create_args["response_format"] = ResponseFormatJSONObject(type="json_object")
            elif json_output is False:
                # Text mode.
                create_args["response_format"] = ResponseFormatText(type="text")
            elif isinstance(json_output, type) and issubclass(json_output, BaseModel):
                if self.model_info["structured_output"] is False:
                    raise ValueError("Model does not support structured output.")
                if response_format_value is not None:
                    raise ValueError(
                        "response_format and json_output cannot be set to a Pydantic model class at the same time."
                    )
                # Beta client mode with Pydantic model class.
                response_format_value = json_output
            else:
                raise ValueError(f"json_output must be a boolean or a Pydantic model class, got {type(json_output)}")

        if response_format_value is not None and "response_format" in create_args:
            warnings.warn(
                "response_format is found in extra_create_args while json_output is set to a Pydantic model class. "
                "Skipping the response_format in extra_create_args in favor of the json_output. "
                "Structured output will be used.",
                UserWarning,
                stacklevel=2,
            )
            # If using beta client, remove response_format from create_args to prevent passing it twice
            del create_args["response_format"]

        # TODO: allow custom handling.
        # For now we raise an error if images are present and vision is not supported
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if self.model_info["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output.")

        if not self.model_info.get("multiple_system_messages", False):
            # Some models accept only one system message(or, it will read only the last one)
            # So, merge system messages into one (if multiple and continuous)
            system_message_content = ""
            _messages: List[LLMMessage] = []
            _first_system_message_idx = -1
            _last_system_message_idx = -1
            # Index of the first system message for adding the merged system message at the correct position
            for idx, message in enumerate(messages):
                if isinstance(message, SystemMessage):
                    if _first_system_message_idx == -1:
                        _first_system_message_idx = idx
                    elif _last_system_message_idx + 1 != idx:
                        # That case, system message is not continuous
                        # Merge system messages only contiues system messages
                        raise ValueError(
                            "Multiple and Not continuous system messages are not supported if model_info['multiple_system_messages'] is False"
                        )
                    system_message_content += message.content + "\n"
                    _last_system_message_idx = idx
                else:
                    _messages.append(message)
            system_message_content = system_message_content.rstrip()
            if system_message_content != "":
                system_message = SystemMessage(content=system_message_content)
                _messages.insert(_first_system_message_idx, system_message)
            messages = _messages

        # in that case, for ad-hoc, we using startswith instead of model_family for code consistency
        if create_args.get("model", "unknown").startswith("claude-"):
            # When Claude models last message is AssistantMessage, It could not end with whitespace
            messages = self._rstrip_last_assistant_message(messages)

        oai_messages_nested = [
            to_oai_type(
                m,
                prepend_name=self._add_name_prefixes,
                model=create_args.get("model", "unknown"),
                model_family=self._model_info["family"],
                include_name_in_message=self._include_name_in_message,
            )
            for m in messages
        ]

        oai_messages = [item for sublist in oai_messages_nested for item in sublist]

        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

        converted_tools = convert_tools(tools)

        # Process tool_choice parameter
        if isinstance(tool_choice, Tool):
            if len(tools) == 0:
                raise ValueError("tool_choice specified but no tools provided")

            # Validate that the tool exists in the provided tools
            tool_names_available: List[str] = []
            for tool in tools:
                if isinstance(tool, Tool):
                    tool_names_available.append(tool.schema["name"])
                else:
                    tool_names_available.append(tool["name"])

            # tool_choice is a single Tool object
            tool_name = tool_choice.schema["name"]
            if tool_name not in tool_names_available:
                raise ValueError(f"tool_choice references '{tool_name}' but it's not in the provided tools")

        if len(converted_tools) > 0:
            # Convert to OpenAI format and add to create_args
            converted_tool_choice = convert_tool_choice(tool_choice)
            create_args["tool_choice"] = converted_tool_choice

        return CreateParams(
            messages=oai_messages,
            tools=converted_tools,
            response_format=response_format_value,
            create_args=create_args,
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        create_params = self._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )
        future: Union[Task[ParsedChatCompletion[BaseModel]], Task[ChatCompletion]]
        if create_params.response_format is not None:
            # Use beta client if response_format is not None
            future = asyncio.ensure_future(
                self._client.beta.chat.completions.parse(
                    messages=create_params.messages,
                    tools=(create_params.tools if len(create_params.tools) > 0 else NOT_GIVEN),
                    response_format=create_params.response_format,
                    **create_params.create_args,
                )
            )
        else:
            # Use the regular client
            future = asyncio.ensure_future(
                self._client.chat.completions.create(
                    messages=create_params.messages,
                    stream=False,
                    tools=(create_params.tools if len(create_params.tools) > 0 else NOT_GIVEN),
                    **create_params.create_args,
                )
            )

        if cancellation_token is not None:
            cancellation_token.link_future(future)
        result: Union[ParsedChatCompletion[BaseModel], ChatCompletion] = await future
        if create_params.response_format is not None:
            result = cast(ParsedChatCompletion[Any], result)

        # Handle the case where OpenAI API might return None for token counts
        # even when result.usage is not None
        usage = RequestUsage(
            # TODO backup token counting
            prompt_tokens=getattr(result.usage, "prompt_tokens", 0) if result.usage is not None else 0,
            completion_tokens=getattr(result.usage, "completion_tokens", 0) if result.usage is not None else 0,
        )

        logger.info(
            LLMCallEvent(
                messages=cast(List[Dict[str, Any]], create_params.messages),
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                tools=create_params.tools,
            )
        )

        if self._resolved_model is not None:
            if self._resolved_model != result.model:
                warnings.warn(
                    f"Resolved model mismatch: {self._resolved_model} != {result.model}. "
                    "Model mapping in autogen_ext.models.openai may be incorrect. "
                    f"Set the model to {result.model} to enhance token/cost estimation and suppress this warning.",
                    stacklevel=2,
                )

        # Limited to a single choice currently.
        choice: Union[ParsedChoice[Any], ParsedChoice[BaseModel], Choice] = result.choices[0]

        # Detect whether it is a function call or not.
        # We don't rely on choice.finish_reason as it is not always accurate, depending on the API used.
        content: Union[str, List[FunctionCall]]
        thought: str | None = None
        if choice.message.function_call is not None:
            raise ValueError("function_call is deprecated and is not supported by this model client.")
        elif choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0:
            if choice.finish_reason != "tool_calls":
                warnings.warn(
                    f"Finish reason mismatch: {choice.finish_reason} != tool_calls "
                    "when tool_calls are present. Finish reason may not be accurate. "
                    "This may be due to the API used that is not returning the correct finish reason.",
                    stacklevel=2,
                )
            if choice.message.content is not None and choice.message.content != "":
                # Put the content in the thought field.
                thought = choice.message.content
            # NOTE: If OAI response type changes, this will need to be updated
            content = []
            for tool_call in choice.message.tool_calls:
                if not isinstance(tool_call.function.arguments, str):
                    warnings.warn(
                        f"Tool call function arguments field is not a string: {tool_call.function.arguments}."
                        "This is unexpected and may due to the API used not returning the correct type. "
                        "Attempting to convert it to string.",
                        stacklevel=2,
                    )
                    if isinstance(tool_call.function.arguments, dict):
                        tool_call.function.arguments = json.dumps(tool_call.function.arguments)
                content.append(
                    FunctionCall(
                        id=tool_call.id,
                        arguments=tool_call.function.arguments,
                        name=normalize_name(tool_call.function.name),
                    )
                )
            finish_reason = "tool_calls"
        else:
            # if not tool_calls, then it is a text response and we populate the content and thought fields.
            finish_reason = choice.finish_reason
            content = choice.message.content or ""
            # if there is a reasoning_content field, then we populate the thought field. This is for models such as R1 - direct from deepseek api.
            if choice.message.model_extra is not None:
                reasoning_content = choice.message.model_extra.get("reasoning_content")
                if reasoning_content is not None:
                    thought = reasoning_content

        logprobs: Optional[List[ChatCompletionTokenLogprob]] = None
        if choice.logprobs and choice.logprobs.content:
            logprobs = [
                ChatCompletionTokenLogprob(
                    token=x.token,
                    logprob=x.logprob,
                    top_logprobs=[TopLogprob(logprob=y.logprob, bytes=y.bytes) for y in x.top_logprobs],
                    bytes=x.bytes,
                )
                for x in choice.logprobs.content
            ]

        #   This is for local R1 models.
        if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1 and thought is None:
            thought, content = parse_r1_content(content)

        response = CreateResult(
            finish_reason=normalize_stop_reason(finish_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=logprobs,
            thought=thought,
        )

        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        # TODO - why is this cast needed?
        return response

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
        include_usage: Optional[bool] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a stream of string chunks from the model ending with a :class:`~autogen_core.models.CreateResult`.

        Extends :meth:`autogen_core.models.ChatCompletionClient.create_stream` to support OpenAI API.

        In streaming, the default behaviour is not return token usage counts.
        See: `OpenAI API reference for possible args <https://platform.openai.com/docs/api-reference/chat/create>`_.

        You can set set the `include_usage` flag to True or `extra_create_args={"stream_options": {"include_usage": True}}`. If both the flag and `stream_options` are set, but to different values, an exception will be raised.
        (if supported by the accessed API) to
        return a final chunk with usage set to a :class:`~autogen_core.models.RequestUsage` object
        with prompt and completion token counts,
        all preceding chunks will have usage as `None`.
        See: `OpenAI API reference for stream options <https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options>`_.

        Other examples of supported arguments that can be included in `extra_create_args`:
            - `temperature` (float): Controls the randomness of the output. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic.
            - `max_tokens` (int): The maximum number of tokens to generate in the completion.
            - `top_p` (float): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            - `frequency_penalty` (float): A value between -2.0 and 2.0 that penalizes new tokens based on their existing frequency in the text so far, decreasing the likelihood of repeated phrases.
            - `presence_penalty` (float): A value between -2.0 and 2.0 that penalizes new tokens based on whether they appear in the text so far, encouraging the model to talk about new topics.
        """

        create_params = self._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )

        if include_usage is not None:
            if "stream_options" in create_params.create_args:
                stream_options = create_params.create_args["stream_options"]
                if "include_usage" in stream_options and stream_options["include_usage"] != include_usage:
                    raise ValueError(
                        "include_usage and extra_create_args['stream_options']['include_usage'] are both set, but differ in value."
                    )
            else:
                # If stream options are not present, add them.
                create_params.create_args["stream_options"] = {"include_usage": True}

        if max_consecutive_empty_chunk_tolerance != 0:
            warnings.warn(
                "The 'max_consecutive_empty_chunk_tolerance' parameter is deprecated and will be removed in the future releases. All of empty chunks will be skipped with a warning.",
                DeprecationWarning,
                stacklevel=2,
            )

        if create_params.response_format is not None:
            chunks = self._create_stream_chunks_beta_client(
                tool_params=create_params.tools,
                oai_messages=create_params.messages,
                response_format=create_params.response_format,
                create_args_no_response_format=create_params.create_args,
                cancellation_token=cancellation_token,
            )
        else:
            chunks = self._create_stream_chunks(
                tool_params=create_params.tools,
                oai_messages=create_params.messages,
                create_args=create_params.create_args,
                cancellation_token=cancellation_token,
            )

        # Prepare data to process streaming chunks.
        chunk: ChatCompletionChunk | None = None
        stop_reason = None
        maybe_model = None
        content_deltas: List[str] = []
        thought_deltas: List[str] = []
        full_tool_calls: Dict[int, FunctionCall] = {}
        logprobs: Optional[List[ChatCompletionTokenLogprob]] = None

        empty_chunk_warning_has_been_issued: bool = False
        empty_chunk_warning_threshold: int = 10
        empty_chunk_count = 0
        first_chunk = True
        is_reasoning = False

        # Process the stream of chunks.
        async for chunk in chunks:
            if first_chunk:
                first_chunk = False
                # Emit the start event.
                logger.info(
                    LLMStreamStartEvent(
                        messages=cast(List[Dict[str, Any]], create_params.messages),
                    )
                )

            # Set the model from the lastest chunk.
            maybe_model = chunk.model

            # Empty chunks has been observed when the endpoint is under heavy load.
            #  https://github.com/microsoft/autogen/issues/4213
            if len(chunk.choices) == 0:
                empty_chunk_count += 1
                if not empty_chunk_warning_has_been_issued and empty_chunk_count >= empty_chunk_warning_threshold:
                    empty_chunk_warning_has_been_issued = True
                    warnings.warn(
                        f"Received more than {empty_chunk_warning_threshold} consecutive empty chunks. Empty chunks are being ignored.",
                        stacklevel=2,
                    )
                continue
            else:
                empty_chunk_count = 0

            if len(chunk.choices) > 1:
                # This is a multi-choice chunk, we need to warn the user.
                warnings.warn(
                    f"Received a chunk with {len(chunk.choices)} choices. Only the first choice will be used.",
                    UserWarning,
                    stacklevel=2,
                )

            # Set the choice to the first choice in the chunk.
            choice = chunk.choices[0]

            # for liteLLM chunk usage, do the following hack keeping the pervious chunk.stop_reason (if set).
            # set the stop_reason for the usage chunk to the prior stop_reason
            stop_reason = choice.finish_reason if chunk.usage is None and stop_reason is None else stop_reason
            maybe_model = chunk.model

            reasoning_content: str | None = None
            if choice.delta.model_extra is not None and "reasoning_content" in choice.delta.model_extra:
                # If there is a reasoning_content field, then we populate the thought field. This is for models such as R1.
                reasoning_content = choice.delta.model_extra.get("reasoning_content")

            if isinstance(reasoning_content, str) and len(reasoning_content) > 0:
                if not is_reasoning:
                    # Enter reasoning mode.
                    reasoning_content = "<think>" + reasoning_content
                    is_reasoning = True
                thought_deltas.append(reasoning_content)
                yield reasoning_content
            elif is_reasoning:
                # Exit reasoning mode.
                reasoning_content = "</think>"
                thought_deltas.append(reasoning_content)
                is_reasoning = False
                yield reasoning_content

            # First try get content
            if choice.delta.content:
                content_deltas.append(choice.delta.content)
                if len(choice.delta.content) > 0:
                    yield choice.delta.content
                # NOTE: for OpenAI, tool_calls and content are mutually exclusive it seems, so we can skip the rest of the loop.
                # However, this may not be the case for other APIs -- we should expect this may need to be updated.
                continue
            # Otherwise, get tool calls
            if choice.delta.tool_calls is not None:
                for tool_call_chunk in choice.delta.tool_calls:
                    idx = tool_call_chunk.index
                    if idx not in full_tool_calls:
                        # We ignore the type hint here because we want to fill in type when the delta provides it
                        full_tool_calls[idx] = FunctionCall(id="", arguments="", name="")

                    if tool_call_chunk.id is not None:
                        full_tool_calls[idx].id += tool_call_chunk.id

                    if tool_call_chunk.function is not None:
                        if tool_call_chunk.function.name is not None:
                            full_tool_calls[idx].name += tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments is not None:
                            full_tool_calls[idx].arguments += tool_call_chunk.function.arguments
            if choice.logprobs and choice.logprobs.content:
                logprobs = [
                    ChatCompletionTokenLogprob(
                        token=x.token,
                        logprob=x.logprob,
                        top_logprobs=[TopLogprob(logprob=y.logprob, bytes=y.bytes) for y in x.top_logprobs],
                        bytes=x.bytes,
                    )
                    for x in choice.logprobs.content
                ]

        # Finalize the CreateResult.

        # TODO: can we remove this?
        if stop_reason == "function_call":
            raise ValueError("Function calls are not supported in this context")

        # We need to get the model from the last chunk, if available.
        model = maybe_model or create_params.create_args["model"]
        model = model.replace("gpt-35", "gpt-3.5")  # hack for Azure API

        # Because the usage chunk is not guaranteed to be the last chunk, we need to check if it is available.
        if chunk and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        else:
            prompt_tokens = 0
            completion_tokens = 0
        usage = RequestUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Detect whether it is a function call or just text.
        content: Union[str, List[FunctionCall]]
        thought: str | None = None
        # Determine the content and thought based on what was collected
        if full_tool_calls:
            # This is a tool call response
            content = list(full_tool_calls.values())
            if content_deltas:
                # Store any text alongside tool calls as thoughts
                thought = "".join(content_deltas)
        else:
            # This is a text response (possibly with thoughts)
            if content_deltas:
                content = "".join(content_deltas)
            else:
                warnings.warn(
                    "No text content or tool calls are available. Model returned empty result.",
                    stacklevel=2,
                )
                content = ""

            # Set thoughts if we have any reasoning content.
            if thought_deltas:
                thought = "".join(thought_deltas).lstrip("<think>").rstrip("</think>")

            # This is for local R1 models whose reasoning content is within the content string.
            if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1 and thought is None:
                thought, content = parse_r1_content(content)

        # Create the result.
        result = CreateResult(
            finish_reason=normalize_stop_reason(stop_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=logprobs,
            thought=thought,
        )

        # Log the end of the stream.
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        # Update the total usage.
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        # Yield the CreateResult.
        yield result

    async def _create_stream_chunks(
        self,
        tool_params: List[ChatCompletionToolParam],
        oai_messages: List[ChatCompletionMessageParam],
        create_args: Dict[str, Any],
        cancellation_token: Optional[CancellationToken],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream_future = asyncio.ensure_future(
            self._client.chat.completions.create(
                messages=oai_messages,
                stream=True,
                tools=tool_params if len(tool_params) > 0 else NOT_GIVEN,
                **create_args,
            )
        )
        if cancellation_token is not None:
            cancellation_token.link_future(stream_future)
        stream = await stream_future
        while True:
            try:
                chunk_future = asyncio.ensure_future(anext(stream))
                if cancellation_token is not None:
                    cancellation_token.link_future(chunk_future)
                chunk = await chunk_future
                yield chunk
            except StopAsyncIteration:
                break

    async def _create_stream_chunks_beta_client(
        self,
        tool_params: List[ChatCompletionToolParam],
        oai_messages: List[ChatCompletionMessageParam],
        create_args_no_response_format: Dict[str, Any],
        response_format: Optional[Type[BaseModel]],
        cancellation_token: Optional[CancellationToken],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        async with self._client.beta.chat.completions.stream(
            messages=oai_messages,
            tools=tool_params if len(tool_params) > 0 else NOT_GIVEN,
            response_format=(response_format if response_format is not None else NOT_GIVEN),
            **create_args_no_response_format,
        ) as stream:
            while True:
                try:
                    event_future = asyncio.ensure_future(anext(stream))
                    if cancellation_token is not None:
                        cancellation_token.link_future(event_future)
                    event = await event_future

                    if event.type == "chunk":
                        chunk = event.chunk
                        yield chunk
                    # We don't handle other event types from the beta client stream.
                    # As the other event types are auxiliary to the chunk event.
                    # See: https://github.com/openai/openai-python/blob/main/helpers.md#chat-completions-events.
                    # Once the beta client is stable, we can move all the logic to the beta client.
                    # Then we can consider handling other event types which may simplify the code overall.
                except StopAsyncIteration:
                    break

    async def close(self) -> None:
        await self._client.close()

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return count_tokens_openai(
            messages,
            self._create_args["model"],
            add_name_prefixes=self._add_name_prefixes,
            tools=tools,
            model_family=self._model_info["family"],
            include_name_in_message=self._include_name_in_message,
        )

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        token_limit = _model_info.get_token_limit(self._create_args["model"])
        return token_limit - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn(
            "capabilities is deprecated, use model_info instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

# From openai/_openai_client.py
class OpenAIChatCompletionClient(BaseOpenAIChatCompletionClient, Component[OpenAIClientConfigurationConfigModel]):
    """Chat completion client for OpenAI hosted models.

    To use this client, you must install the `openai` extra:

    .. code-block:: bash

        pip install "autogen-ext[openai]"

    You can also use this client for OpenAI-compatible ChatCompletion endpoints.
    **Using this client for non-OpenAI models is not tested or guaranteed.**

    For non-OpenAI models, please first take a look at our `community extensions <https://microsoft.github.io/autogen/dev/user-guide/extensions-user-guide/index.html>`_
    for additional model clients.

    Args:
        model (str): Which OpenAI model to use.
        api_key (optional, str): The API key to use. **Required if 'OPENAI_API_KEY' is not found in the environment variables.**
        organization (optional, str): The organization ID to use.
        base_url (optional, str): The base URL to use. **Required if the model is not hosted on OpenAI.**
        timeout: (optional, float): The timeout for the request in seconds.
        max_retries (optional, int): The maximum number of retries to attempt.
        model_info (optional, ModelInfo): The capabilities of the model. **Required if the model name is not a valid OpenAI model.**
        frequency_penalty (optional, float):
        logit_bias: (optional, dict[str, int]):
        max_tokens (optional, int):
        n (optional, int):
        presence_penalty (optional, float):
        response_format (optional, Dict[str, Any]): the format of the response. Possible options are:

            .. code-block:: text

                # Text response, this is the default.
                {"type": "text"}

            .. code-block:: text

                # JSON response, make sure to instruct the model to return JSON.
                {"type": "json_object"}

            .. code-block:: text

                # Structured output response, with a pre-defined JSON schema.
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "name of the schema, must be an identifier.",
                        "description": "description for the model.",
                        # You can convert a Pydantic (v2) model to JSON schema
                        # using the `model_json_schema()` method.
                        "schema": "<the JSON schema itself>",
                        # Whether to enable strict schema adherence when
                        # generating the output. If set to true, the model will
                        # always follow the exact schema defined in the
                        # `schema` field. Only a subset of JSON Schema is
                        # supported when `strict` is `true`.
                        # To learn more, read
                        # https://platform.openai.com/docs/guides/structured-outputs.
                        "strict": False,  # or True
                    },
                }

            It is recommended to use the `json_output` parameter in
            :meth:`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create` or
            :meth:`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create_stream`
            methods instead of `response_format` for structured output.
            The `json_output` parameter is more flexible and allows you to
            specify a Pydantic model class directly.

        seed (optional, int):
        stop (optional, str | List[str]):
        temperature (optional, float):
        top_p (optional, float):
        parallel_tool_calls (optional, bool): Whether to allow parallel tool calls. When not set, defaults to server behavior.
        user (optional, str):
        default_headers (optional, dict[str, str]):  Custom headers; useful for authentication or other custom requirements.
        add_name_prefixes (optional, bool): Whether to prepend the `source` value
            to each :class:`~autogen_core.models.UserMessage` content. E.g.,
            "this is content" becomes "Reviewer said: this is content."
            This can be useful for models that do not support the `name` field in
            message. Defaults to False.
        include_name_in_message (optional, bool): Whether to include the `name` field
            in user message parameters sent to the OpenAI API. Defaults to True. Set to False
            for model providers that don't support the `name` field (e.g., Groq).
        stream_options (optional, dict): Additional options for streaming. Currently only `include_usage` is supported.

    Examples:

        The following code snippet shows how to use the client with an OpenAI model:

        .. code-block:: python

            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_core.models import UserMessage

            openai_client = OpenAIChatCompletionClient(
                model="gpt-4o-2024-08-06",
                # api_key="sk-...", # Optional if you have an OPENAI_API_KEY environment variable set.
            )

            result = await openai_client.create([UserMessage(content="What is the capital of France?", source="user")])  # type: ignore
            print(result)

            # Close the client when done.
            # await openai_client.close()

        To use the client with a non-OpenAI model, you need to provide the base URL of the model and the model info.
        For example, to use Ollama, you can use the following code snippet:

        .. code-block:: python

            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_core.models import ModelFamily

            custom_model_client = OpenAIChatCompletionClient(
                model="deepseek-r1:1.5b",
                base_url="http://localhost:11434/v1",
                api_key="placeholder",
                model_info={
                    "vision": False,
                    "function_calling": False,
                    "json_output": False,
                    "family": ModelFamily.R1,
                    "structured_output": True,
                },
            )

            # Close the client when done.
            # await custom_model_client.close()

        To use streaming mode, you can use the following code snippet:

        .. code-block:: python

            import asyncio
            from autogen_core.models import UserMessage
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                # Similar for AzureOpenAIChatCompletionClient.
                model_client = OpenAIChatCompletionClient(model="gpt-4o")  # assuming OPENAI_API_KEY is set in the environment.

                messages = [UserMessage(content="Write a very short story about a dragon.", source="user")]

                # Create a stream.
                stream = model_client.create_stream(messages=messages)

                # Iterate over the stream and print the responses.
                print("Streamed responses:")
                async for response in stream:
                    if isinstance(response, str):
                        # A partial response is a string.
                        print(response, flush=True, end="")
                    else:
                        # The last response is a CreateResult object with the complete message.
                        print("\\n\\n------------\\n")
                        print("The complete response:", flush=True)
                        print(response.content, flush=True)

                # Close the client when done.
                await model_client.close()


            asyncio.run(main())

        To use structured output as well as function calling, you can use the following code snippet:

        .. code-block:: python

            import asyncio
            from typing import Literal

            from autogen_core.models import (
                AssistantMessage,
                FunctionExecutionResult,
                FunctionExecutionResultMessage,
                SystemMessage,
                UserMessage,
            )
            from autogen_core.tools import FunctionTool
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from pydantic import BaseModel


            # Define the structured output format.
            class AgentResponse(BaseModel):
                thoughts: str
                response: Literal["happy", "sad", "neutral"]


            # Define the function to be called as a tool.
            def sentiment_analysis(text: str) -> str:
                \"\"\"Given a text, return the sentiment.\"\"\"
                return "happy" if "happy" in text else "sad" if "sad" in text else "neutral"


            # Create a FunctionTool instance with `strict=True`,
            # which is required for structured output mode.
            tool = FunctionTool(sentiment_analysis, description="Sentiment Analysis", strict=True)


            async def main() -> None:
                # Create an OpenAIChatCompletionClient instance.
                model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

                # Generate a response using the tool.
                response1 = await model_client.create(
                    messages=[
                        SystemMessage(content="Analyze input text sentiment using the tool provided."),
                        UserMessage(content="I am happy.", source="user"),
                    ],
                    tools=[tool],
                )
                print(response1.content)
                # Should be a list of tool calls.
                # [FunctionCall(name="sentiment_analysis", arguments={"text": "I am happy."}, ...)]

                assert isinstance(response1.content, list)
                response2 = await model_client.create(
                    messages=[
                        SystemMessage(content="Analyze input text sentiment using the tool provided."),
                        UserMessage(content="I am happy.", source="user"),
                        AssistantMessage(content=response1.content, source="assistant"),
                        FunctionExecutionResultMessage(
                            content=[FunctionExecutionResult(content="happy", call_id=response1.content[0].id, is_error=False, name="sentiment_analysis")]
                        ),
                    ],
                    # Use the structured output format.
                    json_output=AgentResponse,
                )
                print(response2.content)
                # Should be a structured output.
                # {"thoughts": "The user is happy.", "response": "happy"}

                # Close the client when done.
                await model_client.close()

            asyncio.run(main())


        To load the client from a configuration, you can use the `load_component` method:

        .. code-block:: python

            from autogen_core.models import ChatCompletionClient

            config = {
                "provider": "OpenAIChatCompletionClient",
                "config": {"model": "gpt-4o", "api_key": "REPLACE_WITH_YOUR_API_KEY"},
            }

            client = ChatCompletionClient.load_component(config)

        To view the full list of available configuration options, see the :py:class:`OpenAIClientConfigurationConfigModel` class.

    """

    component_type = "model"
    component_config_schema = OpenAIClientConfigurationConfigModel
    component_provider_override = "autogen_ext.models.openai.OpenAIChatCompletionClient"

    def __init__(self, **kwargs: Unpack[OpenAIClientConfiguration]):
        if "model" not in kwargs:
            raise ValueError("model is required for OpenAIChatCompletionClient")

        model_capabilities: Optional[ModelCapabilities] = None  # type: ignore
        self._raw_config: Dict[str, Any] = dict(kwargs).copy()
        copied_args = dict(kwargs).copy()

        if "model_capabilities" in kwargs:
            model_capabilities = kwargs["model_capabilities"]
            del copied_args["model_capabilities"]

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
            del copied_args["model_info"]

        add_name_prefixes: bool = False
        if "add_name_prefixes" in kwargs:
            add_name_prefixes = kwargs["add_name_prefixes"]

        include_name_in_message: bool = True
        if "include_name_in_message" in kwargs:
            include_name_in_message = kwargs["include_name_in_message"]

        # Special handling for Gemini model.
        assert "model" in copied_args and isinstance(copied_args["model"], str)
        if copied_args["model"].startswith("gemini-"):
            if "base_url" not in copied_args:
                copied_args["base_url"] = _model_info.GEMINI_OPENAI_BASE_URL
            if "api_key" not in copied_args and "GEMINI_API_KEY" in os.environ:
                copied_args["api_key"] = os.environ["GEMINI_API_KEY"]
        if copied_args["model"].startswith("claude-"):
            if "base_url" not in copied_args:
                copied_args["base_url"] = _model_info.ANTHROPIC_OPENAI_BASE_URL
            if "api_key" not in copied_args and "ANTHROPIC_API_KEY" in os.environ:
                copied_args["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        if copied_args["model"].startswith("Llama-"):
            if "base_url" not in copied_args:
                copied_args["base_url"] = _model_info.LLAMA_API_BASE_URL
            if "api_key" not in copied_args and "LLAMA_API_KEY" in os.environ:
                copied_args["api_key"] = os.environ["LLAMA_API_KEY"]

        client = _openai_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)

        super().__init__(
            client=client,
            create_args=create_args,
            model_capabilities=model_capabilities,
            model_info=model_info,
            add_name_prefixes=add_name_prefixes,
            include_name_in_message=include_name_in_message,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _openai_client_from_config(state["_raw_config"])

    def _to_config(self) -> OpenAIClientConfigurationConfigModel:
        copied_config = self._raw_config.copy()
        return OpenAIClientConfigurationConfigModel(**copied_config)

    @classmethod
    def _from_config(cls, config: OpenAIClientConfigurationConfigModel) -> Self:
        copied_config = config.model_copy().model_dump(exclude_none=True)

        # Handle api_key as SecretStr
        if "api_key" in copied_config and isinstance(config.api_key, SecretStr):
            copied_config["api_key"] = config.api_key.get_secret_value()

        return cls(**copied_config)

# From openai/_openai_client.py
class AzureOpenAIChatCompletionClient(
    BaseOpenAIChatCompletionClient, Component[AzureOpenAIClientConfigurationConfigModel]
):
    """Chat completion client for Azure OpenAI hosted models.

    To use this client, you must install the `azure` and `openai` extensions:

    .. code-block:: bash

        pip install "autogen-ext[openai,azure]"

    Args:

        model (str): Which OpenAI model to use.
        azure_endpoint (str): The endpoint for the Azure model. **Required for Azure models.**
        azure_deployment (str): Deployment name for the Azure model. **Required for Azure models.**
        api_version (str): The API version to use. **Required for Azure models.**
        azure_ad_token (str): The Azure AD token to use. Provide this or `azure_ad_token_provider` for token-based authentication.
        azure_ad_token_provider (optional, Callable[[], Awaitable[str]] | AzureTokenProvider): The Azure AD token provider to use. Provide this or `azure_ad_token` for token-based authentication.
        api_key (optional, str): The API key to use, use this if you are using key based authentication. It is optional if you are using Azure AD token based authentication or `AZURE_OPENAI_API_KEY` environment variable.
        timeout: (optional, float): The timeout for the request in seconds.
        max_retries (optional, int): The maximum number of retries to attempt.
        model_info (optional, ModelInfo): The capabilities of the model. **Required if the model name is not a valid OpenAI model.**
        frequency_penalty (optional, float):
        logit_bias: (optional, dict[str, int]):
        max_tokens (optional, int):
        n (optional, int):
        presence_penalty (optional, float):
        response_format (optional, Dict[str, Any]): the format of the response. Possible options are:

            .. code-block:: text

                # Text response, this is the default.
                {"type": "text"}

            .. code-block:: text

                # JSON response, make sure to instruct the model to return JSON.
                {"type": "json_object"}

            .. code-block:: text

                # Structured output response, with a pre-defined JSON schema.
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "name of the schema, must be an identifier.",
                        "description": "description for the model.",
                        # You can convert a Pydantic (v2) model to JSON schema
                        # using the `model_json_schema()` method.
                        "schema": "<the JSON schema itself>",
                        # Whether to enable strict schema adherence when
                        # generating the output. If set to true, the model will
                        # always follow the exact schema defined in the
                        # `schema` field. Only a subset of JSON Schema is
                        # supported when `strict` is `true`.
                        # To learn more, read
                        # https://platform.openai.com/docs/guides/structured-outputs.
                        "strict": False,  # or True
                    },
                }

            It is recommended to use the `json_output` parameter in
            :meth:`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create` or
            :meth:`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create_stream`
            methods instead of `response_format` for structured output.
            The `json_output` parameter is more flexible and allows you to
            specify a Pydantic model class directly.

        seed (optional, int):
        stop (optional, str | List[str]):
        temperature (optional, float):
        top_p (optional, float):
        parallel_tool_calls (optional, bool): Whether to allow parallel tool calls. When not set, defaults to server behavior.
        user (optional, str):
        default_headers (optional, dict[str, str]):  Custom headers; useful for authentication or other custom requirements.
        add_name_prefixes (optional, bool): Whether to prepend the `source` value
            to each :class:`~autogen_core.models.UserMessage` content. E.g.,
            "this is content" becomes "Reviewer said: this is content."
            This can be useful for models that do not support the `name` field in
            message. Defaults to False.
        include_name_in_message (optional, bool): Whether to include the `name` field
            in user message parameters sent to the OpenAI API. Defaults to True. Set to False
            for model providers that don't support the `name` field (e.g., Groq).
        stream_options (optional, dict): Additional options for streaming. Currently only `include_usage` is supported.


    To use the client, you need to provide your deployment name, Azure Cognitive Services endpoint, and api version.
    For authentication, you can either provide an API key or an Azure Active Directory (AAD) token credential.

    The following code snippet shows how to use AAD authentication.
    The identity used must be assigned the `Cognitive Services OpenAI User <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/role-based-access-control#cognitive-services-openai-user>`_ role.

    .. code-block:: python

        from autogen_ext.auth.azure import AzureTokenProvider
        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
        from azure.identity import DefaultAzureCredential

        # Create the token provider
        token_provider = AzureTokenProvider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )

        az_model_client = AzureOpenAIChatCompletionClient(
            azure_deployment="{your-azure-deployment}",
            model="{model-name, such as gpt-4o}",
            api_version="2024-06-01",
            azure_endpoint="https://{your-custom-endpoint}.openai.azure.com/",
            azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
            # api_key="sk-...", # For key-based authentication.
        )

    See other usage examples in the :class:`OpenAIChatCompletionClient` class.

    To load the client that uses identity based aith from a configuration, you can use the `load_component` method:

    .. code-block:: python

        from autogen_core.models import ChatCompletionClient

        config = {
            "provider": "AzureOpenAIChatCompletionClient",
            "config": {
                "model": "gpt-4o-2024-05-13",
                "azure_endpoint": "https://{your-custom-endpoint}.openai.azure.com/",
                "azure_deployment": "{your-azure-deployment}",
                "api_version": "2024-06-01",
                "azure_ad_token_provider": {
                    "provider": "autogen_ext.auth.azure.AzureTokenProvider",
                    "config": {
                        "provider_kind": "DefaultAzureCredential",
                        "scopes": ["https://cognitiveservices.azure.com/.default"],
                    },
                },
            },
        }

        client = ChatCompletionClient.load_component(config)


    To view the full list of available configuration options, see the :py:class:`AzureOpenAIClientConfigurationConfigModel` class.

    .. note::

        Right now only `DefaultAzureCredential` is supported with no additional args passed to it.

    .. note::

        The Azure OpenAI client by default sets the User-Agent header to `autogen-python/{version}`. To override this, you can set the variable `autogen_ext.models.openai.AZURE_OPENAI_USER_AGENT` environment variable to an empty string.

    See `here <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity#chat-completions>`_ for how to use the Azure client directly or for more info.

    """

    component_type = "model"
    component_config_schema = AzureOpenAIClientConfigurationConfigModel
    component_provider_override = "autogen_ext.models.openai.AzureOpenAIChatCompletionClient"

    def __init__(self, **kwargs: Unpack[AzureOpenAIClientConfiguration]):
        model_capabilities: Optional[ModelCapabilities] = None  # type: ignore
        copied_args = dict(kwargs).copy()
        if "model_capabilities" in kwargs:
            model_capabilities = kwargs["model_capabilities"]
            del copied_args["model_capabilities"]

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
            del copied_args["model_info"]

        add_name_prefixes: bool = False
        if "add_name_prefixes" in kwargs:
            add_name_prefixes = kwargs["add_name_prefixes"]

        include_name_in_message: bool = True
        if "include_name_in_message" in kwargs:
            include_name_in_message = kwargs["include_name_in_message"]

        client = _azure_openai_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)
        self._raw_config: Dict[str, Any] = copied_args
        super().__init__(
            client=client,
            create_args=create_args,
            model_capabilities=model_capabilities,
            model_info=model_info,
            add_name_prefixes=add_name_prefixes,
            include_name_in_message=include_name_in_message,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _azure_openai_client_from_config(state["_raw_config"])

    def _to_config(self) -> AzureOpenAIClientConfigurationConfigModel:
        from ...auth.azure import AzureTokenProvider

        copied_config = self._raw_config.copy()
        if "azure_ad_token_provider" in copied_config:
            if not isinstance(copied_config["azure_ad_token_provider"], AzureTokenProvider):
                raise ValueError("azure_ad_token_provider must be a AzureTokenProvider to be component serialized")

            copied_config["azure_ad_token_provider"] = (
                copied_config["azure_ad_token_provider"].dump_component().model_dump(exclude_none=True)
            )

        return AzureOpenAIClientConfigurationConfigModel(**copied_config)

    @classmethod
    def _from_config(cls, config: AzureOpenAIClientConfigurationConfigModel) -> Self:
        from ...auth.azure import AzureTokenProvider

        copied_config = config.model_copy().model_dump(exclude_none=True)

        # Handle api_key as SecretStr
        if "api_key" in copied_config and isinstance(config.api_key, SecretStr):
            copied_config["api_key"] = config.api_key.get_secret_value()

        if "azure_ad_token_provider" in copied_config:
            copied_config["azure_ad_token_provider"] = AzureTokenProvider.load_component(
                copied_config["azure_ad_token_provider"]
            )

        return cls(**copied_config)

# From openai/_openai_client.py
def type_to_role(message: LLMMessage) -> ChatCompletionRole:
    if isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, UserMessage):
        return "user"
    elif isinstance(message, AssistantMessage):
        return "assistant"
    else:
        return "tool"

# From openai/_openai_client.py
def to_oai_type(
    message: LLMMessage,
    prepend_name: bool = False,
    model: str = "unknown",
    model_family: str = ModelFamily.UNKNOWN,
    include_name_in_message: bool = True,
) -> Sequence[ChatCompletionMessageParam]:
    context = {
        "prepend_name": prepend_name,
        "include_name_in_message": include_name_in_message,
    }
    transformers = get_transformer("openai", model, model_family)

    def raise_value_error(message: LLMMessage, context: Dict[str, Any]) -> Sequence[ChatCompletionMessageParam]:
        raise ValueError(f"Unknown message type: {type(message)}")

    transformer: Callable[[LLMMessage, Dict[str, Any]], Sequence[ChatCompletionMessageParam]] = transformers.get(
        type(message), raise_value_error
    )
    result = transformer(message, context)
    return result

# From openai/_openai_client.py
def calculate_vision_tokens(image: Image, detail: str = "auto") -> int:
    MAX_LONG_EDGE = 2048
    BASE_TOKEN_COUNT = 85
    TOKENS_PER_TILE = 170
    MAX_SHORT_EDGE = 768
    TILE_SIZE = 512

    if detail == "low":
        return BASE_TOKEN_COUNT

    width, height = image.image.size

    # Scale down to fit within a MAX_LONG_EDGE x MAX_LONG_EDGE square if necessary

    if width > MAX_LONG_EDGE or height > MAX_LONG_EDGE:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            # Width is greater than height
            width = MAX_LONG_EDGE
            height = int(MAX_LONG_EDGE / aspect_ratio)
        else:
            # Height is greater than or equal to width
            height = MAX_LONG_EDGE
            width = int(MAX_LONG_EDGE * aspect_ratio)

    # Resize such that the shortest side is MAX_SHORT_EDGE if both dimensions exceed MAX_SHORT_EDGE
    aspect_ratio = width / height
    if width > MAX_SHORT_EDGE and height > MAX_SHORT_EDGE:
        if aspect_ratio > 1:
            # Width is greater than height
            height = MAX_SHORT_EDGE
            width = int(MAX_SHORT_EDGE * aspect_ratio)
        else:
            # Height is greater than or equal to width
            width = MAX_SHORT_EDGE
            height = int(MAX_SHORT_EDGE / aspect_ratio)

    # Calculate the number of tiles based on TILE_SIZE

    tiles_width = math.ceil(width / TILE_SIZE)
    tiles_height = math.ceil(height / TILE_SIZE)
    total_tiles = tiles_width * tiles_height
    # Calculate the total tokens based on the number of tiles and the base token count

    total_tokens = BASE_TOKEN_COUNT + TOKENS_PER_TILE * total_tiles

    return total_tokens

# From openai/_openai_client.py
def convert_tool_choice(tool_choice: Tool | Literal["auto", "required", "none"]) -> Any:
    """Convert tool_choice parameter to OpenAI API format.

    Args:
        tool_choice: A single Tool object to force the model to use, "auto" to let the model choose any available tool, "required" to force tool usage, or "none" to disable tool usage.

    Returns:
        OpenAI API compatible tool_choice value or None if not specified.
    """
    if tool_choice == "none":
        return "none"

    if tool_choice == "auto":
        return "auto"

    if tool_choice == "required":
        return "required"

    # Must be a Tool object
    if isinstance(tool_choice, Tool):
        return {"type": "function", "function": {"name": tool_choice.schema["name"]}}
    else:
        raise ValueError(f"tool_choice must be a Tool object, 'auto', 'required', or 'none', got {type(tool_choice)}")

# From openai/_openai_client.py
def count_tokens_openai(
    messages: Sequence[LLMMessage],
    model: str,
    *,
    add_name_prefixes: bool = False,
    tools: Sequence[Tool | ToolSchema] = [],
    model_family: str = ModelFamily.UNKNOWN,
    include_name_in_message: bool = True,
) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        trace_logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0

    # Message tokens.
    for message in messages:
        num_tokens += tokens_per_message
        oai_message = to_oai_type(
            message,
            prepend_name=add_name_prefixes,
            model=model,
            model_family=model_family,
            include_name_in_message=include_name_in_message,
        )
        for oai_message_part in oai_message:
            for key, value in oai_message_part.items():
                if value is None:
                    continue

                if isinstance(message, UserMessage) and isinstance(value, list):
                    typed_message_value = cast(List[ChatCompletionContentPartParam], value)

                    assert len(typed_message_value) == len(
                        message.content
                    ), "Mismatch in message content and typed message value"

                    # We need image properties that are only in the original message
                    for part, content_part in zip(typed_message_value, message.content, strict=False):
                        if isinstance(content_part, Image):
                            # TODO: add detail parameter
                            num_tokens += calculate_vision_tokens(content_part)
                        elif isinstance(part, str):
                            num_tokens += len(encoding.encode(part))
                        else:
                            try:
                                serialized_part = json.dumps(part)
                                num_tokens += len(encoding.encode(serialized_part))
                            except TypeError:
                                trace_logger.warning(f"Could not convert {part} to string, skipping.")
                else:
                    if not isinstance(value, str):
                        try:
                            value = json.dumps(value)
                        except TypeError:
                            trace_logger.warning(f"Could not convert {value} to string, skipping.")
                            continue
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    # Tool tokens.
    oai_tools = convert_tools(tools)
    for tool in oai_tools:
        function = tool["function"]
        tool_tokens = len(encoding.encode(function["name"]))
        if "description" in function:
            tool_tokens += len(encoding.encode(function["description"]))
        tool_tokens -= 2
        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                assert isinstance(parameters["properties"], dict)
                for propertiesKey in parameters["properties"]:  # pyright: ignore
                    assert isinstance(propertiesKey, str)
                    tool_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]  # pyright: ignore
                    for field in v:  # pyright: ignore
                        if field == "type":
                            tool_tokens += 2
                            tool_tokens += len(encoding.encode(v["type"]))  # pyright: ignore
                        elif field == "description":
                            tool_tokens += 2
                            tool_tokens += len(encoding.encode(v["description"]))  # pyright: ignore
                        elif field == "enum":
                            tool_tokens -= 3
                            for o in v["enum"]:  # pyright: ignore
                                tool_tokens += 3
                                tool_tokens += len(encoding.encode(o))  # pyright: ignore
                        else:
                            trace_logger.warning(f"Not supported field {field}")
                tool_tokens += 11
                if len(parameters["properties"]) == 0:  # pyright: ignore
                    tool_tokens -= 2
        num_tokens += tool_tokens
    num_tokens += 12
    return num_tokens

# From openai/_openai_client.py
def raise_value_error(message: LLMMessage, context: Dict[str, Any]) -> Sequence[ChatCompletionMessageParam]:
        raise ValueError(f"Unknown message type: {type(message)}")

# From openai/_openai_client.py
def create_from_config(cls, config: Dict[str, Any]) -> ChatCompletionClient:
        return OpenAIChatCompletionClient(**config)

from inspect import getfullargspec
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage
from azure.ai.inference.models import ChatCompletions
from azure.ai.inference.models import ChatCompletionsNamedToolChoice
from azure.ai.inference.models import ChatCompletionsNamedToolChoiceFunction
from azure.ai.inference.models import ChatCompletionsToolCall
from azure.ai.inference.models import ChatCompletionsToolDefinition
from azure.ai.inference.models import CompletionsFinishReason
from azure.ai.inference.models import ContentItem
from azure.ai.inference.models import FunctionDefinition
from azure.ai.inference.models import ImageContentItem
from azure.ai.inference.models import ImageDetailLevel
from azure.ai.inference.models import ImageUrl
from azure.ai.inference.models import StreamingChatChoiceUpdate
from azure.ai.inference.models import StreamingChatCompletionsUpdate
from azure.ai.inference.models import TextContentItem
from azure.ai.inference.models import FunctionCall
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import ToolMessage
from azure.ai.inference.models import UserMessage
from typing_extensions import AsyncGenerator
from autogen_ext.models.azure.config import GITHUB_MODELS_ENDPOINT
from autogen_ext.models.azure.config import AzureAIChatCompletionClientConfig

# From azure/_azure_ai_client.py
class AzureAIChatCompletionClient(ChatCompletionClient):
    """
    Chat completion client for models hosted on Azure AI Foundry or GitHub Models.
    See `here <https://learn.microsoft.com/en-us/azure/ai-studio/reference/reference-model-inference-chat-completions>`_ for more info.

    Args:
        endpoint (str): The endpoint to use. **Required.**
        credential (union, AzureKeyCredential, AsyncTokenCredential): The credentials to use. **Required**
        model_info (ModelInfo): The model family and capabilities of the model. **Required.**
        model (str): The name of the model. **Required if model is hosted on GitHub Models.**
        frequency_penalty: (optional,float)
        presence_penalty: (optional,float)
        temperature: (optional,float)
        top_p: (optional,float)
        max_tokens: (optional,int)
        response_format: (optional, literal["text", "json_object"])
        stop: (optional,List[str])
        tools: (optional,List[ChatCompletionsToolDefinition])
        tool_choice: (optional,Union[str, ChatCompletionsToolChoicePreset, ChatCompletionsNamedToolChoice]])
        seed: (optional,int)
        model_extras: (optional,Dict[str, Any])

    To use this client, you must install the `azure` extra:

    .. code-block:: bash

        pip install "autogen-ext[azure]"

    The following code snippet shows how to use the client with GitHub Models:

    .. code-block:: python

        import asyncio
        import os
        from azure.core.credentials import AzureKeyCredential
        from autogen_ext.models.azure import AzureAIChatCompletionClient
        from autogen_core.models import UserMessage


        async def main():
            client = AzureAIChatCompletionClient(
                model="Phi-4",
                endpoint="https://models.github.ai/inference",
                # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
                # Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
                credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
                model_info={
                    "json_output": False,
                    "function_calling": False,
                    "vision": False,
                    "family": "unknown",
                    "structured_output": False,
                },
            )

            result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
            print(result)

            # Close the client.
            await client.close()


        if __name__ == "__main__":
            asyncio.run(main())

    To use streaming, you can use the `create_stream` method:

    .. code-block:: python

        import asyncio
        import os

        from autogen_core.models import UserMessage
        from autogen_ext.models.azure import AzureAIChatCompletionClient
        from azure.core.credentials import AzureKeyCredential


        async def main():
            client = AzureAIChatCompletionClient(
                model="Phi-4",
                endpoint="https://models.github.ai/inference",
                # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
                # Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
                credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
                model_info={
                    "json_output": False,
                    "function_calling": False,
                    "vision": False,
                    "family": "unknown",
                    "structured_output": False,
                },
            )

            # Create a stream.
            stream = client.create_stream([UserMessage(content="Write a poem about the ocean", source="user")])
            async for chunk in stream:
                print(chunk, end="", flush=True)
            print()

            # Close the client.
            await client.close()


        if __name__ == "__main__":
            asyncio.run(main())


    """

    def __init__(self, **kwargs: Unpack[AzureAIChatCompletionClientConfig]):
        config = self._validate_config(kwargs)  # type: ignore
        self._model_info = config["model_info"]  # type: ignore
        self._client = self._create_client(config)
        self._create_args = self._prepare_create_args(config)

        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> AzureAIChatCompletionClientConfig:
        if "endpoint" not in config:
            raise ValueError("endpoint is required for AzureAIChatCompletionClient")
        if "credential" not in config:
            raise ValueError("credential is required for AzureAIChatCompletionClient")
        if "model_info" not in config:
            raise ValueError("model_info is required for AzureAIChatCompletionClient")
        validate_model_info(config["model_info"])
        if _is_github_model(config["endpoint"]) and "model" not in config:
            raise ValueError("model is required for when using a Github model with AzureAIChatCompletionClient")
        return cast(AzureAIChatCompletionClientConfig, config)

    @staticmethod
    def _create_client(config: AzureAIChatCompletionClientConfig) -> ChatCompletionsClient:
        # Only pass the parameters that ChatCompletionsClient accepts
        # Remove 'model_info' and other client-specific parameters
        client_config = {k: v for k, v in config.items() if k not in ("model_info",)}
        return ChatCompletionsClient(**client_config)  # type: ignore

    @staticmethod
    def _prepare_create_args(config: Mapping[str, Any]) -> Dict[str, Any]:
        create_args = {k: v for k, v in config.items() if k in create_kwargs}
        return create_args

    def add_usage(self, usage: RequestUsage) -> None:
        self._total_usage = RequestUsage(
            self._total_usage.prompt_tokens + usage.prompt_tokens,
            self._total_usage.completion_tokens + usage.completion_tokens,
        )

    def _validate_model_info(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        json_output: Optional[bool | type[BaseModel]],
        create_args: Dict[str, Any],
    ) -> None:
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if json_output is not None:
            if self.model_info["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output")

            if isinstance(json_output, type):
                # TODO: we should support this in the future.
                raise ValueError("Structured output is not currently supported for AzureAIChatCompletionClient")

            if json_output is True and "response_format" not in create_args:
                create_args["response_format"] = "json_object"

        if self.model_info["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output")
        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")

        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        self._validate_model_info(messages, tools, json_output, create_args)

        azure_messages_nested = [to_azure_message(msg) for msg in messages]
        azure_messages = [item for sublist in azure_messages_nested for item in sublist]

        task: Task[ChatCompletions]

        if len(tools) > 0:
            if isinstance(tool_choice, Tool):
                create_args["tool_choice"] = ChatCompletionsNamedToolChoice(
                    function=ChatCompletionsNamedToolChoiceFunction(name=tool_choice.name)
                )
            else:
                create_args["tool_choice"] = tool_choice
            converted_tools = convert_tools(tools)
            task = asyncio.create_task(  # type: ignore
                self._client.complete(messages=azure_messages, tools=converted_tools, **create_args)  # type: ignore
            )
        else:
            task = asyncio.create_task(  # type: ignore
                self._client.complete(  # type: ignore
                    messages=azure_messages,
                    **create_args,
                )
            )

        if cancellation_token is not None:
            cancellation_token.link_future(task)

        result: ChatCompletions = await task

        usage = RequestUsage(
            prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
            completion_tokens=result.usage.completion_tokens if result.usage else 0,
        )

        logger.info(
            LLMCallEvent(
                messages=[m.as_dict() for m in azure_messages],
                response=result.as_dict(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        choice = result.choices[0]
        thought = None

        if choice.finish_reason == CompletionsFinishReason.TOOL_CALLS:
            assert choice.message.tool_calls is not None
            content: Union[str, List[FunctionCall]] = [
                FunctionCall(
                    id=x.id,
                    arguments=x.function.arguments,
                    name=normalize_name(x.function.name),
                )
                for x in choice.message.tool_calls
            ]
            finish_reason = "function_calls"

            if choice.message.content:
                thought = choice.message.content
        else:
            if isinstance(choice.finish_reason, CompletionsFinishReason):
                finish_reason = choice.finish_reason.value
            else:
                finish_reason = choice.finish_reason  # type: ignore
            content = choice.message.content or ""

        if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1:
            thought, content = parse_r1_content(content)

        response = CreateResult(
            finish_reason=finish_reason,  # type: ignore
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )

        self.add_usage(usage)

        return response

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")

        create_args: Dict[str, Any] = self._create_args.copy()
        create_args.update(extra_create_args)

        self._validate_model_info(messages, tools, json_output, create_args)

        # azure_messages = [to_azure_message(m) for m in messages]
        azure_messages_nested = [to_azure_message(msg) for msg in messages]
        azure_messages = [item for sublist in azure_messages_nested for item in sublist]

        if len(tools) > 0:
            if isinstance(tool_choice, Tool):
                create_args["tool_choice"] = ChatCompletionsNamedToolChoice(
                    function=ChatCompletionsNamedToolChoiceFunction(name=tool_choice.name)
                )
            else:
                create_args["tool_choice"] = tool_choice
            converted_tools = convert_tools(tools)
            task = asyncio.create_task(
                self._client.complete(messages=azure_messages, tools=converted_tools, stream=True, **create_args)
            )
        else:
            task = asyncio.create_task(self._client.complete(messages=azure_messages, stream=True, **create_args))

        if cancellation_token is not None:
            cancellation_token.link_future(task)

        # result: ChatCompletions = await task
        finish_reason: Optional[FinishReasons] = None
        content_deltas: List[str] = []
        full_tool_calls: Dict[str, FunctionCall] = {}
        prompt_tokens = 0
        completion_tokens = 0
        chunk: Optional[StreamingChatCompletionsUpdate] = None
        choice: Optional[StreamingChatChoiceUpdate] = None
        first_chunk = True
        thought = None

        async for chunk in await task:  # type: ignore
            if first_chunk:
                first_chunk = False
                # Emit the start event.
                logger.info(
                    LLMStreamStartEvent(
                        messages=[m.as_dict() for m in azure_messages],
                    )
                )
            assert isinstance(chunk, StreamingChatCompletionsUpdate)
            choice = chunk.choices[0] if len(chunk.choices) > 0 else None
            if choice and choice.finish_reason is not None:
                if isinstance(choice.finish_reason, CompletionsFinishReason):
                    finish_reason = cast(FinishReasons, choice.finish_reason.value)
                else:
                    if choice.finish_reason in ["stop", "length", "function_calls", "content_filter", "unknown"]:
                        finish_reason = choice.finish_reason  # type: ignore
                    else:
                        raise ValueError(f"Unexpected finish reason: {choice.finish_reason}")

            # We first try to load the content
            if choice and choice.delta.content is not None:
                content_deltas.append(choice.delta.content)
                yield choice.delta.content
            # Otherwise, we try to load the tool calls
            if choice and choice.delta.tool_calls is not None:
                for tool_call_chunk in choice.delta.tool_calls:
                    # print(tool_call_chunk)
                    if "index" in tool_call_chunk:
                        idx = tool_call_chunk["index"]
                    else:
                        idx = tool_call_chunk.id
                    if idx not in full_tool_calls:
                        full_tool_calls[idx] = FunctionCall(id="", arguments="", name="")

                    full_tool_calls[idx].id += tool_call_chunk.id
                    full_tool_calls[idx].name += tool_call_chunk.function.name
                    full_tool_calls[idx].arguments += tool_call_chunk.function.arguments

        if chunk and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens

        if finish_reason is None:
            raise ValueError("No stop reason found")

        if choice and choice.finish_reason is CompletionsFinishReason.TOOL_CALLS:
            finish_reason = "function_calls"

        content: Union[str, List[FunctionCall]]

        if len(content_deltas) > 1:
            content = "".join(content_deltas)
            if chunk and chunk.usage:
                completion_tokens = chunk.usage.completion_tokens
            else:
                completion_tokens = 0
        else:
            content = list(full_tool_calls.values())

            if len(content_deltas) > 0:
                thought = "".join(content_deltas)

        usage = RequestUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )

        if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1:
            thought, content = parse_r1_content(content)

        result = CreateResult(
            finish_reason=finish_reason,
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )

        # Log the end of the stream.
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        self.add_usage(usage)

        yield result

    async def close(self) -> None:
        await self._client.close()

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

# From azure/_azure_ai_client.py
def to_azure_message(message: LLMMessage) -> Sequence[AzureMessage]:
    if isinstance(message, SystemMessage):
        return [_system_message_to_azure(message)]
    elif isinstance(message, UserMessage):
        return [_user_message_to_azure(message)]
    elif isinstance(message, AssistantMessage):
        return [_assistant_message_to_azure(message)]
    else:
        return _tool_message_to_azure(message)

# From azure/_azure_ai_client.py
def add_usage(self, usage: RequestUsage) -> None:
        self._total_usage = RequestUsage(
            self._total_usage.prompt_tokens + usage.prompt_tokens,
            self._total_usage.completion_tokens + usage.completion_tokens,
        )

from ollama import AsyncClient
from ollama import ChatResponse
from ollama import Message
from ollama import Image
from ollama import Tool
from ollama._types import ChatRequest
from pydantic.json_schema import JsonSchemaValue
from config import BaseOllamaClientConfiguration
from config import BaseOllamaClientConfigurationConfigModel

# From ollama/_ollama_client.py
class BaseOllamaChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        client: AsyncClient,
        *,
        create_args: Dict[str, Any],
        model_capabilities: Optional[ModelCapabilities] = None,  # type: ignore
        model_info: Optional[ModelInfo] = None,
    ):
        self._client = client
        self._model_name = create_args["model"]
        if model_capabilities is None and model_info is None:
            try:
                self._model_info = _model_info.get_info(create_args["model"])
            except KeyError as err:
                raise ValueError("model_info is required when model name is not a valid OpenAI model") from err
        elif model_capabilities is not None and model_info is not None:
            raise ValueError("model_capabilities and model_info are mutually exclusive")
        elif model_capabilities is not None and model_info is None:
            warnings.warn("model_capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
            info = cast(ModelInfo, model_capabilities)
            info["family"] = ModelFamily.UNKNOWN
            self._model_info = info
        elif model_capabilities is None and model_info is not None:
            self._model_info = model_info

        self._resolved_model: Optional[str] = None
        self._model_class: Optional[str] = None
        if "model" in create_args:
            self._resolved_model = create_args["model"]
            self._model_class = _model_info.resolve_model_class(create_args["model"])

        if (
            not self._model_info["json_output"]
            and "response_format" in create_args
            and (
                isinstance(create_args["response_format"], dict)
                and create_args["response_format"]["type"] == "json_object"
            )
        ):
            raise ValueError("Model does not support JSON output.")

        self._create_args = create_args
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        # Ollama doesn't have IDs for tools, so we just increment a counter
        self._tool_id = 0

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> ChatCompletionClient:
        return OllamaChatCompletionClient(**config)

    def get_create_args(self) -> Mapping[str, Any]:
        return self._create_args

    def _process_create_args(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        tool_choice: Tool | Literal["auto", "required", "none"],
        json_output: Optional[bool | type[BaseModel]],
        extra_create_args: Mapping[str, Any],
    ) -> CreateParams:
        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)
        create_args = _create_args_from_config(create_args)

        response_format_value: JsonSchemaValue | Literal["json"] | None = None

        if "response_format" in create_args:
            warnings.warn(
                "Using response_format will be deprecated. Use json_output instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            value = create_args["response_format"]
            if isinstance(value, type) and issubclass(value, BaseModel):
                response_format_value = value.model_json_schema()
                # Remove response_format from create_args to prevent passing it twice.
                del create_args["response_format"]
            else:
                raise ValueError(f"response_format must be a Pydantic model class, not {type(value)}")

        if json_output is not None:
            if self.model_info["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output.")
            if json_output is True:
                # JSON mode.
                response_format_value = "json"
            elif json_output is False:
                # Text mode.
                response_format_value = None
            elif isinstance(json_output, type) and issubclass(json_output, BaseModel):
                if response_format_value is not None:
                    raise ValueError(
                        "response_format and json_output cannot be set to a Pydantic model class at the same time. "
                        "Use json_output instead."
                    )
                # Beta client mode with Pydantic model class.
                response_format_value = json_output.model_json_schema()
            else:
                raise ValueError(f"json_output must be a boolean or a Pydantic model class, got {type(json_output)}")

        if "format" in create_args:
            # Handle the case where format is set from create_args.
            if json_output is not None:
                raise ValueError("json_output and format cannot be set at the same time. Use json_output instead.")
            assert response_format_value is None
            response_format_value = create_args["format"]
            # Remove format from create_args to prevent passing it twice.
            del create_args["format"]

        # TODO: allow custom handling.
        # For now we raise an error if images are present and vision is not supported
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        if self.model_info["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output.")

        ollama_messages_nested = [to_ollama_type(m) for m in messages]
        ollama_messages = [item for sublist in ollama_messages_nested for item in sublist]

        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling and tools were provided")

        converted_tools: List[OllamaTool] = []

        # Handle tool_choice parameter in a way that is compatible with Ollama API.
        if isinstance(tool_choice, Tool):
            # If tool_choice is a Tool, convert it to OllamaTool.
            converted_tools = convert_tools([tool_choice])
        elif tool_choice == "none":
            # No tool choice, do not pass tools to the API.
            converted_tools = []
        elif tool_choice == "required":
            # Required tool choice, pass tools to the API.
            converted_tools = convert_tools(tools)
            if len(converted_tools) == 0:
                raise ValueError("tool_choice 'required' specified but no tools provided")
        else:
            converted_tools = convert_tools(tools)

        return CreateParams(
            messages=ollama_messages,
            tools=converted_tools,
            format=response_format_value,
            create_args=create_args,
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # Make sure all extra_create_args are valid
        # TODO: kwarg checking logic
        # extra_create_args_keys = set(extra_create_args.keys())
        # if not create_kwargs.issuperset(extra_create_args_keys):
        #     raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")
        create_params = self._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )
        future = asyncio.ensure_future(
            self._client.chat(  # type: ignore
                # model=self._model_name,
                messages=create_params.messages,
                tools=create_params.tools if len(create_params.tools) > 0 else None,
                stream=False,
                format=create_params.format,
                **create_params.create_args,
            )
        )
        if cancellation_token is not None:
            cancellation_token.link_future(future)
        result: ChatResponse = await future

        usage = RequestUsage(
            # TODO backup token counting
            prompt_tokens=result.prompt_eval_count if result.prompt_eval_count is not None else 0,
            completion_tokens=(result.eval_count if result.eval_count is not None else 0),
        )

        logger.info(
            LLMCallEvent(
                messages=[m.model_dump() for m in create_params.messages],
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        if self._resolved_model is not None:
            if self._resolved_model != result.model:
                warnings.warn(
                    f"Resolved model mismatch: {self._resolved_model} != {result.model}. "
                    "Model mapping in autogen_ext.models.openai may be incorrect.",
                    stacklevel=2,
                )

        # Detect whether it is a function call or not.
        # We don't rely on choice.finish_reason as it is not always accurate, depending on the API used.
        content: Union[str, List[FunctionCall]]
        thought: Optional[str] = None
        if result.message.tool_calls is not None:
            if result.message.content is not None and result.message.content != "":
                thought = result.message.content
            # NOTE: If OAI response type changes, this will need to be updated
            content = [
                FunctionCall(
                    id=str(self._tool_id),
                    arguments=json.dumps(x.function.arguments),
                    name=normalize_name(x.function.name),
                )
                for x in result.message.tool_calls
            ]
            finish_reason = "tool_calls"
            self._tool_id += 1
        else:
            finish_reason = result.done_reason or ""
            content = result.message.content or ""

        # Ollama currently doesn't provide these.
        # Currently open ticket: https://github.com/ollama/ollama/issues/2415
        # logprobs: Optional[List[ChatCompletionTokenLogprob]] = None
        # if choice.logprobs and choice.logprobs.content:
        #     logprobs = [
        #         ChatCompletionTokenLogprob(
        #             token=x.token,
        #             logprob=x.logprob,
        #             top_logprobs=[TopLogprob(logprob=y.logprob, bytes=y.bytes) for y in x.top_logprobs],
        #             bytes=x.bytes,
        #         )
        #         for x in choice.logprobs.content
        #     ]
        response = CreateResult(
            finish_reason=normalize_stop_reason(finish_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=thought,
        )

        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        return response

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        # Make sure all extra_create_args are valid
        # TODO: kwarg checking logic
        # extra_create_args_keys = set(extra_create_args.keys())
        # if not create_kwargs.issuperset(extra_create_args_keys):
        #     raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")
        create_params = self._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )
        stream_future = asyncio.ensure_future(
            self._client.chat(  # type: ignore
                # model=self._model_name,
                messages=create_params.messages,
                tools=create_params.tools if len(create_params.tools) > 0 else None,
                stream=True,
                format=create_params.format,
                **create_params.create_args,
            )
        )
        if cancellation_token is not None:
            cancellation_token.link_future(stream_future)
        stream = await stream_future

        chunk = None
        stop_reason = None
        content_chunks: List[str] = []
        full_tool_calls: List[FunctionCall] = []
        completion_tokens = 0
        first_chunk = True
        while True:
            try:
                chunk_future = asyncio.ensure_future(anext(stream))
                if cancellation_token is not None:
                    cancellation_token.link_future(chunk_future)
                chunk = await chunk_future

                if first_chunk:
                    first_chunk = False
                    # Emit the start event.
                    logger.info(
                        LLMStreamStartEvent(
                            messages=[m.model_dump() for m in create_params.messages],
                        )
                    )
                # set the stop_reason for the usage chunk to the prior stop_reason
                stop_reason = chunk.done_reason if chunk.done and stop_reason is None else stop_reason
                # First try get content
                if chunk.message.content is not None:
                    content_chunks.append(chunk.message.content)
                    if len(chunk.message.content) > 0:
                        yield chunk.message.content

                # Get tool calls
                if chunk.message.tool_calls is not None:
                    full_tool_calls.extend(
                        [
                            FunctionCall(
                                id=str(self._tool_id),
                                arguments=json.dumps(x.function.arguments),
                                name=normalize_name(x.function.name),
                            )
                            for x in chunk.message.tool_calls
                        ]
                    )

                # TODO: logprobs currently unsupported in ollama.
                # See: https://github.com/ollama/ollama/issues/2415
                # if choice.logprobs and choice.logprobs.content:
                #     logprobs = [
                #         ChatCompletionTokenLogprob(
                #             token=x.token,
                #             logprob=x.logprob,
                #             top_logprobs=[TopLogprob(logprob=y.logprob, bytes=y.bytes) for y in x.top_logprobs],
                #             bytes=x.bytes,
                #         )
                #         for x in choice.logprobs.content
                #     ]

            except StopAsyncIteration:
                break

        if chunk and chunk.prompt_eval_count:
            prompt_tokens = chunk.prompt_eval_count
        else:
            prompt_tokens = 0

        content: Union[str, List[FunctionCall]]
        thought: Optional[str] = None

        if len(content_chunks) > 0 and len(full_tool_calls) > 0:
            content = full_tool_calls
            thought = "".join(content_chunks)
            if chunk and chunk.eval_count:
                completion_tokens = chunk.eval_count
            else:
                completion_tokens = 0
        elif len(content_chunks) > 1:
            content = "".join(content_chunks)
            if chunk and chunk.eval_count:
                completion_tokens = chunk.eval_count
            else:
                completion_tokens = 0
        else:
            completion_tokens = 0
            content = full_tool_calls

        usage = RequestUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        result = CreateResult(
            finish_reason=normalize_stop_reason(stop_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=thought,
        )

        # Emit the end event.
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        yield result

    async def close(self) -> None:
        pass  # ollama has no close method?

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return count_tokens_ollama(messages, self._create_args["model"], tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        token_limit = _model_info.get_token_limit(self._create_args["model"])
        return token_limit - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn("capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

# From ollama/_ollama_client.py
class OllamaChatCompletionClient(BaseOllamaChatCompletionClient, Component[BaseOllamaClientConfigurationConfigModel]):
    """Chat completion client for Ollama hosted models.

    Ollama must be installed and the appropriate model pulled.

    Args:
        model (str): Which Ollama model to use.
        host (optional, str): Model host url.
        response_format (optional, pydantic.BaseModel): The format of the response. If provided, the response will be parsed into this format as json.
        options (optional, Mapping[str, Any] | Options): Additional options to pass to the Ollama client.
        model_info (optional, ModelInfo): The capabilities of the model. **Required if the model is not listed in the ollama model info.**

    Note:
        Only models with 200k+ downloads (as of Jan 21, 2025), + phi4, deepseek-r1 have pre-defined model infos. See `this file <https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/models/ollama/_model_info.py>`__ for the full list. An entry for one model encompases all parameter variants of that model.

    To use this client, you must install the `ollama` extension:

    .. code-block:: bash

        pip install "autogen-ext[ollama]"

    The following code snippet shows how to use the client with an Ollama model:

    .. code-block:: python

        from autogen_ext.models.ollama import OllamaChatCompletionClient
        from autogen_core.models import UserMessage

        ollama_client = OllamaChatCompletionClient(
            model="llama3",
        )

        result = await ollama_client.create([UserMessage(content="What is the capital of France?", source="user")])  # type: ignore
        print(result)

    To load the client from a configuration, you can use the `load_component` method:

    .. code-block:: python

        from autogen_core.models import ChatCompletionClient

        config = {
            "provider": "OllamaChatCompletionClient",
            "config": {"model": "llama3"},
        }

        client = ChatCompletionClient.load_component(config)

    To output structured data, you can use the `response_format` argument:

    .. code-block:: python

        from autogen_ext.models.ollama import OllamaChatCompletionClient
        from autogen_core.models import UserMessage
        from pydantic import BaseModel


        class StructuredOutput(BaseModel):
            first_name: str
            last_name: str


        ollama_client = OllamaChatCompletionClient(
            model="llama3",
            response_format=StructuredOutput,
        )
        result = await ollama_client.create([UserMessage(content="Who was the first man on the moon?", source="user")])  # type: ignore
        print(result)

    Note:
        Tool usage in ollama is stricter than in its OpenAI counterparts. While OpenAI accepts a map of [str, Any], Ollama requires a map of [str, Property] where Property is a typed object containing ``type`` and ``description`` fields. Therefore, only the keys ``type`` and ``description`` will be converted from the properties blob in the tool schema.

    To view the full list of available configuration options, see the :py:class:`OllamaClientConfigurationConfigModel` class.

    """

    component_type = "model"
    component_config_schema = BaseOllamaClientConfigurationConfigModel
    component_provider_override = "autogen_ext.models.ollama.OllamaChatCompletionClient"

    def __init__(self, **kwargs: Unpack[BaseOllamaClientConfiguration]):
        if "model" not in kwargs:
            raise ValueError("model is required for OllamaChatCompletionClient")

        model_capabilities: Optional[ModelCapabilities] = None  # type: ignore
        copied_args = dict(kwargs).copy()
        if "model_capabilities" in kwargs:
            model_capabilities = kwargs["model_capabilities"]
            del copied_args["model_capabilities"]

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
            del copied_args["model_info"]

        client = _ollama_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)
        self._raw_config: Dict[str, Any] = copied_args
        super().__init__(
            client=client, create_args=create_args, model_capabilities=model_capabilities, model_info=model_info
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _ollama_client_from_config(state["_raw_config"])

    def _to_config(self) -> BaseOllamaClientConfigurationConfigModel:
        copied_config = self._raw_config.copy()
        return BaseOllamaClientConfigurationConfigModel(**copied_config)

    @classmethod
    def _from_config(cls, config: BaseOllamaClientConfigurationConfigModel) -> Self:
        copied_config = config.model_copy().model_dump(exclude_none=True)
        return cls(**copied_config)

# From ollama/_ollama_client.py
def user_message_to_ollama(message: UserMessage) -> Sequence[Message]:
    assert_valid_name(message.source)
    if isinstance(message.content, str):
        return [
            Message(
                content=message.content,
                role="user",
                # name=message.source, # TODO: No name parameter in Ollama
            )
        ]
    else:
        ollama_messages: List[Message] = []
        for part in message.content:
            if isinstance(part, str):
                ollama_messages.append(Message(content=part, role="user"))
            elif isinstance(part, Image):
                # TODO: should images go into their own message? Should each image get its own message?
                if not ollama_messages:
                    ollama_messages.append(Message(role="user", images=[OllamaImage(value=part.to_base64())]))
                else:
                    if ollama_messages[-1].images is None:
                        ollama_messages[-1].images = [OllamaImage(value=part.to_base64())]
                    else:
                        ollama_messages[-1].images.append(OllamaImage(value=part.to_base64()))  # type: ignore
            else:
                raise ValueError(f"Unknown content type: {part}")
        return ollama_messages

# From ollama/_ollama_client.py
def system_message_to_ollama(message: SystemMessage) -> Message:
    return Message(
        content=message.content,
        role="system",
    )

# From ollama/_ollama_client.py
def func_call_to_ollama(message: FunctionCall) -> Message.ToolCall:
    return Message.ToolCall(
        function=Message.ToolCall.Function(
            name=message.name,
            arguments=_func_args_to_ollama_args(message.arguments),
        )
    )

# From ollama/_ollama_client.py
def tool_message_to_ollama(
    message: FunctionExecutionResultMessage,
) -> Sequence[Message]:
    return [Message(content=x.content, role="tool") for x in message.content]

# From ollama/_ollama_client.py
def assistant_message_to_ollama(
    message: AssistantMessage,
) -> Message:
    assert_valid_name(message.source)
    if isinstance(message.content, list):
        return Message(
            tool_calls=[func_call_to_ollama(x) for x in message.content],
            role="assistant",
            # name=message.source,
        )
    else:
        return Message(
            content=message.content,
            role="assistant",
        )

# From ollama/_ollama_client.py
def to_ollama_type(message: LLMMessage) -> Sequence[Message]:
    if isinstance(message, SystemMessage):
        return [system_message_to_ollama(message)]
    elif isinstance(message, UserMessage):
        return user_message_to_ollama(message)
    elif isinstance(message, AssistantMessage):
        return [assistant_message_to_ollama(message)]
    else:
        return tool_message_to_ollama(message)

# From ollama/_ollama_client.py
def count_tokens_ollama(messages: Sequence[LLMMessage], model: str, *, tools: Sequence[Tool | ToolSchema] = []) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        trace_logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3
    num_tokens = 0

    # Message tokens.
    for message in messages:
        num_tokens += tokens_per_message
        ollama_message = to_ollama_type(message)
        for ollama_message_part in ollama_message:
            if isinstance(message.content, Image):
                num_tokens += calculate_vision_tokens(message.content)
            elif ollama_message_part.content is not None:
                num_tokens += len(encoding.encode(ollama_message_part.content))
    # TODO: every model family has its own message sequence.
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    # Tool tokens.
    ollama_tools = convert_tools(tools)
    for tool in ollama_tools:
        function = tool["function"]
        tool_tokens = len(encoding.encode(function["name"]))
        if "description" in function:
            tool_tokens += len(encoding.encode(function["description"]))
        tool_tokens -= 2
        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                assert isinstance(parameters["properties"], dict)
                for propertiesKey in parameters["properties"]:  # pyright: ignore
                    assert isinstance(propertiesKey, str)
                    tool_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]  # pyright: ignore
                    for field in v:  # pyright: ignore
                        if field == "type":
                            tool_tokens += 2
                            tool_tokens += len(encoding.encode(v["type"]))  # pyright: ignore
                        elif field == "description":
                            tool_tokens += 2
                            tool_tokens += len(encoding.encode(v["description"]))  # pyright: ignore
                        elif field == "enum":
                            tool_tokens -= 3
                            for o in v["enum"]:  # pyright: ignore
                                tool_tokens += 3
                                tool_tokens += len(encoding.encode(o))  # pyright: ignore
                        else:
                            trace_logger.warning(f"Not supported field {field}")
                tool_tokens += 11
                if len(parameters["properties"]) == 0:  # pyright: ignore
                    tool_tokens -= 2
        num_tokens += tool_tokens
    num_tokens += 12
    return num_tokens

# From ollama/_ollama_client.py
def get_create_args(self) -> Mapping[str, Any]:
        return self._create_args

from typing import Coroutine
from typing import Iterable
from typing import overload
from anthropic import AsyncAnthropic
from anthropic import AsyncAnthropicBedrock
from anthropic import AsyncStream
from anthropic.types import Base64ImageSourceParam
from anthropic.types import ContentBlock
from anthropic.types import ImageBlockParam
from anthropic.types import Message
from anthropic.types import MessageParam
from anthropic.types import RawMessageStreamEvent
from anthropic.types import TextBlock
from anthropic.types import TextBlockParam
from anthropic.types import ToolParam
from anthropic.types import ToolResultBlockParam
from anthropic.types import ToolUseBlock
from autogen_core.utils import extract_json_from_str
from config import AnthropicBedrockClientConfiguration
from config import AnthropicBedrockClientConfigurationConfigModel
from config import AnthropicClientConfiguration
from config import AnthropicClientConfigurationConfigModel
from config import BedrockInfo

# From anthropic/_anthropic_client.py
class BaseAnthropicChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        client: Any,
        *,
        create_args: Dict[str, Any],
        model_info: Optional[ModelInfo] = None,
    ):
        self._client = client

        if model_info is None:
            try:
                self._model_info = _model_info.get_info(create_args["model"])
            except KeyError as err:
                raise ValueError("model_info is required when model name is not recognized") from err
        else:
            self._model_info = model_info

        # Validate model_info
        validate_model_info(self._model_info)

        self._create_args = create_args
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    def _serialize_message(self, message: MessageParam) -> Dict[str, Any]:
        """Convert an Anthropic MessageParam to a JSON-serializable format."""
        if isinstance(message, dict):
            result: Dict[str, Any] = {}
            for key, value in message.items():
                if key == "content" and isinstance(value, list):
                    serialized_blocks: List[Any] = []
                    for block in value:  # type: ignore
                        if isinstance(block, BaseModel):
                            serialized_blocks.append(block.model_dump())
                        else:
                            serialized_blocks.append(block)
                    result[key] = serialized_blocks
                else:
                    result[key] = value
            return result
        else:
            return {"role": "unknown", "content": str(message)}

    def _merge_system_messages(self, messages: Sequence[LLMMessage]) -> Sequence[LLMMessage]:
        """
        Merge continuous system messages into a single message.
        """
        _messages: List[LLMMessage] = []
        system_message_content = ""
        _first_system_message_idx = -1
        _last_system_message_idx = -1
        # Index of the first system message for adding the merged system message at the correct position
        for idx, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                if _first_system_message_idx == -1:
                    _first_system_message_idx = idx
                elif _last_system_message_idx + 1 != idx:
                    # That case, system message is not continuous
                    # Merge system messages only contiues system messages
                    raise ValueError("Multiple and Not continuous system messages are not supported")
                system_message_content += message.content + "\n"
                _last_system_message_idx = idx
            else:
                _messages.append(message)
        system_message_content = system_message_content.rstrip()
        if system_message_content != "":
            system_message = SystemMessage(content=system_message_content)
            _messages.insert(_first_system_message_idx, system_message)
        messages = _messages

        return messages

    def _rstrip_last_assistant_message(self, messages: Sequence[LLMMessage]) -> Sequence[LLMMessage]:
        """
        Remove the last assistant message if it is empty.
        """
        # When Claude models last message is AssistantMessage, It could not end with whitespace
        if isinstance(messages[-1], AssistantMessage):
            if isinstance(messages[-1].content, str):
                messages[-1].content = messages[-1].content.rstrip()

        return messages

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # Copy create args and update with extra args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        # Check for vision capability if images are present
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        # Handle JSON output format
        if json_output is not None:
            if self.model_info["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output")

            if json_output is True:
                create_args["response_format"] = {"type": "json_object"}
            elif isinstance(json_output, type):
                raise ValueError("Structured output is currently not supported for Anthropic models")

        # Process system message separately
        system_message = None
        anthropic_messages: List[MessageParam] = []

        # Merge continuous system messages into a single message
        messages = self._merge_system_messages(messages)
        messages = self._rstrip_last_assistant_message(messages)

        for message in messages:
            if isinstance(message, SystemMessage):
                if system_message is not None:
                    # if that case, system message is must only one
                    raise ValueError("Multiple system messages are not supported")
                system_message = to_anthropic_type(message)
            else:
                anthropic_message = to_anthropic_type(message)
                if isinstance(anthropic_message, list):
                    anthropic_messages.extend(anthropic_message)
                elif isinstance(anthropic_message, str):
                    msg = MessageParam(
                        role="user" if isinstance(message, UserMessage) else "assistant", content=anthropic_message
                    )
                    anthropic_messages.append(msg)
                else:
                    anthropic_messages.append(anthropic_message)

        # Check for function calling support
        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

        # Set up the request
        request_args: Dict[str, Any] = {
            "model": create_args["model"],
            "messages": anthropic_messages,
            "max_tokens": create_args.get("max_tokens", 4096),
            "temperature": create_args.get("temperature", 1.0),
        }

        # Add system message if present
        if system_message is not None:
            request_args["system"] = system_message

        has_tool_results = any(isinstance(msg, FunctionExecutionResultMessage) for msg in messages)

        # Store and add tools if present
        if len(tools) > 0:
            converted_tools = convert_tools(tools)
            self._last_used_tools = converted_tools
            request_args["tools"] = converted_tools
        elif has_tool_results:
            # anthropic requires tools to be present even if there is any tool use
            request_args["tools"] = self._last_used_tools

        # Process tool_choice parameter
        if isinstance(tool_choice, Tool):
            if len(tools) == 0 and not has_tool_results:
                raise ValueError("tool_choice specified but no tools provided")

            # Validate that the tool exists in the provided tools
            tool_names_available: List[str] = []
            if len(tools) > 0:
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_names_available.append(tool.schema["name"])
                    else:
                        tool_names_available.append(tool["name"])
            else:
                # Use last used tools names if available
                for tool_param in self._last_used_tools:
                    tool_names_available.append(tool_param["name"])

            # tool_choice is a single Tool object
            tool_name = tool_choice.schema["name"]
            if tool_name not in tool_names_available:
                raise ValueError(f"tool_choice references '{tool_name}' but it's not in the available tools")

        # Convert to Anthropic format and add to request_args only if tools are provided
        # According to Anthropic API, tool_choice may only be specified while providing tools
        if len(tools) > 0 or has_tool_results:
            converted_tool_choice = convert_tool_choice_anthropic(tool_choice)
            if converted_tool_choice is not None:
                request_args["tool_choice"] = converted_tool_choice

        # Optional parameters
        for param in ["top_p", "top_k", "stop_sequences", "metadata"]:
            if param in create_args:
                request_args[param] = create_args[param]

        # Execute the request
        future: asyncio.Task[Message] = asyncio.ensure_future(self._client.messages.create(**request_args))  # type: ignore

        if cancellation_token is not None:
            cancellation_token.link_future(future)  # type: ignore

        result: Message = cast(Message, await future)  # type: ignore

        # Extract usage statistics
        usage = RequestUsage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
        )
        serializable_messages: List[Dict[str, Any]] = [self._serialize_message(msg) for msg in anthropic_messages]

        logger.info(
            LLMCallEvent(
                messages=serializable_messages,
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        # Process the response
        content: Union[str, List[FunctionCall]]
        thought = None

        # Check if the response includes tool uses
        tool_uses = [block for block in result.content if getattr(block, "type", None) == "tool_use"]

        if tool_uses:
            # Handle tool use response
            content = []

            # Check for text content that should be treated as thought
            text_blocks: List[TextBlock] = [block for block in result.content if isinstance(block, TextBlock)]
            if text_blocks:
                thought = "".join([block.text for block in text_blocks])

            # Process tool use blocks
            for tool_use in tool_uses:
                if isinstance(tool_use, ToolUseBlock):
                    tool_input = tool_use.input
                    if isinstance(tool_input, dict):
                        tool_input = json.dumps(tool_input)
                    else:
                        tool_input = str(tool_input) if tool_input is not None else ""

                    content.append(
                        FunctionCall(
                            id=tool_use.id,
                            name=normalize_name(tool_use.name),
                            arguments=tool_input,
                        )
                    )
        else:
            # Handle text response
            content = "".join([block.text if isinstance(block, TextBlock) else "" for block in result.content])

        # Create the final result
        response = CreateResult(
            finish_reason=normalize_stop_reason(result.stop_reason),
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )

        # Update usage statistics
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        return response

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """
        Creates an AsyncGenerator that yields a stream of completions based on the provided messages and tools.
        """
        # Copy create args and update with extra args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        # Check for vision capability if images are present
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")

        # Handle JSON output format
        if json_output is not None:
            if self.model_info["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output")

            if json_output is True:
                create_args["response_format"] = {"type": "json_object"}

            if isinstance(json_output, type):
                raise ValueError("Structured output is currently not supported for Anthropic models")

        # Process system message separately
        system_message = None
        anthropic_messages: List[MessageParam] = []

        # Merge continuous system messages into a single message
        messages = self._merge_system_messages(messages)
        messages = self._rstrip_last_assistant_message(messages)

        for message in messages:
            if isinstance(message, SystemMessage):
                if system_message is not None:
                    # if that case, system message is must only one
                    raise ValueError("Multiple system messages are not supported")
                system_message = to_anthropic_type(message)
            else:
                anthropic_message = to_anthropic_type(message)
                if isinstance(anthropic_message, list):
                    anthropic_messages.extend(anthropic_message)
                elif isinstance(anthropic_message, str):
                    msg = MessageParam(
                        role="user" if isinstance(message, UserMessage) else "assistant", content=anthropic_message
                    )
                    anthropic_messages.append(msg)
                else:
                    anthropic_messages.append(anthropic_message)

        # Check for function calling support
        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")

        # Set up the request
        request_args: Dict[str, Any] = {
            "model": create_args["model"],
            "messages": anthropic_messages,
            "max_tokens": create_args.get("max_tokens", 4096),
            "temperature": create_args.get("temperature", 1.0),
            "stream": True,
        }

        # Add system message if present
        if system_message is not None:
            request_args["system"] = system_message

        # Check if any message is a tool result
        has_tool_results = any(isinstance(msg, FunctionExecutionResultMessage) for msg in messages)

        # Add tools if present
        if len(tools) > 0:
            converted_tools = convert_tools(tools)
            self._last_used_tools = converted_tools
            request_args["tools"] = converted_tools
        elif has_tool_results:
            request_args["tools"] = self._last_used_tools

        # Process tool_choice parameter
        if isinstance(tool_choice, Tool):
            if len(tools) == 0 and not has_tool_results:
                raise ValueError("tool_choice specified but no tools provided")

            # Validate that the tool exists in the provided tools
            tool_names_available: List[str] = []
            if len(tools) > 0:
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_names_available.append(tool.schema["name"])
                    else:
                        tool_names_available.append(tool["name"])
            else:
                # Use last used tools names if available
                for last_used_tool in self._last_used_tools:
                    tool_names_available.append(last_used_tool["name"])

            # tool_choice is a single Tool object
            tool_name = tool_choice.schema["name"]
            if tool_name not in tool_names_available:
                raise ValueError(f"tool_choice references '{tool_name}' but it's not in the available tools")

        # Convert to Anthropic format and add to request_args only if tools are provided
        # According to Anthropic API, tool_choice may only be specified while providing tools
        if len(tools) > 0 or has_tool_results:
            converted_tool_choice = convert_tool_choice_anthropic(tool_choice)
            if converted_tool_choice is not None:
                request_args["tool_choice"] = converted_tool_choice

        # Optional parameters
        for param in ["top_p", "top_k", "stop_sequences", "metadata"]:
            if param in create_args:
                request_args[param] = create_args[param]

        # Stream the response
        stream_future: asyncio.Task[AsyncStream[RawMessageStreamEvent]] = asyncio.ensure_future(
            cast(Coroutine[Any, Any, AsyncStream[RawMessageStreamEvent]], self._client.messages.create(**request_args))
        )

        if cancellation_token is not None:
            cancellation_token.link_future(stream_future)  # type: ignore

        stream: AsyncStream[RawMessageStreamEvent] = cast(AsyncStream[RawMessageStreamEvent], await stream_future)  # type: ignore

        text_content: List[str] = []
        tool_calls: Dict[str, Dict[str, Any]] = {}  # Track tool calls by ID
        current_tool_id: Optional[str] = None
        input_tokens: int = 0
        output_tokens: int = 0
        stop_reason: Optional[str] = None

        first_chunk = True
        serialized_messages: List[Dict[str, Any]] = [self._serialize_message(msg) for msg in anthropic_messages]

        # Process the stream
        async for chunk in stream:
            if first_chunk:
                first_chunk = False
                # Emit the start event.
                logger.info(
                    LLMStreamStartEvent(
                        messages=serialized_messages,
                    )
                )
            # Handle different event types
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    # Start of a tool use block
                    current_tool_id = chunk.content_block.id
                    tool_calls[current_tool_id] = {
                        "id": chunk.content_block.id,
                        "name": chunk.content_block.name,
                        "input": "",  # Will be populated from deltas
                    }

            elif chunk.type == "content_block_delta":
                if hasattr(chunk.delta, "type") and chunk.delta.type == "text_delta":
                    # Handle text content
                    delta_text = chunk.delta.text
                    text_content.append(delta_text)
                    if delta_text:
                        yield delta_text

                # Handle tool input deltas - they come as InputJSONDelta
                elif hasattr(chunk.delta, "type") and chunk.delta.type == "input_json_delta":
                    if current_tool_id is not None and hasattr(chunk.delta, "partial_json"):
                        # Accumulate partial JSON for the current tool
                        tool_calls[current_tool_id]["input"] += chunk.delta.partial_json

            elif chunk.type == "content_block_stop":
                # End of a content block (could be text or tool)
                current_tool_id = None

            elif chunk.type == "message_delta":
                if hasattr(chunk.delta, "stop_reason") and chunk.delta.stop_reason:
                    stop_reason = chunk.delta.stop_reason

                # Get usage info if available
                if hasattr(chunk, "usage") and hasattr(chunk.usage, "output_tokens"):
                    output_tokens = chunk.usage.output_tokens

            elif chunk.type == "message_start":
                if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                    if hasattr(chunk.message.usage, "input_tokens"):
                        input_tokens = chunk.message.usage.input_tokens
                    if hasattr(chunk.message.usage, "output_tokens"):
                        output_tokens = chunk.message.usage.output_tokens

        # Prepare the final response
        usage = RequestUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )

        # Determine content based on what was received
        content: Union[str, List[FunctionCall]]
        thought = None

        if tool_calls:
            # We received tool calls
            if text_content:
                # Text before tool calls is treated as thought
                thought = "".join(text_content)

            # Convert tool calls to FunctionCall objects
            content = []
            for _, tool_data in tool_calls.items():
                # Parse the JSON input if needed
                input_str = tool_data["input"]
                try:
                    # If it's valid JSON, parse it; otherwise use as-is
                    if input_str.strip().startswith("{") and input_str.strip().endswith("}"):
                        parsed_input = json.loads(input_str)
                        input_str = json.dumps(parsed_input)  # Re-serialize to ensure valid JSON
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass

                content.append(
                    FunctionCall(
                        id=tool_data["id"],
                        name=normalize_name(tool_data["name"]),
                        arguments=input_str,
                    )
                )
        else:
            # Just text content
            content = "".join(text_content)

        # Create the final result
        result = CreateResult(
            finish_reason=normalize_stop_reason(stop_reason),
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )

        # Emit the end event.
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        # Update usage statistics
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        yield result

    async def close(self) -> None:
        await self._client.close()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """
        Estimate the number of tokens used by messages and tools.

        Note: This is an estimation based on common tokenization patterns and may not perfectly
        match Anthropic's exact token counting for Claude models.
        """
        # Use cl100k_base encoding as an approximation for Claude's tokenizer
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("gpt2")  # Fallback

        num_tokens = 0

        # System message tokens (if any)
        system_content = None
        for message in messages:
            if isinstance(message, SystemMessage):
                system_content = message.content
                break

        if system_content:
            num_tokens += len(encoding.encode(system_content)) + 15  # Approximate system message overhead

        # Message tokens
        for message in messages:
            if isinstance(message, SystemMessage):
                continue  # Already counted

            # Base token cost per message
            num_tokens += 10  # Approximate message role & formatting overhead

            # Content tokens
            if isinstance(message, UserMessage) or isinstance(message, AssistantMessage):
                if isinstance(message.content, str):
                    num_tokens += len(encoding.encode(message.content))
                elif isinstance(message.content, list):
                    # Handle different content types
                    for part in message.content:
                        if isinstance(part, str):
                            num_tokens += len(encoding.encode(part))
                        elif isinstance(part, Image):
                            # Estimate vision tokens (simplified)
                            num_tokens += 512  # Rough estimation for image tokens
                        elif isinstance(part, FunctionCall):
                            num_tokens += len(encoding.encode(part.name))
                            num_tokens += len(encoding.encode(part.arguments))
                            num_tokens += 10  # Function call overhead
            elif isinstance(message, FunctionExecutionResultMessage):
                for result in message.content:
                    num_tokens += len(encoding.encode(result.content))
                    num_tokens += 10  # Function result overhead

        # Tool tokens
        for tool in tools:
            if isinstance(tool, Tool):
                tool_schema = tool.schema
            else:
                tool_schema = tool

            # Name and description
            num_tokens += len(encoding.encode(tool_schema["name"]))
            if "description" in tool_schema:
                num_tokens += len(encoding.encode(tool_schema["description"]))

            # Parameters
            if "parameters" in tool_schema:
                params = tool_schema["parameters"]

                if "properties" in params:
                    for prop_name, prop_schema in params["properties"].items():
                        num_tokens += len(encoding.encode(prop_name))

                        if "type" in prop_schema:
                            num_tokens += len(encoding.encode(prop_schema["type"]))

                        if "description" in prop_schema:
                            num_tokens += len(encoding.encode(prop_schema["description"]))

                        # Special handling for enums
                        if "enum" in prop_schema:
                            for value in prop_schema["enum"]:
                                if isinstance(value, str):
                                    num_tokens += len(encoding.encode(value))
                                else:
                                    num_tokens += 2  # Non-string enum values

            # Tool overhead
            num_tokens += 20

        return num_tokens

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Calculate the remaining tokens based on the model's token limit."""
        token_limit = _model_info.get_token_limit(self._create_args["model"])
        return token_limit - self.count_tokens(messages, tools=tools)

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn("capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

# From anthropic/_anthropic_client.py
class AnthropicChatCompletionClient(
    BaseAnthropicChatCompletionClient, Component[AnthropicClientConfigurationConfigModel]
):
    """
    Chat completion client for Anthropic's Claude models.

    Args:
        model (str): The Claude model to use (e.g., "claude-3-sonnet-20240229", "claude-3-opus-20240229")
        api_key (str, optional): Anthropic API key. Required if not in environment variables.
        base_url (str, optional): Override the default API endpoint.
        max_tokens (int, optional): Maximum tokens in the response. Default is 4096.
        temperature (float, optional): Controls randomness. Lower is more deterministic. Default is 1.0.
        top_p (float, optional): Controls diversity via nucleus sampling. Default is 1.0.
        top_k (int, optional): Controls diversity via top-k sampling. Default is -1 (disabled).
        model_info (ModelInfo, optional): The capabilities of the model. Required if using a custom model.

    To use this client, you must install the Anthropic extension:

    .. code-block:: bash

        pip install "autogen-ext[anthropic]"

    Example:

    .. code-block:: python

        import asyncio
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        from autogen_core.models import UserMessage


        async def main():
            anthropic_client = AnthropicChatCompletionClient(
                model="claude-3-sonnet-20240229",
                api_key="your-api-key",  # Optional if ANTHROPIC_API_KEY is set in environment
            )

            result = await anthropic_client.create([UserMessage(content="What is the capital of France?", source="user")])  # type: ignore
            print(result)


        if __name__ == "__main__":
            asyncio.run(main())

    To load the client from a configuration:

    .. code-block:: python

        from autogen_core.models import ChatCompletionClient

        config = {
            "provider": "AnthropicChatCompletionClient",
            "config": {"model": "claude-3-sonnet-20240229"},
        }

        client = ChatCompletionClient.load_component(config)
    """

    component_type = "model"
    component_config_schema = AnthropicClientConfigurationConfigModel
    component_provider_override = "autogen_ext.models.anthropic.AnthropicChatCompletionClient"

    def __init__(self, **kwargs: Unpack[AnthropicClientConfiguration]):
        if "model" not in kwargs:
            raise ValueError("model is required for AnthropicChatCompletionClient")

        self._raw_config: Dict[str, Any] = dict(kwargs).copy()
        copied_args = dict(kwargs).copy()

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
            del copied_args["model_info"]

        client = _anthropic_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)

        super().__init__(
            client=client,
            create_args=create_args,
            model_info=model_info,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _anthropic_client_from_config(state["_raw_config"])

    def _to_config(self) -> AnthropicClientConfigurationConfigModel:
        copied_config = self._raw_config.copy()
        return AnthropicClientConfigurationConfigModel(**copied_config)

    @classmethod
    def _from_config(cls, config: AnthropicClientConfigurationConfigModel) -> Self:
        copied_config = config.model_copy().model_dump(exclude_none=True)

        # Handle api_key as SecretStr
        if "api_key" in copied_config and isinstance(config.api_key, SecretStr):
            copied_config["api_key"] = config.api_key.get_secret_value()

        return cls(**copied_config)

# From anthropic/_anthropic_client.py
class AnthropicBedrockChatCompletionClient(
    BaseAnthropicChatCompletionClient, Component[AnthropicBedrockClientConfigurationConfigModel]
):
    """
    Chat completion client for Anthropic's Claude models on AWS Bedrock.

    Args:
        model (str): The Claude model to use (e.g., "claude-3-sonnet-20240229", "claude-3-opus-20240229")
        api_key (str, optional): Anthropic API key. Required if not in environment variables.
        base_url (str, optional): Override the default API endpoint.
        max_tokens (int, optional): Maximum tokens in the response. Default is 4096.
        temperature (float, optional): Controls randomness. Lower is more deterministic. Default is 1.0.
        top_p (float, optional): Controls diversity via nucleus sampling. Default is 1.0.
        top_k (int, optional): Controls diversity via top-k sampling. Default is -1 (disabled).
        model_info (ModelInfo, optional): The capabilities of the model. Required if using a custom model.
        bedrock_info (BedrockInfo, optional): The capabilities of the model in bedrock. Required if using a model from AWS bedrock.

    To use this client, you must install the Anthropic extension:

    .. code-block:: bash

        pip install "autogen-ext[anthropic]"

    Example:

    .. code-block:: python

        import asyncio
        from autogen_ext.models.anthropic import AnthropicBedrockChatCompletionClient, BedrockInfo
        from autogen_core.models import UserMessage, ModelInfo


        async def main():
            anthropic_client = AnthropicBedrockChatCompletionClient(
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                temperature=0.1,
                model_info=ModelInfo(
                    vision=False, function_calling=True, json_output=False, family="unknown", structured_output=True
                ),
                bedrock_info=BedrockInfo(
                    aws_access_key="<aws_access_key>",
                    aws_secret_key="<aws_secret_key>",
                    aws_session_token="<aws_session_token>",
                    aws_region="<aws_region>",
                ),
            )

            result = await anthropic_client.create([UserMessage(content="What is the capital of France?", source="user")])  # type: ignore
            print(result)


        if __name__ == "__main__":
            asyncio.run(main())
    """

    component_type = "model"
    component_config_schema = AnthropicBedrockClientConfigurationConfigModel
    component_provider_override = "autogen_ext.models.anthropic.AnthropicBedrockChatCompletionClient"

    def __init__(self, **kwargs: Unpack[AnthropicBedrockClientConfiguration]):
        if "model" not in kwargs:
            raise ValueError("model is required for  AnthropicBedrockChatCompletionClient")

        self._raw_config: Dict[str, Any] = dict(kwargs).copy()
        copied_args = dict(kwargs).copy()

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
            del copied_args["model_info"]

        bedrock_info: Optional[BedrockInfo] = None
        if "bedrock_info" in kwargs:
            bedrock_info = kwargs["bedrock_info"]

        if bedrock_info is None:
            raise ValueError("bedrock_info is required for AnthropicBedrockChatCompletionClient")

        # Handle bedrock_info
        aws_region = bedrock_info["aws_region"]
        aws_access_key: Optional[str] = None
        aws_secret_key: Optional[str] = None
        aws_session_token: Optional[str] = None
        if all(key in bedrock_info for key in ("aws_access_key", "aws_secret_key", "aws_session_token")):
            aws_access_key = bedrock_info["aws_access_key"]
            aws_secret_key = bedrock_info["aws_secret_key"]
            aws_session_token = bedrock_info["aws_session_token"]

        client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
        )
        create_args = _create_args_from_config(copied_args)

        super().__init__(
            client=client,
            create_args=create_args,
            model_info=model_info,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _anthropic_client_from_config(state["_raw_config"])

    def _to_config(self) -> AnthropicBedrockClientConfigurationConfigModel:
        copied_config = self._raw_config.copy()
        return AnthropicBedrockClientConfigurationConfigModel(**copied_config)

    @classmethod
    def _from_config(cls, config: AnthropicBedrockClientConfigurationConfigModel) -> Self:
        copied_config = config.model_copy().model_dump(exclude_none=True)

        # Handle api_key as SecretStr
        if "api_key" in copied_config and isinstance(config.api_key, SecretStr):
            copied_config["api_key"] = config.api_key.get_secret_value()

        # Handle bedrock_info as SecretStr
        if "bedrock_info" in copied_config and isinstance(config.bedrock_info, dict):
            copied_config["bedrock_info"] = {
                "aws_access_key": config.bedrock_info["aws_access_key"].get_secret_value(),
                "aws_secret_key": config.bedrock_info["aws_secret_key"].get_secret_value(),
                "aws_session_token": config.bedrock_info["aws_session_token"].get_secret_value(),
                "aws_region": config.bedrock_info["aws_region"],
            }

        return cls(**copied_config)

# From anthropic/_anthropic_client.py
def get_mime_type_from_image(image: Image) -> Literal["image/jpeg", "image/png", "image/gif", "image/webp"]:
    """Get a valid Anthropic media type from an Image object."""
    # Get base64 data first
    base64_data = image.to_base64()

    # Decode the base64 string
    image_data = base64.b64decode(base64_data)

    # Check the first few bytes for known signatures
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
        return "image/gif"
    elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    else:
        # Default to JPEG as a fallback
        return "image/jpeg"

# From anthropic/_anthropic_client.py
def convert_tool_choice_anthropic(tool_choice: Tool | Literal["auto", "required", "none"]) -> Any:
    """Convert tool_choice parameter to Anthropic API format.

    Args:
        tool_choice: A single Tool object to force the model to use, "auto" to let the model choose any available tool, "required" to force tool usage, or "none" to disable tool usage.

    Returns:
        Anthropic API compatible tool_choice value.
    """
    if tool_choice == "none":
        return {"type": "none"}

    if tool_choice == "auto":
        return {"type": "auto"}

    if tool_choice == "required":
        return {"type": "any"}  # Anthropic uses "any" for required

    # Must be a Tool object
    if isinstance(tool_choice, Tool):
        return {"type": "tool", "name": tool_choice.schema["name"]}
    else:
        raise ValueError(f"tool_choice must be a Tool object, 'auto', 'required', or 'none', got {type(tool_choice)}")

# From anthropic/_anthropic_client.py
def user_message_to_anthropic(message: UserMessage) -> MessageParam:
    assert_valid_name(message.source)

    if isinstance(message.content, str):
        return {
            "role": "user",
            "content": __empty_content_to_whitespace(message.content),
        }
    else:
        blocks: List[Union[TextBlockParam, ImageBlockParam]] = []

        for part in message.content:
            if isinstance(part, str):
                blocks.append(TextBlockParam(type="text", text=__empty_content_to_whitespace(part)))
            elif isinstance(part, Image):
                blocks.append(
                    ImageBlockParam(
                        type="image",
                        source=Base64ImageSourceParam(
                            type="base64",
                            media_type=get_mime_type_from_image(part),
                            data=part.to_base64(),
                        ),
                    )
                )
            else:
                raise ValueError(f"Unknown content type: {part}")

        return {
            "role": "user",
            "content": blocks,
        }

# From anthropic/_anthropic_client.py
def system_message_to_anthropic(message: SystemMessage) -> str:
    return __empty_content_to_whitespace(message.content)

# From anthropic/_anthropic_client.py
def assistant_message_to_anthropic(message: AssistantMessage) -> MessageParam:
    assert_valid_name(message.source)

    if isinstance(message.content, list):
        # Tool calls
        tool_use_blocks: List[ToolUseBlock] = []

        for func_call in message.content:
            # Parse the arguments and convert to dict if it's a JSON string
            args = func_call.arguments
            args = __empty_content_to_whitespace(args)
            if isinstance(args, str):
                try:
                    json_objs = extract_json_from_str(args)
                    if len(json_objs) != 1:
                        raise ValueError(f"Expected a single JSON object, but found {len(json_objs)}")
                    args_dict = json_objs[0]
                except json.JSONDecodeError:
                    args_dict = {"text": args}
            else:
                args_dict = args

            tool_use_blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=func_call.id,
                    name=func_call.name,
                    input=args_dict,
                )
            )

        # Include thought if available
        content_blocks: List[ContentBlock] = []
        if hasattr(message, "thought") and message.thought is not None:
            content_blocks.append(TextBlock(type="text", text=message.thought))

        content_blocks.extend(tool_use_blocks)

        return {
            "role": "assistant",
            "content": content_blocks,
        }
    else:
        # Simple text content
        return {
            "role": "assistant",
            "content": message.content,
        }

# From anthropic/_anthropic_client.py
def tool_message_to_anthropic(message: FunctionExecutionResultMessage) -> List[MessageParam]:
    # Create a single user message containing all tool results
    content_blocks: List[ToolResultBlockParam] = []

    for result in message.content:
        content_blocks.append(
            ToolResultBlockParam(
                type="tool_result",
                tool_use_id=result.call_id,
                content=result.content,
            )
        )

    return [
        {
            "role": "user",  # Changed from "tool" to "user"
            "content": content_blocks,
        }
    ]

# From anthropic/_anthropic_client.py
def to_anthropic_type(message: LLMMessage) -> Union[str, List[MessageParam], MessageParam]:
    if isinstance(message, SystemMessage):
        return system_message_to_anthropic(message)
    elif isinstance(message, UserMessage):
        return user_message_to_anthropic(message)
    elif isinstance(message, AssistantMessage):
        return assistant_message_to_anthropic(message)
    else:
        return tool_message_to_anthropic(message)

from autogen_core.tools import FunctionTool
from topics import sales_agent_topic_type
from topics import issues_and_repairs_agent_topic_type
from topics import triage_agent_topic_type

# From core_streaming_handoffs_fastapi/tools_delegate.py
def transfer_to_sales_agent() -> str:
    return sales_agent_topic_type

# From core_streaming_handoffs_fastapi/tools_delegate.py
def transfer_to_issues_and_repairs() -> str:
    return issues_and_repairs_agent_topic_type

# From core_streaming_handoffs_fastapi/tools_delegate.py
def transfer_back_to_triage() -> str:
    return triage_agent_topic_type


# From core_streaming_handoffs_fastapi/tools.py
def execute_order(product: str, price: int) -> Dict[str, Union[str, int]]:
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    return {"product":product,"price":price}

# From core_streaming_handoffs_fastapi/tools.py
def look_up_item(search_query: str) -> Dict[str, str]:
    item_id = "item_132612938"
    return {"item_id":item_id,"status":"found"}

# From core_streaming_handoffs_fastapi/tools.py
def execute_refund(item_id: str, reason: str = "not provided") -> Dict[str, str]:
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return {"item_id":item_id, "reason":reason, "refund_status":"Successful"}

import traceback
from agentforge.agent import Agent
from agentforge.utils.logger import Logger
from agentforge.storage.chroma_storage import ChromaStorage
from agentforge.utils.parsing_processor import ParsingProcessor
from config import Config
from utils.tool_utils import ToolUtils

# From modules/actions.py
class Actions:
    """
    Provides a series of methods for developers to create custom solutions for managing and executing actions and tools
    within the framework. This class offers the necessary flexibility and modularity to support both in-depth custom
    implementations and generic examples.

    The `auto_execute` method serves as a comprehensive example of how to use the provided methods to orchestrate the
    flow from loading action-specific tools, executing these tools, to injecting the processed data into the knowledge
    graph. Developers can use this method directly or reference it to build their own tailored workflows.
    """

    # --------------------------------------------------------------------------------------------------------
    # -------------------------------- Constructor and Initialization Methods --------------------------------
    # --------------------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Initializes the Actions class, setting up logger, storage utilities, and loading necessary components for
        action processing.
        """
        # Initialize the logger, storage, and functions
        self.logger = Logger(name=self.__class__.__name__)
        self.config = Config()
        self.storage = ChromaStorage.get_or_create(storage_id="actions_module")
        self.tool_utils = ToolUtils()
        self.parsing_utils = ParsingProcessor()

        # Initialize the agents
        self.action_creation = Agent("ActionCreationAgent")
        self.action_selection = Agent("ActionSelectionAgent")
        self.priming_agent = Agent("ToolPrimingAgent")

        # Load the actions and tools from the config
        self.actions = self.initialize_collection('Actions')
        self.tools = self.initialize_collection('Tools')

    # --------------------------------------------------------------------------------------------------------
    # ------------------------------------------- Helper Methods ---------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    def initialize_collection(self, collection_name: str) -> Dict[str, Dict]:
        """
        Initializes a specified collection in the vector database with preloaded data. Mainly used to load the
        actions and tools data into the database, allowing for semantic search.

        Parameters:
            collection_name (str): The name of the collection to initialize.

        Returns:
            Dict[str, Dict]: A dictionary where keys are item names and values are item details.
        """
        item_list = {}
        data = self.config.data[collection_name.lower()]
        ids = id_generator(data)

        for (key, value), act_id in zip(data.items(), ids):
            value['ID'] = act_id
            item_list[value['Name']] = value

        description = [value['Description'] for value in item_list.values()]
        metadata = [{'Name': key} for key, value in item_list.items()]

        # Save the item into the selected collection
        self.storage.save_memory(collection_name=collection_name, data=description, ids=ids, metadata=metadata)
        self.logger.log(f"\n{collection_name} collection initialized", 'info', 'Actions')

        return item_list

    def get_relevant_items_for_objective(self, collection_name: str, objective: str,
                                         threshold: Optional[float] = None,
                                         num_results: int = 1, parse_result: bool = True) -> Dict[str, Dict]:
        """
        Loads items (actions or tools) based on the current objective and specified criteria.

        Parameters:
            collection_name (str): The name of the collection to search in ('Actions' or 'Tools').
            objective (str): The objective to find relevant items for.
            threshold (Optional[float]): The threshold for item relevance (Lower is stricter).
            num_results (int): The number of results to return. Default is 1.
            parse_result (bool): Whether to parse the result. Default is True.
                If False, returns the results as they come from the database.
                If True, parses the results to include only items that are loaded in the system.

        Returns:
            Dict[str, Dict]: The item list or an empty dictionary if no items are found.
        """
        item_list = {}
        try:
            item_list = self.storage.search_storage_by_threshold(collection_name=collection_name,
                                                                 query=objective,
                                                                 threshold=threshold,
                                                                 num_results=num_results)
        except Exception as e:
            self.logger.log(f"Error loading {collection_name.lower()}: {e}", 'error', 'Actions')

        if not item_list:
            self.logger.log(f"No {collection_name} Found", 'info', 'Actions')
            return {}

        if parse_result:
            parsed_item_list = {}
            for metadata in item_list.get('metadatas', []):
                item_name = metadata.get('Name')
                if item_name in getattr(self, collection_name.lower()):
                    parsed_item_list[item_name] = getattr(self, collection_name.lower())[item_name]
            item_list = parsed_item_list

        return item_list

    def get_tools_in_action(self, action: Dict) -> Optional[List[Dict]]:
        """
        Loads the tools specified in the action's configuration.

        Parameters:
            action (Dict): The action containing the tools to load.

        Returns:
            Optional[List[Dict]]: A list with the loaded tools or None.

        Raises:
            Exception: If an error occurs while loading action tools.
        """
        try:
            tools = [self.tools[tool] for tool in action['Tools']]
        except Exception as e:
            error_message = f"Error in loading tools from action '{action['Name']}': {e}"
            self.logger.log(error_message, 'error', 'Actions')
            tools = {'error': error_message, 'traceback': traceback.format_exc()}

        return tools

    # --------------------------------------------------------------------------------------------------------
    # ----------------------------------- Primary Module (Agents) Methods ------------------------------------
    # --------------------------------------------------------------------------------------------------------

    def select_action_for_objective(self, objective: str, action_list: Union[str, Dict], context: Optional[str] = None,
                                    parse_result: bool = True) -> Union[str, Dict]:
        """
        Selects an action for the given objective from the provided action list.

        Parameters:
            objective (str): The objective to select an action for.
            action_list (Union[str, Dict]): The list of actions to select from.
                If given a Dict, the method will attempt to convert to a string.
            context (Optional[str]): The context for action selection.
            parse_result (bool): Whether to parse the result. Default is True.

        Returns:
            Union[str, Dict]: The selected action or formatted result.
        """
        if isinstance(action_list, Dict):
            action_list = self.tool_utils.format_item_list(action_list)

        selected_action = self.action_selection.run(objective=objective, action_list=action_list, context=context)

        if parse_result:
            selected_action = self.parsing_utils.parse_yaml_content(selected_action)

        return selected_action

    def craft_action_for_objective(self, objective: str, tool_list: Union[Dict, str], context: Optional[str] = None,
                                   parse_result: bool = True) -> Union[str, Dict]:
        """
        Crafts a new action for the given objective.

        Parameters:
            objective (str): The objective to craft an action for.
            tool_list (Union[Dict, str]): The list of tools to be used.
                Will attempt to convert to a string if given a Dict.
            context (Optional[str]): The context for action crafting.
            parse_result (bool): Whether to parse the result. Default is True.

        Returns:
            Union[str, Dict]: The crafted action or formatted result.
        """
        if isinstance(tool_list, Dict):
            tool_list = self.tool_utils.format_item_list(tool_list)

        new_action = self.action_creation.run(objective=objective,
                                              context=context,
                                              tool_list=tool_list)

        if parse_result:
            new_action = self.parsing_utils.parse_yaml_content(new_action)

            if new_action is None:
                msg = {'error': "Error Creating Action"}
                self.logger.log(msg['error'], 'error', 'Actions')
                return msg
            # else:
            #     path = f".agentforge/actions/unverified/{new_action['Name'].replace(' ', '_')}.yaml"
            #     with open(path, "w") as file:
            #         yaml.dump(new_action, file)
            #     # self.functions.agent_utils.config.add_item(new_action, 'Actions')
            #     count = self.storage.count_documents(collection_name='actions') + 1
            #     metadata = [{'Name': new_action['Name'], 'Description': new_action['Description'], 'Path': path}]
            #     self.storage.save_memory(collection_name='actions', data=new_action['Description'], ids=count,
            #                              metadata=metadata)

        return new_action

    def prime_tool_for_action(self, objective: str, action: Union[str, Dict], tool: Dict,
                              previous_results: Optional[str] = None,
                              tool_context: Optional[str] = None, action_info_order: Optional[List[str]] = None,
                              tool_info_order: Optional[List[str]] = None) -> Dict:
        """
        Prepares the tool for execution by running the ToolPrimingAgent.

        Parameters:
            objective (str): The objective for tool priming.
            action (Union[str, Dict]): The action to prime the tool for.
                If a dictionary, it will be formatted using the tool_info_order methods.
            tool (Dict): The tool to be primed.
            previous_results (Optional[str]): The results from previous tool executions.
            tool_context (Optional[str]): The context for the tool.
            action_info_order (Optional[List[str]]): The order of action information to include in the Agent prompt.
            tool_info_order (Optional[List[str]]): The order of tool information to include in the Agent prompt.

        Returns:
            Dict: The formatted payload for the tool.

        Raises:
            Exception: If an error occurs during tool priming.
        """
        formatted_tool = self.tool_utils.format_item(tool, tool_info_order)

        if isinstance(action, Dict):
            action = self.tool_utils.format_item(action, action_info_order)

        try:
            # Load the paths into a dictionary
            paths_dict = self.storage.config.settings.system.paths

            # Construct the work_paths string by iterating over the dictionary
            work_paths = None
            if paths_dict:
                work_paths = "\n".join(f"{key}: {value}" for key, value in paths_dict.items())

            payload = self.priming_agent.run(objective=objective,
                                             action=action,
                                             tool_name=tool.get('Name'),
                                             tool_info=formatted_tool,
                                             path=work_paths,
                                             previous_results=previous_results,
                                             tool_context=tool_context)

            formatted_payload = self.parsing_utils.parse_yaml_content(payload)

            if formatted_payload is None:
                return {'error': 'Parsing Error - Model did not respond in specified format'}

            self.logger.log(f"Tool Payload: {formatted_payload}", 'info', 'Actions')
            return formatted_payload
        except Exception as e:
            message = f"Error in priming tool '{tool['Name']}': {e}"
            self.logger.log(message, 'error', "Actions")
            return {'error': message, 'traceback': traceback.format_exc()}

    def run_tools_in_sequence(self, objective: str, action: Dict,
                              action_info_order: Optional[List[str]] = None,
                              tool_info_order: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Runs the specified tools in sequence for the given objective and action.

        Parameters:
            objective (str): The objective for running the tools.
            action (Dict): The action containing the tools to run.
            action_info_order (Optional[List[str]]): The order of action information to include in the Agent prompt.
            tool_info_order (Optional[List[str]]): The order of tool information to include in the Agent prompt.

        Returns:
            Optional[Dict]: The final result of the tool execution or an error dictionary.

        Raises:
            Exception: If an error occurs while running the tools in sequence.
        """
        results: Dict = {}
        tool_context: str = ''

        try:
            tools = self.get_tools_in_action(action=action)

            # Check if an error occurred
            if 'error' in tools:
                return tools  # Stop execution and return the error message

            for tool in tools:
                payload = self.prime_tool_for_action(objective=objective,
                                                     action=action,
                                                     tool=tool,
                                                     previous_results=results.get('data', None),
                                                     tool_context=tool_context,
                                                     action_info_order=action_info_order,
                                                     tool_info_order=tool_info_order)

                if isinstance(payload, Dict) and 'error' in payload:
                    return payload  # Stop execution and return the error message

                tool_context = payload['thoughts'].get('next_tool_context')
                results = self.tool_utils.dynamic_tool(tool, payload)

                # Check if an error occurred
                if isinstance(results, Dict) and results['status'] != 'success':
                    return results  # Stop loop and return the error message

            return results

        except Exception as e:
            error_message = f"Error running tools in sequence: {e}"
            self.logger.log(error_message, 'error')
            return {'error': error_message, 'traceback': traceback.format_exc()}

    # --------------------------------------------------------------------------------------------------------
    # ------------------------------------------ Solution Example --------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    def auto_execute(self, objective: str, context: Optional[str] = None,
                     threshold: Optional[float] = 0.8) -> Union[Dict, str, None]:
        """
        Automatically executes the actions for the given objective and context.

        Parameters:
            objective (str): The objective for the execution.
            context (Optional[str]): The context for the execution.
            threshold (Optional[float]): The threshold for action relevance (Lower is stricter). Default is 0.8.

        Returns:
            Union[Dict, str, None]: The result of the execution or an error dictionary.

        Raises:
            Exception: If an error occurs during execution.
        """
        try:
            action_list = self.get_relevant_items_for_objective(collection_name='Actions', objective=objective,
                                                                threshold=threshold, num_results=10)
            if action_list:
                self.logger.log(f"\nSelecting Action for Objective:\n{objective}", 'info', 'Actions')
                order = ["Name", "Description"]
                available_actions = self.tool_utils.format_item_list(action_list, order)
                selected_action = self.select_action_for_objective(objective=objective,
                                                                   action_list=available_actions,
                                                                   context=context)
                selected_action = self.actions[selected_action['action']]
                self.logger.log(f"\nSelected Action:\n{selected_action}", 'info', 'Actions')
            else:
                self.logger.log(f"\nCrafting Action for Objective:\n{objective}", 'info', 'Actions')
                order = ["Name", "Description", "Args"]
                threshold = 1
                tool_list = self.get_relevant_items_for_objective(collection_name='Tools', objective=objective,
                                                                  threshold=threshold, num_results=10)
                available_tools = self.tool_utils.format_item_list(tool_list, order)
                selected_action = self.craft_action_for_objective(objective=objective,
                                                                  tool_list=available_tools,
                                                                  context=context)
                self.logger.log(f"\nCrafted Action:\n{selected_action}", 'info', 'Actions')

                if 'error' in selected_action:
                    return selected_action

            action_info_order = ["Name", "Description"]
            tool_info_order = ["Name", "Description", "Args", "Instruction", "Example"]
            result: Dict = self.run_tools_in_sequence(objective=objective,
                                                      action=selected_action,
                                                      action_info_order=action_info_order,
                                                      tool_info_order=tool_info_order)
            # Check if an error occurred
            if isinstance(result, Dict) and result['status'] != 'success':
                self.logger.log(f"\nAction Failed:\n{result['message']}", 'error', 'Actions')
                return result  # Stop execution and return the error message

            self.logger.log(f"\nAction Result:\n{result['data']}", 'info', 'Actions')
            return result
        except Exception as e:
            error_message = f"Error in running action: {e}"
            self.logger.log(error_message, 'error', 'Actions')
            return {'error': error_message, 'traceback': traceback.format_exc()}

# From modules/actions.py
def id_generator(data: List[Dict]) -> List[str]:
    """
    Generates a list of string IDs for the given data.

    Parameters:
        data (List[Dict]): The data for which to generate IDs.

    Returns:
        List[str]: A list of generated string IDs.
    """
    return [str(i + 1) for i in range(len(data))]

# From modules/actions.py
def initialize_collection(self, collection_name: str) -> Dict[str, Dict]:
        """
        Initializes a specified collection in the vector database with preloaded data. Mainly used to load the
        actions and tools data into the database, allowing for semantic search.

        Parameters:
            collection_name (str): The name of the collection to initialize.

        Returns:
            Dict[str, Dict]: A dictionary where keys are item names and values are item details.
        """
        item_list = {}
        data = self.config.data[collection_name.lower()]
        ids = id_generator(data)

        for (key, value), act_id in zip(data.items(), ids):
            value['ID'] = act_id
            item_list[value['Name']] = value

        description = [value['Description'] for value in item_list.values()]
        metadata = [{'Name': key} for key, value in item_list.items()]

        # Save the item into the selected collection
        self.storage.save_memory(collection_name=collection_name, data=description, ids=ids, metadata=metadata)
        self.logger.log(f"\n{collection_name} collection initialized", 'info', 'Actions')

        return item_list

# From modules/actions.py
def get_relevant_items_for_objective(self, collection_name: str, objective: str,
                                         threshold: Optional[float] = None,
                                         num_results: int = 1, parse_result: bool = True) -> Dict[str, Dict]:
        """
        Loads items (actions or tools) based on the current objective and specified criteria.

        Parameters:
            collection_name (str): The name of the collection to search in ('Actions' or 'Tools').
            objective (str): The objective to find relevant items for.
            threshold (Optional[float]): The threshold for item relevance (Lower is stricter).
            num_results (int): The number of results to return. Default is 1.
            parse_result (bool): Whether to parse the result. Default is True.
                If False, returns the results as they come from the database.
                If True, parses the results to include only items that are loaded in the system.

        Returns:
            Dict[str, Dict]: The item list or an empty dictionary if no items are found.
        """
        item_list = {}
        try:
            item_list = self.storage.search_storage_by_threshold(collection_name=collection_name,
                                                                 query=objective,
                                                                 threshold=threshold,
                                                                 num_results=num_results)
        except Exception as e:
            self.logger.log(f"Error loading {collection_name.lower()}: {e}", 'error', 'Actions')

        if not item_list:
            self.logger.log(f"No {collection_name} Found", 'info', 'Actions')
            return {}

        if parse_result:
            parsed_item_list = {}
            for metadata in item_list.get('metadatas', []):
                item_name = metadata.get('Name')
                if item_name in getattr(self, collection_name.lower()):
                    parsed_item_list[item_name] = getattr(self, collection_name.lower())[item_name]
            item_list = parsed_item_list

        return item_list

# From modules/actions.py
def get_tools_in_action(self, action: Dict) -> Optional[List[Dict]]:
        """
        Loads the tools specified in the action's configuration.

        Parameters:
            action (Dict): The action containing the tools to load.

        Returns:
            Optional[List[Dict]]: A list with the loaded tools or None.

        Raises:
            Exception: If an error occurs while loading action tools.
        """
        try:
            tools = [self.tools[tool] for tool in action['Tools']]
        except Exception as e:
            error_message = f"Error in loading tools from action '{action['Name']}': {e}"
            self.logger.log(error_message, 'error', 'Actions')
            tools = {'error': error_message, 'traceback': traceback.format_exc()}

        return tools

# From modules/actions.py
def select_action_for_objective(self, objective: str, action_list: Union[str, Dict], context: Optional[str] = None,
                                    parse_result: bool = True) -> Union[str, Dict]:
        """
        Selects an action for the given objective from the provided action list.

        Parameters:
            objective (str): The objective to select an action for.
            action_list (Union[str, Dict]): The list of actions to select from.
                If given a Dict, the method will attempt to convert to a string.
            context (Optional[str]): The context for action selection.
            parse_result (bool): Whether to parse the result. Default is True.

        Returns:
            Union[str, Dict]: The selected action or formatted result.
        """
        if isinstance(action_list, Dict):
            action_list = self.tool_utils.format_item_list(action_list)

        selected_action = self.action_selection.run(objective=objective, action_list=action_list, context=context)

        if parse_result:
            selected_action = self.parsing_utils.parse_yaml_content(selected_action)

        return selected_action

# From modules/actions.py
def craft_action_for_objective(self, objective: str, tool_list: Union[Dict, str], context: Optional[str] = None,
                                   parse_result: bool = True) -> Union[str, Dict]:
        """
        Crafts a new action for the given objective.

        Parameters:
            objective (str): The objective to craft an action for.
            tool_list (Union[Dict, str]): The list of tools to be used.
                Will attempt to convert to a string if given a Dict.
            context (Optional[str]): The context for action crafting.
            parse_result (bool): Whether to parse the result. Default is True.

        Returns:
            Union[str, Dict]: The crafted action or formatted result.
        """
        if isinstance(tool_list, Dict):
            tool_list = self.tool_utils.format_item_list(tool_list)

        new_action = self.action_creation.run(objective=objective,
                                              context=context,
                                              tool_list=tool_list)

        if parse_result:
            new_action = self.parsing_utils.parse_yaml_content(new_action)

            if new_action is None:
                msg = {'error': "Error Creating Action"}
                self.logger.log(msg['error'], 'error', 'Actions')
                return msg
            # else:
            #     path = f".agentforge/actions/unverified/{new_action['Name'].replace(' ', '_')}.yaml"
            #     with open(path, "w") as file:
            #         yaml.dump(new_action, file)
            #     # self.functions.agent_utils.config.add_item(new_action, 'Actions')
            #     count = self.storage.count_documents(collection_name='actions') + 1
            #     metadata = [{'Name': new_action['Name'], 'Description': new_action['Description'], 'Path': path}]
            #     self.storage.save_memory(collection_name='actions', data=new_action['Description'], ids=count,
            #                              metadata=metadata)

        return new_action

# From modules/actions.py
def prime_tool_for_action(self, objective: str, action: Union[str, Dict], tool: Dict,
                              previous_results: Optional[str] = None,
                              tool_context: Optional[str] = None, action_info_order: Optional[List[str]] = None,
                              tool_info_order: Optional[List[str]] = None) -> Dict:
        """
        Prepares the tool for execution by running the ToolPrimingAgent.

        Parameters:
            objective (str): The objective for tool priming.
            action (Union[str, Dict]): The action to prime the tool for.
                If a dictionary, it will be formatted using the tool_info_order methods.
            tool (Dict): The tool to be primed.
            previous_results (Optional[str]): The results from previous tool executions.
            tool_context (Optional[str]): The context for the tool.
            action_info_order (Optional[List[str]]): The order of action information to include in the Agent prompt.
            tool_info_order (Optional[List[str]]): The order of tool information to include in the Agent prompt.

        Returns:
            Dict: The formatted payload for the tool.

        Raises:
            Exception: If an error occurs during tool priming.
        """
        formatted_tool = self.tool_utils.format_item(tool, tool_info_order)

        if isinstance(action, Dict):
            action = self.tool_utils.format_item(action, action_info_order)

        try:
            # Load the paths into a dictionary
            paths_dict = self.storage.config.settings.system.paths

            # Construct the work_paths string by iterating over the dictionary
            work_paths = None
            if paths_dict:
                work_paths = "\n".join(f"{key}: {value}" for key, value in paths_dict.items())

            payload = self.priming_agent.run(objective=objective,
                                             action=action,
                                             tool_name=tool.get('Name'),
                                             tool_info=formatted_tool,
                                             path=work_paths,
                                             previous_results=previous_results,
                                             tool_context=tool_context)

            formatted_payload = self.parsing_utils.parse_yaml_content(payload)

            if formatted_payload is None:
                return {'error': 'Parsing Error - Model did not respond in specified format'}

            self.logger.log(f"Tool Payload: {formatted_payload}", 'info', 'Actions')
            return formatted_payload
        except Exception as e:
            message = f"Error in priming tool '{tool['Name']}': {e}"
            self.logger.log(message, 'error', "Actions")
            return {'error': message, 'traceback': traceback.format_exc()}

# From modules/actions.py
def run_tools_in_sequence(self, objective: str, action: Dict,
                              action_info_order: Optional[List[str]] = None,
                              tool_info_order: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Runs the specified tools in sequence for the given objective and action.

        Parameters:
            objective (str): The objective for running the tools.
            action (Dict): The action containing the tools to run.
            action_info_order (Optional[List[str]]): The order of action information to include in the Agent prompt.
            tool_info_order (Optional[List[str]]): The order of tool information to include in the Agent prompt.

        Returns:
            Optional[Dict]: The final result of the tool execution or an error dictionary.

        Raises:
            Exception: If an error occurs while running the tools in sequence.
        """
        results: Dict = {}
        tool_context: str = ''

        try:
            tools = self.get_tools_in_action(action=action)

            # Check if an error occurred
            if 'error' in tools:
                return tools  # Stop execution and return the error message

            for tool in tools:
                payload = self.prime_tool_for_action(objective=objective,
                                                     action=action,
                                                     tool=tool,
                                                     previous_results=results.get('data', None),
                                                     tool_context=tool_context,
                                                     action_info_order=action_info_order,
                                                     tool_info_order=tool_info_order)

                if isinstance(payload, Dict) and 'error' in payload:
                    return payload  # Stop execution and return the error message

                tool_context = payload['thoughts'].get('next_tool_context')
                results = self.tool_utils.dynamic_tool(tool, payload)

                # Check if an error occurred
                if isinstance(results, Dict) and results['status'] != 'success':
                    return results  # Stop loop and return the error message

            return results

        except Exception as e:
            error_message = f"Error running tools in sequence: {e}"
            self.logger.log(error_message, 'error')
            return {'error': error_message, 'traceback': traceback.format_exc()}

# From modules/actions.py
def auto_execute(self, objective: str, context: Optional[str] = None,
                     threshold: Optional[float] = 0.8) -> Union[Dict, str, None]:
        """
        Automatically executes the actions for the given objective and context.

        Parameters:
            objective (str): The objective for the execution.
            context (Optional[str]): The context for the execution.
            threshold (Optional[float]): The threshold for action relevance (Lower is stricter). Default is 0.8.

        Returns:
            Union[Dict, str, None]: The result of the execution or an error dictionary.

        Raises:
            Exception: If an error occurs during execution.
        """
        try:
            action_list = self.get_relevant_items_for_objective(collection_name='Actions', objective=objective,
                                                                threshold=threshold, num_results=10)
            if action_list:
                self.logger.log(f"\nSelecting Action for Objective:\n{objective}", 'info', 'Actions')
                order = ["Name", "Description"]
                available_actions = self.tool_utils.format_item_list(action_list, order)
                selected_action = self.select_action_for_objective(objective=objective,
                                                                   action_list=available_actions,
                                                                   context=context)
                selected_action = self.actions[selected_action['action']]
                self.logger.log(f"\nSelected Action:\n{selected_action}", 'info', 'Actions')
            else:
                self.logger.log(f"\nCrafting Action for Objective:\n{objective}", 'info', 'Actions')
                order = ["Name", "Description", "Args"]
                threshold = 1
                tool_list = self.get_relevant_items_for_objective(collection_name='Tools', objective=objective,
                                                                  threshold=threshold, num_results=10)
                available_tools = self.tool_utils.format_item_list(tool_list, order)
                selected_action = self.craft_action_for_objective(objective=objective,
                                                                  tool_list=available_tools,
                                                                  context=context)
                self.logger.log(f"\nCrafted Action:\n{selected_action}", 'info', 'Actions')

                if 'error' in selected_action:
                    return selected_action

            action_info_order = ["Name", "Description"]
            tool_info_order = ["Name", "Description", "Args", "Instruction", "Example"]
            result: Dict = self.run_tools_in_sequence(objective=objective,
                                                      action=selected_action,
                                                      action_info_order=action_info_order,
                                                      tool_info_order=tool_info_order)
            # Check if an error occurred
            if isinstance(result, Dict) and result['status'] != 'success':
                self.logger.log(f"\nAction Failed:\n{result['message']}", 'error', 'Actions')
                return result  # Stop execution and return the error message

            self.logger.log(f"\nAction Result:\n{result['data']}", 'info', 'Actions')
            return result
        except Exception as e:
            error_message = f"Error in running action: {e}"
            self.logger.log(error_message, 'error', 'Actions')
            return {'error': error_message, 'traceback': traceback.format_exc()}

from fastapi import APIRouter
from fastapi import Depends
from fastapi.responses import StreamingResponse
from reworkd_platform.schemas.agent import AgentChat
from reworkd_platform.schemas.agent import AgentRun
from reworkd_platform.schemas.agent import AgentSummarize
from reworkd_platform.schemas.agent import AgentTaskAnalyze
from reworkd_platform.schemas.agent import AgentTaskCreate
from reworkd_platform.schemas.agent import AgentTaskExecute
from reworkd_platform.schemas.agent import NewTasksResponse
from reworkd_platform.web.api.agent.agent_service.agent_service import AgentService
from reworkd_platform.web.api.agent.agent_service.agent_service_provider import get_agent_service
from reworkd_platform.web.api.agent.analysis import Analysis
from reworkd_platform.web.api.agent.dependancies import agent_analyze_validator
from reworkd_platform.web.api.agent.dependancies import agent_chat_validator
from reworkd_platform.web.api.agent.dependancies import agent_create_validator
from reworkd_platform.web.api.agent.dependancies import agent_execute_validator
from reworkd_platform.web.api.agent.dependancies import agent_start_validator
from reworkd_platform.web.api.agent.dependancies import agent_summarize_validator
from reworkd_platform.web.api.agent.tools.tools import get_external_tools
from reworkd_platform.web.api.agent.tools.tools import get_tool_name

# From agent/views.py
class ToolModel(BaseModel):
    name: str
    description: str
    color: str
    image_url: Optional[str]

# From agent/views.py
class ToolsResponse(BaseModel):
    tools: List[ToolModel]

from reworkd_platform.db.crud.oauth import OAuthCrud
from reworkd_platform.schemas.user import UserBase
from reworkd_platform.web.api.agent.tools.code import Code
from reworkd_platform.web.api.agent.tools.image import Image
from reworkd_platform.web.api.agent.tools.search import Search
from reworkd_platform.web.api.agent.tools.sidsearch import SID
from reworkd_platform.web.api.agent.tools.tool import Tool

# From tools/tools.py
def get_available_tools() -> List[Type[Tool]]:
    return get_external_tools() + get_default_tools()

# From tools/tools.py
def get_available_tools_names() -> List[str]:
    return [get_tool_name(tool) for tool in get_available_tools()]

# From tools/tools.py
def get_external_tools() -> List[Type[Tool]]:
    return [
        # Wikipedia,  # TODO: Remove if async doesn't work
        Image,
        Code,
        SID,
    ]

# From tools/tools.py
def get_default_tools() -> List[Type[Tool]]:
    return [
        Search,
    ]

# From tools/tools.py
def get_tool_name(tool: Type[Tool]) -> str:
    return format_tool_name(tool.__name__)

# From tools/tools.py
def format_tool_name(tool_name: str) -> str:
    return tool_name.lower()

# From tools/tools.py
def get_tools_overview(tools: List[Type[Tool]]) -> str:
    """Return a formatted string of name: description pairs for all available tools"""

    # Create a list of formatted strings
    formatted_strings = [
        f"'{get_tool_name(tool)}': {tool.description}" for tool in tools
    ]

    # Remove duplicates by converting the list to a set and back to a list
    unique_strings = list(set(formatted_strings))

    # Join the unique strings with newlines
    return "\n".join(unique_strings)

# From tools/tools.py
def get_tool_from_name(tool_name: str) -> Type[Tool]:
    for tool in get_available_tools():
        if get_tool_name(tool) == format_tool_name(tool_name):
            return tool

    return get_default_tool()

# From tools/tools.py
def get_default_tool() -> Type[Tool]:
    return Search

# From tools/tools.py
def get_default_tool_name() -> str:
    return get_tool_name(get_default_tool())

from  import files
from dirty_json import DirtyJson
import regex

# From helpers/extract_tools.py
def json_parse_dirty(json:str) -> dict[str,Any] | None:
    ext_json = extract_json_object_string(json)
    if ext_json:
        # ext_json = fix_json_string(ext_json)
        data = DirtyJson.parse_string(ext_json)
        if isinstance(data,dict): return data
    return None

# From helpers/extract_tools.py
def extract_json_object_string(content):
    start = content.find('{')
    if start == -1:
        return ""

    # Find the first '{'
    end = content.rfind('}')
    if end == -1:
        # If there's no closing '}', return from start to the end
        return content[start:]
    else:
        # If there's a closing '}', return the substring from start to end
        return content[start:end+1]

# From helpers/extract_tools.py
def extract_json_string(content):
    # Regular expression pattern to match a JSON object
    pattern = r'\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\]|"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    
    # Search for the pattern in the content
    match = regex.search(pattern, content)
    
    if match:
        # Return the matched JSON string
        return match.group(0)
    else:
        print("No JSON content found.")
        return ""

# From helpers/extract_tools.py
def fix_json_string(json_string):
    # Function to replace unescaped line breaks within JSON string values
    def replace_unescaped_newlines(match):
        return match.group(0).replace('\n', '\\n')

    # Use regex to find string values and apply the replacement function
    fixed_string = re.sub(r'(?<=: ")(.*?)(?=")', replace_unescaped_newlines, json_string, flags=re.DOTALL)
    return fixed_string

# From helpers/extract_tools.py
def replace_unescaped_newlines(match):
        return match.group(0).replace('\n', '\\n')

