# Merged file for agent_frameworks/memory
# This file contains code merged from multiple repositories

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Generic
from typing import Optional
from typing import TypeVar
from pydantic import BaseModel
from typing_extensions import Self
from _component_config import Component
from _component_config import ComponentBase

# From autogen_core/_cache_store.py
class CacheStore(ABC, Generic[T], ComponentBase[BaseModel]):
    """
    This protocol defines the basic interface for store/cache operations.

    Sub-classes should handle the lifecycle of underlying storage.
    """

    component_type = "cache_store"

    @abstractmethod
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Retrieve an item from the store.

        Args:
            key: The key identifying the item in the store.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The value associated with the key if found, else the default value.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: T) -> None:
        """
        Set an item in the store.

        Args:
            key: The key under which the item is to be stored.
            value: The value to be stored in the store.
        """
        ...

# From autogen_core/_cache_store.py
class InMemoryStoreConfig(BaseModel):
    pass

# From autogen_core/_cache_store.py
class InMemoryStore(CacheStore[T], Component[InMemoryStoreConfig]):
    component_provider_override = "autogen_core.InMemoryStore"
    component_config_schema = InMemoryStoreConfig

    def __init__(self) -> None:
        self.store: Dict[str, T] = {}

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        return self.store.get(key, default)

    def set(self, key: str, value: T) -> None:
        self.store[key] = value

    def _to_config(self) -> InMemoryStoreConfig:
        return InMemoryStoreConfig()

    @classmethod
    def _from_config(cls, config: InMemoryStoreConfig) -> Self:
        return cls()

# From autogen_core/_cache_store.py
def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Retrieve an item from the store.

        Args:
            key: The key identifying the item in the store.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The value associated with the key if found, else the default value.
        """
        ...

# From autogen_core/_cache_store.py
def set(self, key: str, value: T) -> None:
        """
        Set an item in the store.

        Args:
            key: The key under which the item is to be stored.
            value: The value to be stored in the store.
        """
        ...

from typing import Any
from typing import List
from pydantic import Field
from _cancellation_token import CancellationToken
from model_context import ChatCompletionContext
from models import SystemMessage
from _base_memory import Memory
from _base_memory import MemoryContent
from _base_memory import MemoryQueryResult
from _base_memory import UpdateContextResult

# From memory/_list_memory.py
class ListMemoryConfig(BaseModel):
    """Configuration for ListMemory component."""

    name: str | None = None
    """Optional identifier for this memory instance."""
    memory_contents: List[MemoryContent] = Field(default_factory=list)
    """List of memory contents stored in this memory instance."""

# From memory/_list_memory.py
class ListMemory(Memory, Component[ListMemoryConfig]):
    """Simple chronological list-based memory implementation.

    This memory implementation stores contents in a list and retrieves them in
    chronological order. It has an `update_context` method that updates model contexts
    by appending all stored memories.

    The memory content can be directly accessed and modified through the content property,
    allowing external applications to manage memory contents directly.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import ListMemory, MemoryContent
            from autogen_core.model_context import BufferedChatCompletionContext


            async def main() -> None:
                # Initialize memory
                memory = ListMemory(name="chat_history")

                # Add memory content
                content = MemoryContent(content="User prefers formal language", mime_type="text/plain")
                await memory.add(content)

                # Directly modify memory contents
                memory.content = [MemoryContent(content="New preference", mime_type="text/plain")]

                # Create a model context
                model_context = BufferedChatCompletionContext(buffer_size=10)

                # Update a model context with memory
                await memory.update_context(model_context)

                # See the updated model context
                print(await model_context.get_messages())


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance

    """

    component_type = "memory"
    component_provider_override = "autogen_core.memory.ListMemory"
    component_config_schema = ListMemoryConfig

    def __init__(self, name: str | None = None, memory_contents: List[MemoryContent] | None = None) -> None:
        self._name = name or "default_list_memory"
        self._contents: List[MemoryContent] = memory_contents if memory_contents is not None else []

    @property
    def name(self) -> str:
        """Get the memory instance identifier.

        Returns:
            str: Memory instance name
        """
        return self._name

    @property
    def content(self) -> List[MemoryContent]:
        """Get the current memory contents.

        Returns:
            List[MemoryContent]: List of stored memory contents
        """
        return self._contents

    @content.setter
    def content(self, value: List[MemoryContent]) -> None:
        """Set the memory contents.

        Args:
            value: New list of memory contents to store
        """
        self._contents = value

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Update the model context by appending memory content.

        This method mutates the provided model_context by adding all memories as a
        SystemMessage.

        Args:
            model_context: The context to update. Will be mutated if memories exist.

        Returns:
            UpdateContextResult containing the memories that were added to the context
        """

        if not self._contents:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(self._contents, 1)]

        if memory_strings:
            memory_context = "\nRelevant memory content (in chronological order):\n" + "\n".join(memory_strings) + "\n"
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=MemoryQueryResult(results=self._contents))

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return all memories without any filtering.

        Args:
            query: Ignored in this implementation
            cancellation_token: Optional token to cancel operation
            **kwargs: Additional parameters (ignored)

        Returns:
            MemoryQueryResult containing all stored memories
        """
        _ = query, cancellation_token, kwargs
        return MemoryQueryResult(results=self._contents)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add new content to memory.

        Args:
            content: Memory content to store
            cancellation_token: Optional token to cancel operation
        """
        self._contents.append(content)

    async def clear(self) -> None:
        """Clear all memory content."""
        self._contents = []

    async def close(self) -> None:
        """Cleanup resources if needed."""
        pass

    @classmethod
    def _from_config(cls, config: ListMemoryConfig) -> Self:
        return cls(name=config.name, memory_contents=config.memory_contents)

    def _to_config(self) -> ListMemoryConfig:
        return ListMemoryConfig(name=self.name, memory_contents=self._contents)

# From memory/_list_memory.py
def name(self) -> str:
        """Get the memory instance identifier.

        Returns:
            str: Memory instance name
        """
        return self._name

# From memory/_list_memory.py
def content(self) -> List[MemoryContent]:
        """Get the current memory contents.

        Returns:
            List[MemoryContent]: List of stored memory contents
        """
        return self._contents

from enum import Enum
from typing import Union
from pydantic import ConfigDict
from pydantic import field_serializer
from _image import Image

# From memory/_base_memory.py
class MemoryMimeType(Enum):
    """Supported MIME types for memory content."""

    TEXT = "text/plain"
    JSON = "application/json"
    MARKDOWN = "text/markdown"
    IMAGE = "image/*"
    BINARY = "application/octet-stream"

# From memory/_base_memory.py
class MemoryContent(BaseModel):
    """A memory content item."""

    content: ContentType
    """The content of the memory item. It can be a string, bytes, dict, or :class:`~autogen_core.Image`."""

    mime_type: MemoryMimeType | str
    """The MIME type of the memory content."""

    metadata: Dict[str, Any] | None = None
    """Metadata associated with the memory item."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("mime_type")
    def serialize_mime_type(self, mime_type: MemoryMimeType | str) -> str:
        """Serialize the MIME type to a string."""
        if isinstance(mime_type, MemoryMimeType):
            return mime_type.value
        return mime_type

# From memory/_base_memory.py
class MemoryQueryResult(BaseModel):
    """Result of a memory :meth:`~autogen_core.memory.Memory.query` operation."""

    results: List[MemoryContent]

# From memory/_base_memory.py
class UpdateContextResult(BaseModel):
    """Result of a memory :meth:`~autogen_core.memory.Memory.update_context` operation."""

    memories: MemoryQueryResult

# From memory/_base_memory.py
class Memory(ABC, ComponentBase[BaseModel]):
    """Protocol defining the interface for memory implementations.

    A memory is the storage for data that can be used to enrich or modify the model context.

    A memory implementation can use any storage mechanism, such as a list, a database, or a file system.
    It can also use any retrieval mechanism, such as vector search or text search.
    It is up to the implementation to decide how to store and retrieve data.

    It is also a memory implementation's responsibility to update the model context
    with relevant memory content based on the current model context and querying the memory store.

    See :class:`~autogen_core.memory.ListMemory` for an example implementation.
    """

    component_type = "memory"

    @abstractmethod
    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """
        Update the provided model context using relevant memory content.

        Args:
            model_context: The context to update.

        Returns:
            UpdateContextResult containing relevant memories
        """
        ...

    @abstractmethod
    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """
        Query the memory store and return relevant entries.

        Args:
            query: Query content item
            cancellation_token: Optional token to cancel operation
            **kwargs: Additional implementation-specific parameters

        Returns:
            MemoryQueryResult containing memory entries with relevance scores
        """
        ...

    @abstractmethod
    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """
        Add a new content to memory.

        Args:
            content: The memory content to add
            cancellation_token: Optional token to cancel operation
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from memory."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any resources used by the memory implementation."""
        ...

# From memory/_base_memory.py
def serialize_mime_type(self, mime_type: MemoryMimeType | str) -> str:
        """Serialize the MIME type to a string."""
        if isinstance(mime_type, MemoryMimeType):
            return mime_type.value
        return mime_type

from autogen_core import CancellationToken
from autogen_core.memory import Memory
from autogen_core.memory import MemoryContent
from autogen_core.memory import MemoryMimeType
from autogen_core.memory import MemoryQueryResult
from autogen_core.memory import UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from _canvas_writer import ApplyPatchTool
from _canvas_writer import UpdateFileTool
from _text_canvas import TextCanvas

# From canvas/_text_canvas_memory.py
class TextCanvasMemory(Memory):
    """
    A memory implementation that uses a Canvas for storing file-like content.
    Inserts the current state of the canvas into the ChatCompletionContext on each turn.

    .. warning::

        This is an experimental API and may change in the future.

    The TextCanvasMemory provides a persistent, file-like storage mechanism that can be used
    by agents to read and write content. It automatically injects the current state of all files
    in the canvas into the model context before each inference.

    This is particularly useful for:
    - Allowing agents to create and modify documents over multiple turns
    - Enabling collaborative document editing between multiple agents
    - Maintaining persistent state across conversation turns
    - Working with content too large to fit in a single message

    The canvas provides tools for:
    - Creating or updating files with new content
    - Applying patches (unified diff format) to existing files

    Examples:

        **Example: Using TextCanvasMemory with an AssistantAgent**

        The following example demonstrates how to create a TextCanvasMemory and use it with
        an AssistantAgent to write and update a story file.

        .. code-block:: python

            import asyncio
            from autogen_core import CancellationToken
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.messages import TextMessage
            from autogen_ext.memory.canvas import TextCanvasMemory


            async def main():
                # Create a model client
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    # api_key = "your_openai_api_key"
                )

                # Create the canvas memory
                text_canvas_memory = TextCanvasMemory()

                # Get tools for working with the canvas
                update_file_tool = text_canvas_memory.get_update_file_tool()
                apply_patch_tool = text_canvas_memory.get_apply_patch_tool()

                # Create an agent with the canvas memory and tools
                writer_agent = AssistantAgent(
                    name="Writer",
                    model_client=model_client,
                    description="A writer agent that creates and updates stories.",
                    system_message='''
                    You are a Writer Agent. Your focus is to generate a story based on the user's request.

                    Instructions for using the canvas:

                    - The story should be stored on the canvas in a file named "story.md".
                    - If "story.md" does not exist, create it by calling the 'update_file' tool.
                    - If "story.md" already exists, generate a unified diff (patch) from the current
                      content to the new version, and call the 'apply_patch' tool to apply the changes.

                    IMPORTANT: Do not include the full story text in your chat messages.
                    Only write the story content to the canvas using the tools.
                    ''',
                    tools=[update_file_tool, apply_patch_tool],
                    memory=[text_canvas_memory],
                )

                # Send a message to the agent
                await writer_agent.on_messages(
                    [TextMessage(content="Write a short story about a bunny and a sunflower.", source="user")],
                    CancellationToken(),
                )

                # Retrieve the content from the canvas
                story_content = text_canvas_memory.canvas.get_latest_content("story.md")
                print("Story content from canvas:")
                print(story_content)


            if __name__ == "__main__":
                asyncio.run(main())

        **Example: Using TextCanvasMemory with multiple agents**

        The following example shows how to use TextCanvasMemory with multiple agents
        collaborating on the same document.

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_agentchat.conditions import TextMentionTermination
            from autogen_ext.memory.canvas import TextCanvasMemory


            async def main():
                # Create a model client
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    # api_key = "your_openai_api_key"
                )

                # Create the shared canvas memory
                text_canvas_memory = TextCanvasMemory()
                update_file_tool = text_canvas_memory.get_update_file_tool()
                apply_patch_tool = text_canvas_memory.get_apply_patch_tool()

                # Create a writer agent
                writer_agent = AssistantAgent(
                    name="Writer",
                    model_client=model_client,
                    description="A writer agent that creates stories.",
                    system_message="You write children's stories on the canvas in story.md.",
                    tools=[update_file_tool, apply_patch_tool],
                    memory=[text_canvas_memory],
                )

                # Create a critique agent
                critique_agent = AssistantAgent(
                    name="Critique",
                    model_client=model_client,
                    description="A critique agent that provides feedback on stories.",
                    system_message="You review the story.md file and provide constructive feedback.",
                    memory=[text_canvas_memory],
                )

                # Create a team with both agents
                team = RoundRobinGroupChat(
                    participants=[writer_agent, critique_agent],
                    termination_condition=TextMentionTermination("TERMINATE"),
                    max_turns=10,
                )

                # Run the team on a task
                await team.run(task="Create a children's book about a bunny and a sunflower")

                # Get the final story
                story = text_canvas_memory.canvas.get_latest_content("story.md")
                print(story)


            if __name__ == "__main__":
                asyncio.run(main())
    """

    def __init__(self, canvas: Optional[TextCanvas] = None):
        super().__init__()
        self.canvas = canvas if canvas is not None else TextCanvas()

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """
        Inject the entire canvas summary (or a selected subset) as reference data.
        Here, we just put it into a system message, but you could customize.
        """
        snapshot = self.canvas.get_all_contents_for_context()
        if snapshot.strip():
            msg = SystemMessage(content=snapshot)
            await model_context.add_message(msg)

            # Return it for debugging/logging
            memory_content = MemoryContent(content=snapshot, mime_type=MemoryMimeType.TEXT)
            return UpdateContextResult(memories=MemoryQueryResult(results=[memory_content]))

        return UpdateContextResult(memories=MemoryQueryResult(results=[]))

    async def query(
        self, query: str | MemoryContent, cancellation_token: Optional[CancellationToken] = None, **kwargs: Any
    ) -> MemoryQueryResult:
        """
        Potentially search for matching filenames or file content.
        This example returns empty.
        """
        return MemoryQueryResult(results=[])

    async def add(self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        Example usage: Possibly interpret content as a patch or direct file update.
        Could also be done by a specialized "CanvasTool" instead.
        """
        # NO-OP here, leaving actual changes to the CanvasTool
        pass

    async def clear(self) -> None:
        """Clear the entire canvas by replacing it with a new empty instance."""
        # Create a new TextCanvas instance instead of calling __init__ directly
        self.canvas = TextCanvas()

    async def close(self) -> None:
        pass

    def get_update_file_tool(self) -> UpdateFileTool:
        """
        Returns an UpdateFileTool instance that works with this memory's canvas.
        """
        return UpdateFileTool(self.canvas)

    def get_apply_patch_tool(self) -> ApplyPatchTool:
        """
        Returns an ApplyPatchTool instance that works with this memory's canvas.
        """
        return ApplyPatchTool(self.canvas)

# From canvas/_text_canvas_memory.py
def get_update_file_tool(self) -> UpdateFileTool:
        """
        Returns an UpdateFileTool instance that works with this memory's canvas.
        """
        return UpdateFileTool(self.canvas)

# From canvas/_text_canvas_memory.py
def get_apply_patch_tool(self) -> ApplyPatchTool:
        """
        Returns an ApplyPatchTool instance that works with this memory's canvas.
        """
        return ApplyPatchTool(self.canvas)

import logging
from typing import Literal
from autogen_core import Component
from redis import Redis
from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.utils.utils import deserialize
from redisvl.utils.utils import serialize

# From redis/_redis_memory.py
class RedisMemoryConfig(BaseModel):
    """
    Configuration for Redis-based vector memory.

    This class defines the configuration options for using Redis as a vector memory store,
    supporting semantic memory. It allows customization of the Redis connection, index settings,
    similarity search parameters, and embedding model.
    """

    redis_url: str = Field(default="redis://localhost:6379", description="url of the Redis instance")
    index_name: str = Field(default="chat_history", description="Name of the Redis collection")
    prefix: str = Field(default="memory", description="prefix of the Redis collection")
    distance_metric: Literal["cosine", "ip", "l2"] = "cosine"
    algorithm: Literal["flat", "hnsw"] = "flat"
    top_k: int = Field(default=10, description="Number of results to return in queries")
    datatype: Literal["uint8", "int8", "float16", "float32", "float64", "bfloat16"] = "float32"
    distance_threshold: float = Field(default=0.7, description="Minimum similarity score threshold")
    model_name: str | None = Field(
        default="sentence-transformers/all-mpnet-base-v2", description="Embedding model name"
    )

# From redis/_redis_memory.py
class RedisMemory(Memory, Component[RedisMemoryConfig]):
    """
    Store and retrieve memory using vector similarity search powered by RedisVL.

    `RedisMemory` provides a vector-based memory implementation that uses RedisVL for storing and
    retrieving content based on semantic similarity. It enhances agents with the ability to recall
    contextually relevant information during conversations by leveraging vector embeddings to find
    similar content.

        This implementation requires the RedisVL extra to be installed. Install with:

        .. code-block:: bash

            pip install "autogen-ext[redisvl]"

        Additionally, you will need access to a Redis instance.
        To run a local instance of redis in docker:

        .. code-block:: bash

            docker run -d --name redis -p 6379:6379 redis:8

        To download and run Redis locally:

        .. code-block:: bash

            curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
            sudo apt-get update  > /dev/null 2>&1
            sudo apt-get install redis-server  > /dev/null 2>&1
            redis-server --daemonize yes

    Args:
        config (RedisMemoryConfig | None): Configuration for the Redis memory.
            If None, defaults to a RedisMemoryConfig with recommended settings.

    Example:

        .. code-block:: python

            from logging import WARNING, getLogger

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.redis import RedisMemory, RedisMemoryConfig
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            logger = getLogger()
            logger.setLevel(WARNING)


            # Define tool to use
            async def get_weather(city: str, units: str = "imperial") -> str:
                if units == "imperial":
                    return f"The weather in {city} is 73 °F and Sunny."
                elif units == "metric":
                    return f"The weather in {city} is 23 °C and Sunny."
                else:
                    return f"Sorry, I don't know the weather in {city}."


            async def main():
                # Initailize Redis memory
                redis_memory = RedisMemory(
                    config=RedisMemoryConfig(
                        redis_url="redis://localhost:6379",
                        index_name="chat_history",
                        prefix="memory",
                    )
                )

                # Add user preferences to memory
                await redis_memory.add(
                    MemoryContent(
                        content="The weather should be in metric units",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"category": "preferences", "type": "units"},
                    )
                )

                await redis_memory.add(
                    MemoryContent(
                        content="Meal recipe must be vegan",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"category": "preferences", "type": "dietary"},
                    )
                )

                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                )

                # Create assistant agent with ChromaDB memory
                assistant_agent = AssistantAgent(
                    name="assistant_agent",
                    model_client=model_client,
                    tools=[get_weather],
                    memory=[redis_memory],
                )

                stream = assistant_agent.run_stream(task="What is the weather in New York?")
                await Console(stream)

                await model_client.close()
                await redis_memory.close()


            asyncio.run(main())

        Output:

        .. code-block:: text

            ---------- TextMessage (user) ----------
            What is the weather in New York?
            ---------- MemoryQueryEvent (assistant_agent) ----------
            [MemoryContent(content='The weather should be in metric units', mime_type=<MemoryMimeType.TEXT: 'text/plain'>, metadata={'category': 'preferences', 'type': 'units'})]
            ---------- ToolCallRequestEvent (assistant_agent) ----------
            [FunctionCall(id='call_tyCPvPPAV4SHWhtfpM6UMemr', arguments='{"city":"New York","units":"metric"}', name='get_weather')]
            ---------- ToolCallExecutionEvent (assistant_agent) ----------
            [FunctionExecutionResult(content='The weather in New York is 23 °C and Sunny.', name='get_weather', call_id='call_tyCPvPPAV4SHWhtfpM6UMemr', is_error=False)]
            ---------- ToolCallSummaryMessage (assistant_agent) ----------
            The weather in New York is 23 °C and Sunny.

    """

    component_config_schema = RedisMemoryConfig
    component_provider_override = "autogen_ext.memory.redis_memory.RedisMemory"

    def __init__(self, config: RedisMemoryConfig | None = None) -> None:
        """Initialize RedisMemory."""
        self.config = config or RedisMemoryConfig()
        client = Redis.from_url(url=self.config.redis_url)  # type: ignore[reportUknownMemberType]

        self.message_history = SemanticMessageHistory(name=self.config.index_name, redis_client=client)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """
        Update the model context with relevant memory content.

        This method retrieves memory content relevant to the last message in the context
        and adds it as a system message. This implementation uses the last message in the context
        as a query to find semantically similar memories and adds them all to the context as a
        single system message.

        Args:
            model_context (ChatCompletionContext): The model context to update with relevant
                memories.

        Returns:
            UpdateContextResult: Object containing the memories that were used to update the
                context.
        """
        messages = await model_context.get_messages()
        if messages:
            last_message = str(messages[-1].content)
        else:
            last_message = ""

        query_results = await self.query(last_message)

        stringified_messages = "\n\n".join([str(m.content) for m in query_results.results])

        await model_context.add_message(SystemMessage(content=stringified_messages))

        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a memory content object to Redis.

        .. note::

            To perform semantic search over stored memories RedisMemory creates a vector embedding
            from the content field of a MemoryContent object. This content is assumed to be text,
            JSON, or Markdown, and is passed to the vector embedding model specified in
            RedisMemoryConfig.

        Args:
            content (MemoryContent): The memory content to store within Redis.
            cancellation_token (CancellationToken): Token passed to cease operation. Not used.
        """
        if content.mime_type == MemoryMimeType.TEXT:
            memory_content = content.content
            mime_type = "text/plain"
        elif content.mime_type == MemoryMimeType.JSON:
            memory_content = serialize(content.content)
            mime_type = "application/json"
        elif content.mime_type == MemoryMimeType.MARKDOWN:
            memory_content = content.content
            mime_type = "text/markdown"
        else:
            raise NotImplementedError(
                f"Error: {content.mime_type} is not supported. Only MemoryMimeType.TEXT, MemoryMimeType.JSON, and MemoryMimeType.MARKDOWN are currently supported."
            )
        metadata = {"mime_type": mime_type}
        metadata.update(content.metadata if content.metadata else {})
        self.message_history.add_message(
            {"role": "user", "content": memory_content, "tool_call_id": serialize(metadata)}  # type: ignore[reportArgumentType]
        )

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query memory content based on semantic vector similarity.

        .. note::

            RedisMemory.query() supports additional keyword arguments to improve query performance.
            top_k (int): The maximum number of relevant memories to include. Defaults to 10.
            distance_threshold (float): The maximum distance in vector space to consider a memory
            semantically similar when performining cosine similarity search. Defaults to 0.7.

        Args:
            query (str | MemoryContent): query to perform vector similarity search with. If a
                string is passed, a vector embedding is created from it with the model specified
                in the RedisMemoryConfig. If a MemoryContent object is passed, the content field
                of this object is extracted and a vector embedding is created from it with the
                model specified in the RedisMemoryConfig.
            cancellation_token (CancellationToken): Token passed to cease operation. Not used.

        Returns:
            memoryQueryResult: Object containing memories relevant to the provided query.
        """
        # get the query string, or raise an error for unsupported MemoryContent types
        if isinstance(query, str):
            prompt = query
        elif isinstance(query, MemoryContent):
            if query.mime_type in (MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN):
                prompt = str(query.content)
            elif query.mime_type == MemoryMimeType.JSON:
                prompt = serialize(query.content)
            else:
                raise NotImplementedError(
                    f"Error: {query.mime_type} is not supported. Only MemoryMimeType.TEXT, MemoryMimeType.JSON, MemoryMimeType.MARKDOWN are currently supported."
                )
        else:
            raise TypeError("'query' must be either a string or MemoryContent")

        top_k = kwargs.pop("top_k", self.config.top_k)
        distance_threshold = kwargs.pop("distance_threshold", self.config.distance_threshold)

        results = self.message_history.get_relevant(
            prompt=prompt,  # type: ignore[reportArgumentType]
            top_k=top_k,
            distance_threshold=distance_threshold,
            raw=False,
        )

        memories: List[MemoryContent] = []
        for result in results:
            metadata = deserialize(result["tool_call_id"])  # type: ignore[reportArgumentType]
            mime_type = MemoryMimeType(metadata.pop("mime_type"))
            if mime_type in (MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN):
                memory_content = result["content"]  # type: ignore[reportArgumentType]
            elif mime_type == MemoryMimeType.JSON:
                memory_content = deserialize(result["content"])  # type: ignore[reportArgumentType]
            else:
                raise NotImplementedError(
                    f"Error: {mime_type} is not supported. Only MemoryMimeType.TEXT, MemoryMimeType.JSON, and MemoryMimeType.MARKDOWN are currently supported."
                )
            memory = MemoryContent(
                content=memory_content,  # type: ignore[reportArgumentType]
                mime_type=mime_type,
                metadata=metadata,
            )
            memories.append(memory)  # type: ignore[reportUknownMemberType]

        return MemoryQueryResult(results=memories)  # type: ignore[reportUknownMemberType]

    async def clear(self) -> None:
        """Clear all entries from memory, preserving the RedisMemory resources."""
        self.message_history.clear()

    async def close(self) -> None:
        """Clears all entries from memory, and cleans up Redis client, index and resources."""
        self.message_history.delete()

import io
import uuid
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from datetime import datetime
from typing import TypedDict
from typing import cast
from autogen_core import ComponentBase
from mem0 import Memory
from mem0 import MemoryClient
import json

# From mem0/_mem0.py
class Mem0MemoryConfig(BaseModel):
    """Configuration for Mem0Memory component."""

    user_id: Optional[str] = Field(
        default=None, description="User ID for memory operations. If not provided, a UUID will be generated."
    )
    limit: int = Field(default=10, description="Maximum number of results to return in memory queries.")
    is_cloud: bool = Field(default=True, description="Whether to use cloud Mem0 client (True) or local client (False).")
    api_key: Optional[str] = Field(
        default=None, description="API key for cloud Mem0 client. Required if is_cloud=True."
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Configuration dictionary for local Mem0 client. Required if is_cloud=False."
    )

# From mem0/_mem0.py
class MemoryResult(TypedDict, total=False):
    memory: str
    score: float
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    categories: List[str]

# From mem0/_mem0.py
class Mem0Memory(Memory, Component[Mem0MemoryConfig], ComponentBase[Mem0MemoryConfig]):
    """Mem0 memory implementation for AutoGen.

    This component integrates with Mem0.ai's memory system, providing an implementation
    of AutoGen's Memory interface. It supports both cloud and local backends through the
    mem0ai Python package.

    To use this component, you need to have the `mem0` (for cloud-only) or `mem0-local` (for local)
    extra installed for the `autogen-ext` package:

    .. code-block:: bash

        pip install -U "autogen-ext[mem0]" # For cloud-based Mem0
        pip install -U "autogen-ext[mem0-local]" # For local Mem0

    The memory component can store and retrieve information that agents need to remember
    across conversations. It also provides context updating for language models with
    relevant memories.

    Examples:

        .. code-block:: python

            import asyncio
            from autogen_ext.memory.mem0 import Mem0Memory
            from autogen_core.memory import MemoryContent


            async def main() -> None:
                # Create a local Mem0Memory (no API key required)
                memory = Mem0Memory(
                    is_cloud=False,
                    config={"path": ":memory:"},  # Use in-memory storage for testing
                )
                print("Memory initialized successfully!")

                # Add something to memory
                test_content = "User likes the color blue."
                await memory.add(MemoryContent(content=test_content, mime_type="text/plain"))
                print(f"Added content: {test_content}")

                # Retrieve memories with a search query
                results = await memory.query("What color does the user like?")
                print(f"Query results: {len(results.results)} found")

                for i, result in enumerate(results.results):
                    print(f"Result {i+1}: {result}")


            asyncio.run(main())

        Output:

        .. code-block:: text

            Memory initialized successfully!
            Added content: User likes the color blue.
            Query results: 1 found
            Result 1: content='User likes the color blue' mime_type='text/plain' metadata={'score': 0.6977155806281953, 'created_at': datetime.datetime(2025, 7, 6, 17, 25, 18, 754725, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200)))}

        Using it with an :class:`~autogen_agentchat.agents.AssistantAgent`:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.memory import MemoryContent
            from autogen_ext.memory.mem0 import Mem0Memory
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                # Create a model client
                model_client = OpenAIChatCompletionClient(model="gpt-4.1")

                # Create a Mem0 memory instance
                memory = Mem0Memory(
                    user_id="user123",
                    is_cloud=False,
                    config={"path": ":memory:"},  # Use in-memory storage for testing
                )

                # Add something to memory
                test_content = "User likes the color blue."
                await memory.add(MemoryContent(content=test_content, mime_type="text/plain"))

                # Create an assistant agent with Mem0 memory
                agent = AssistantAgent(
                    name="assistant",
                    model_client=model_client,
                    memory=[memory],
                    system_message="You are a helpful assistant that remembers user preferences.",
                )

                # Run a sample task
                result = await agent.run(task="What color does the user like?")
                print(result.messages[-1].content)  # type: ignore


            asyncio.run(main())

        Output:

        .. code-block:: text

            User likes the color blue.

    Args:
        user_id: Optional user ID for memory operations. If not provided, a UUID will be generated.
        limit: Maximum number of results to return in memory queries.
        is_cloud: Whether to use cloud Mem0 client (True) or local client (False).
        api_key: API key for cloud Mem0 client. It will read from the environment MEM0_API_KEY if not provided.
        config: Configuration dictionary for local Mem0 client. Required if is_cloud=False.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.mem0.Mem0Memory"
    component_config_schema = Mem0MemoryConfig

    def __init__(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
        is_cloud: bool = True,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate parameters
        if not is_cloud and config is None:
            raise ValueError("config is required when using local Mem0 client (is_cloud=False)")

        # Initialize instance variables
        self._user_id = user_id or str(uuid.uuid4())
        self._limit = limit
        self._is_cloud = is_cloud
        self._api_key = api_key
        self._config = config

        # Initialize client
        if self._is_cloud:
            self._client = MemoryClient(api_key=self._api_key)
        else:
            assert self._config is not None
            config_dict = self._config
            self._client = Memory0.from_config(config_dict=config_dict)  # type: ignore

    @property
    def user_id(self) -> str:
        """Get the user ID for memory operations."""
        return self._user_id

    @property
    def limit(self) -> int:
        """Get the maximum number of results to return in memory queries."""
        return self._limit

    @property
    def is_cloud(self) -> bool:
        """Check if the Mem0 client is cloud-based."""
        return self._is_cloud

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration for the Mem0 client."""
        return self._config

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> None:
        """Add content to memory.

        Args:
            content: The memory content to add.
            cancellation_token: Optional token to cancel operation.

        Raises:
            Exception: If there's an error adding content to mem0 memory.
        """
        # Extract content based on mime type
        if hasattr(content, "content") and hasattr(content, "mime_type"):
            if content.mime_type in ["text/plain", "text/markdown"]:
                message = str(content.content)
            elif content.mime_type == "application/json":
                # Convert JSON content to string representation
                if isinstance(content.content, str):
                    message = content.content
                else:
                    # Convert dict or other JSON serializable objects to string
                    import json

                    message = json.dumps(content.content)
            else:
                message = str(content.content)

            # Extract metadata
            metadata = content.metadata or {}
        else:
            # Handle case where content is directly provided as string
            message = str(content)
            metadata = {}

        # Check if operation is cancelled
        if cancellation_token is not None and cancellation_token.cancelled:  # type: ignore
            return

        # Add to mem0 client
        try:
            user_id = metadata.pop("user_id", self._user_id)
            # Suppress warning messages from mem0 MemoryClient
            kwargs = {} if self._client.__class__.__name__ == "Memory" else {"output_format": "v1.1"}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self._client.add([{"role": "user", "content": message}], user_id=user_id, metadata=metadata, **kwargs)  # type: ignore
        except Exception as e:
            # Log the error but don't crash
            logger.error(f"Error adding to mem0 memory: {str(e)}")
            raise

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query memory for relevant content.

        Args:
            query: The query to search for, either as string or MemoryContent.
            cancellation_token: Optional token to cancel operation.
            **kwargs: Additional query parameters to pass to mem0.

        Returns:
            MemoryQueryResult containing search results.
        """
        # Extract query text
        if isinstance(query, str):
            query_text = query
        elif hasattr(query, "content"):
            query_text = str(query.content)
        else:
            query_text = str(query)

        # Check if operation is cancelled
        if (
            cancellation_token
            and hasattr(cancellation_token, "cancelled")
            and getattr(cancellation_token, "cancelled", False)
        ):
            return MemoryQueryResult(results=[])

        try:
            limit = kwargs.pop("limit", self._limit)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                # Query mem0 client
                results = self._client.search(  # type: ignore
                    query_text,
                    user_id=self._user_id,
                    limit=limit,
                    **kwargs,
                )

                # Type-safe handling of results
                if isinstance(results, dict) and "results" in results:
                    result_list = cast(List[MemoryResult], results["results"])
                else:
                    result_list = cast(List[MemoryResult], results)

            # Convert results to MemoryContent objects
            memory_contents: List[MemoryContent] = []
            for result in result_list:
                content_text = result.get("memory", "")
                metadata: Dict[str, Any] = {}

                if "metadata" in result and result["metadata"]:
                    metadata = result["metadata"]

                # Add relevant fields to metadata
                if "score" in result:
                    metadata["score"] = result["score"]

                # For created_at
                if "created_at" in result and result.get("created_at"):
                    try:
                        metadata["created_at"] = datetime.fromisoformat(result["created_at"])
                    except (ValueError, TypeError):
                        pass

                # For updated_at
                if "updated_at" in result and result.get("updated_at"):
                    try:
                        metadata["updated_at"] = datetime.fromisoformat(result["updated_at"])
                    except (ValueError, TypeError):
                        pass

                # For categories
                if "categories" in result and result.get("categories"):
                    metadata["categories"] = result["categories"]

                # Create MemoryContent object
                memory_content = MemoryContent(
                    content=content_text,
                    mime_type="text/plain",  # Default to text/plain
                    metadata=metadata,
                )
                memory_contents.append(memory_content)

            return MemoryQueryResult(results=memory_contents)

        except Exception as e:
            # Log the error but return empty results
            logger.error(f"Error querying mem0 memory: {str(e)}")
            return MemoryQueryResult(results=[])

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Update the model context with relevant memories.

        This method retrieves the conversation history from the model context,
        uses the last message as a query to find relevant memories, and then
        adds those memories to the context as a system message.

        Args:
            model_context: The model context to update.

        Returns:
            UpdateContextResult containing memories added to the context.
        """
        # Get messages from context
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Use the last message as query
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        # Query memory
        query_results = await self.query(query_text, limit=self._limit)

        # If we have results, add them to the context
        if query_results.results:
            # Format memories as numbered list
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memories:\n" + "\n".join(memory_strings)

            # Add as system message
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    async def clear(self) -> None:
        """Clear all content from memory for the current user.

        Raises:
            Exception: If there's an error clearing mem0 memory.
        """
        try:
            self._client.delete_all(user_id=self._user_id)  # type: ignore
        except Exception as e:
            logger.error(f"Error clearing mem0 memory: {str(e)}")
            raise

    async def close(self) -> None:
        """Clean up resources if needed.

        This is a no-op for Mem0 clients as they don't require explicit cleanup.
        """
        pass

    @classmethod
    def _from_config(cls, config: Mem0MemoryConfig) -> Self:
        """Create instance from configuration.

        Args:
            config: Configuration for Mem0Memory component.

        Returns:
            A new Mem0Memory instance.
        """
        return cls(
            user_id=config.user_id,
            limit=config.limit,
            is_cloud=config.is_cloud,
            api_key=config.api_key,
            config=config.config,
        )

    def _to_config(self) -> Mem0MemoryConfig:
        """Convert instance to configuration.

        Returns:
            Configuration representing this Mem0Memory instance.
        """
        return Mem0MemoryConfig(
            user_id=self._user_id,
            limit=self._limit,
            is_cloud=self._is_cloud,
            api_key=self._api_key,
            config=self._config,
        )

# From mem0/_mem0.py
def user_id(self) -> str:
        """Get the user ID for memory operations."""
        return self._user_id

# From mem0/_mem0.py
def limit(self) -> int:
        """Get the maximum number of results to return in memory queries."""
        return self._limit

# From mem0/_mem0.py
def is_cloud(self) -> bool:
        """Check if the Mem0 client is cloud-based."""
        return self._is_cloud

# From mem0/_mem0.py
def config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration for the Mem0 client."""
        return self._config

from autogen_core import Image
from chromadb import HttpClient
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Document
from chromadb.api.types import Metadata
from _chroma_configs import ChromaDBVectorMemoryConfig
from _chroma_configs import CustomEmbeddingFunctionConfig
from _chroma_configs import DefaultEmbeddingFunctionConfig
from _chroma_configs import HttpChromaDBVectorMemoryConfig
from _chroma_configs import OpenAIEmbeddingFunctionConfig
from _chroma_configs import PersistentChromaDBVectorMemoryConfig
from _chroma_configs import SentenceTransformerEmbeddingFunctionConfig
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# From chromadb/_chromadb.py
class ChromaDBVectorMemory(Memory, Component[ChromaDBVectorMemoryConfig]):
    """
    Store and retrieve memory using vector similarity search powered by ChromaDB.

    `ChromaDBVectorMemory` provides a vector-based memory implementation that uses ChromaDB for
    storing and retrieving content based on semantic similarity. It enhances agents with the ability
    to recall contextually relevant information during conversations by leveraging vector embeddings
    to find similar content.

    This implementation serves as a reference for more complex memory systems using vector embeddings.
    For advanced use cases requiring specialized formatting of retrieved content, users should extend
    this class and override the `update_context()` method.

    This implementation requires the ChromaDB extra to be installed. Install with:

    .. code-block:: bash

        pip install "autogen-ext[chromadb]"

    Args:
        config (ChromaDBVectorMemoryConfig | None): Configuration for the ChromaDB memory.
            If None, defaults to a PersistentChromaDBVectorMemoryConfig with default values.
            Two config types are supported:
            * PersistentChromaDBVectorMemoryConfig: For local storage
            * HttpChromaDBVectorMemoryConfig: For connecting to a remote ChromaDB server

    Example:

        .. code-block:: python

            import os
            import asyncio
            from pathlib import Path
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.chromadb import (
                ChromaDBVectorMemory,
                PersistentChromaDBVectorMemoryConfig,
                SentenceTransformerEmbeddingFunctionConfig,
                OpenAIEmbeddingFunctionConfig,
            )
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            def get_weather(city: str) -> str:
                return f"The weather in {city} is sunny with a high of 90°F and a low of 70°F."


            def fahrenheit_to_celsius(fahrenheit: float) -> float:
                return (fahrenheit - 32) * 5.0 / 9.0


            async def main() -> None:
                # Use default embedding function
                default_memory = ChromaDBVectorMemory(
                    config=PersistentChromaDBVectorMemoryConfig(
                        collection_name="user_preferences",
                        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
                        k=3,  # Return top 3 results
                        score_threshold=0.5,  # Minimum similarity score
                    )
                )

                # Using a custom SentenceTransformer model
                custom_memory = ChromaDBVectorMemory(
                    config=PersistentChromaDBVectorMemoryConfig(
                        collection_name="multilingual_memory",
                        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
                        embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                            model_name="paraphrase-multilingual-mpnet-base-v2"
                        ),
                    )
                )

                # Using OpenAI embeddings
                openai_memory = ChromaDBVectorMemory(
                    config=PersistentChromaDBVectorMemoryConfig(
                        collection_name="openai_memory",
                        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
                        embedding_function_config=OpenAIEmbeddingFunctionConfig(
                            api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
                        ),
                    )
                )

                # Add user preferences to memory
                await openai_memory.add(
                    MemoryContent(
                        content="The user prefers weather temperatures in Celsius",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"category": "preferences", "type": "units"},
                    )
                )

                # Create assistant agent with ChromaDB memory
                assistant = AssistantAgent(
                    name="assistant",
                    model_client=OpenAIChatCompletionClient(
                        model="gpt-4.1",
                    ),
                    tools=[
                        get_weather,
                        fahrenheit_to_celsius,
                    ],
                    max_tool_iterations=10,
                    memory=[openai_memory],
                )

                # The memory will automatically retrieve relevant content during conversations
                await Console(assistant.run_stream(task="What's the temperature in New York?"))

                # Remember to close the memory when finished
                await default_memory.close()
                await custom_memory.close()
                await openai_memory.close()


            asyncio.run(main())

        Output:

        .. code-block:: text

            ---------- TextMessage (user) ----------
            What's the temperature in New York?
            ---------- MemoryQueryEvent (assistant) ----------
            [MemoryContent(content='The user prefers weather temperatures in Celsius', mime_type='MemoryMimeType.TEXT', metadata={'type': 'units', 'category': 'preferences', 'mime_type': 'MemoryMimeType.TEXT', 'score': 0.3133561611175537, 'id': 'fb00506c-acf4-4174-93d7-2a942593f3f7'}), MemoryContent(content='The user prefers weather temperatures in Celsius', mime_type='MemoryMimeType.TEXT', metadata={'mime_type': 'MemoryMimeType.TEXT', 'category': 'preferences', 'type': 'units', 'score': 0.3133561611175537, 'id': '34311689-b419-4e1a-8bc4-09143f356c66'})]
            ---------- ToolCallRequestEvent (assistant) ----------
            [FunctionCall(id='call_7TjsFd430J1aKwU5T2w8bvdh', arguments='{"city":"New York"}', name='get_weather')]
            ---------- ToolCallExecutionEvent (assistant) ----------
            [FunctionExecutionResult(content='The weather in New York is sunny with a high of 90°F and a low of 70°F.', name='get_weather', call_id='call_7TjsFd430J1aKwU5T2w8bvdh', is_error=False)]
            ---------- ToolCallRequestEvent (assistant) ----------
            [FunctionCall(id='call_RTjMHEZwDXtjurEYTjDlvq9c', arguments='{"fahrenheit": 90}', name='fahrenheit_to_celsius'), FunctionCall(id='call_3mMuCK1aqtzZPTqIHPoHKxtP', arguments='{"fahrenheit": 70}', name='fahrenheit_to_celsius')]
            ---------- ToolCallExecutionEvent (assistant) ----------
            [FunctionExecutionResult(content='32.22222222222222', name='fahrenheit_to_celsius', call_id='call_RTjMHEZwDXtjurEYTjDlvq9c', is_error=False), FunctionExecutionResult(content='21.11111111111111', name='fahrenheit_to_celsius', call_id='call_3mMuCK1aqtzZPTqIHPoHKxtP', is_error=False)]
            ---------- TextMessage (assistant) ----------
            The temperature in New York today is sunny with a high of about 32°C and a low of about 21°C.

    """

    component_config_schema = ChromaDBVectorMemoryConfig
    component_provider_override = "autogen_ext.memory.chromadb.ChromaDBVectorMemory"

    def __init__(self, config: ChromaDBVectorMemoryConfig | None = None) -> None:
        self._config = config or PersistentChromaDBVectorMemoryConfig()
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    @property
    def collection_name(self) -> str:
        """Get the name of the ChromaDB collection."""
        return self._config.collection_name

    def _create_embedding_function(self) -> Any:
        """Create an embedding function based on the configuration.

        Returns:
            A ChromaDB-compatible embedding function.

        Raises:
            ValueError: If the embedding function type is unsupported.
            ImportError: If required dependencies are not installed.
        """
        try:
            from chromadb.utils import embedding_functions
        except ImportError as e:
            raise ImportError(
                "ChromaDB embedding functions not available. Ensure chromadb is properly installed."
            ) from e

        config = self._config.embedding_function_config

        if isinstance(config, DefaultEmbeddingFunctionConfig):
            return embedding_functions.DefaultEmbeddingFunction()

        elif isinstance(config, SentenceTransformerEmbeddingFunctionConfig):
            try:
                return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.model_name)
            except Exception as e:
                raise ImportError(
                    f"Failed to create SentenceTransformer embedding function with model '{config.model_name}'. "
                    f"Ensure sentence-transformers is installed and the model is available. Error: {e}"
                ) from e

        elif isinstance(config, OpenAIEmbeddingFunctionConfig):
            try:
                return embedding_functions.OpenAIEmbeddingFunction(api_key=config.api_key, model_name=config.model_name)
            except Exception as e:
                raise ImportError(
                    f"Failed to create OpenAI embedding function with model '{config.model_name}'. "
                    f"Ensure openai is installed and API key is valid. Error: {e}"
                ) from e

        elif isinstance(config, CustomEmbeddingFunctionConfig):
            try:
                return config.function(**config.params)
            except Exception as e:
                raise ValueError(f"Failed to create custom embedding function. Error: {e}") from e

        else:
            raise ValueError(f"Unsupported embedding function config type: {type(config)}")

    def _ensure_initialized(self) -> None:
        """Ensure ChromaDB client and collection are initialized."""
        if self._client is None:
            try:
                from chromadb.config import Settings

                settings = Settings(allow_reset=self._config.allow_reset)

                if isinstance(self._config, PersistentChromaDBVectorMemoryConfig):
                    self._client = PersistentClient(
                        path=self._config.persistence_path,
                        settings=settings,
                        tenant=self._config.tenant,
                        database=self._config.database,
                    )
                elif isinstance(self._config, HttpChromaDBVectorMemoryConfig):
                    self._client = HttpClient(
                        host=self._config.host,
                        port=self._config.port,
                        ssl=self._config.ssl,
                        headers=self._config.headers,
                        settings=settings,
                        tenant=self._config.tenant,
                        database=self._config.database,
                    )
                else:
                    raise ValueError(f"Unsupported config type: {type(self._config)}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                raise

        if self._collection is None:
            try:
                # Create embedding function
                embedding_function = self._create_embedding_function()

                # Create or get collection with embedding function
                self._collection = self._client.get_or_create_collection(
                    name=self._config.collection_name,
                    metadata={"distance_metric": self._config.distance_metric},
                    embedding_function=embedding_function,
                )
            except Exception as e:
                logger.error(f"Failed to get/create collection: {e}")
                raise

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from content."""
        if isinstance(content_item, str):
            return content_item

        content = content_item.content
        mime_type = content_item.mime_type

        if mime_type in [MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN]:
            return str(content)
        elif mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                # Store original JSON string representation
                return str(content).lower()
            raise ValueError("JSON content must be a dict")
        elif isinstance(content, Image):
            raise ValueError("Image content cannot be converted to text")
        else:
            raise ValueError(f"Unsupported content type: {mime_type}")

    def _calculate_score(self, distance: float) -> float:
        """Convert ChromaDB distance to a similarity score."""
        if self._config.distance_metric == "cosine":
            return 1.0 - (distance / 2.0)
        return 1.0 / (1.0 + distance)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Extract query from last message
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        # Query memory and get results
        query_results = await self.query(query_text)

        if query_results.results:
            # Format results for context
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory content:\n" + "\n".join(memory_strings)

            # Add to context
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        self._ensure_initialized()
        if self._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")

        try:
            # Extract text from content
            text = self._extract_text(content)

            # Use metadata directly from content
            metadata_dict = content.metadata or {}
            metadata_dict["mime_type"] = str(content.mime_type)

            # Add to ChromaDB
            self._collection.add(documents=[text], metadatas=[metadata_dict], ids=[str(uuid.uuid4())])

        except Exception as e:
            logger.error(f"Failed to add content to ChromaDB: {e}")
            raise

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        self._ensure_initialized()
        if self._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")

        try:
            # Extract text for query
            query_text = self._extract_text(query)

            # Query ChromaDB
            results = self._collection.query(
                query_texts=[query_text],
                n_results=self._config.k,
                include=["documents", "metadatas", "distances"],
                **kwargs,
            )

            # Convert results to MemoryContent list
            memory_results: List[MemoryContent] = []

            if (
                not results
                or not results.get("documents")
                or not results.get("metadatas")
                or not results.get("distances")
            ):
                return MemoryQueryResult(results=memory_results)

            documents: List[Document] = results["documents"][0] if results["documents"] else []
            metadatas: List[Metadata] = results["metadatas"][0] if results["metadatas"] else []
            distances: List[float] = results["distances"][0] if results["distances"] else []
            ids: List[str] = results["ids"][0] if results["ids"] else []

            for doc, metadata_dict, distance, doc_id in zip(documents, metadatas, distances, ids, strict=False):
                # Calculate score
                score = self._calculate_score(distance)
                metadata = dict(metadata_dict)
                metadata["score"] = score
                metadata["id"] = doc_id
                if self._config.score_threshold is not None and score < self._config.score_threshold:
                    continue

                # Extract mime_type from metadata
                mime_type = str(metadata_dict.get("mime_type", MemoryMimeType.TEXT.value))

                # Create MemoryContent
                content = MemoryContent(
                    content=doc,
                    mime_type=mime_type,
                    metadata=metadata,
                )
                memory_results.append(content)

            return MemoryQueryResult(results=memory_results)

        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            raise

    async def clear(self) -> None:
        self._ensure_initialized()
        if self._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")

        try:
            results = self._collection.get()
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
            raise

    async def close(self) -> None:
        """Clean up ChromaDB client and resources."""
        self._collection = None
        self._client = None

    async def reset(self) -> None:
        self._ensure_initialized()
        if not self._config.allow_reset:
            raise RuntimeError("Reset not allowed. Set allow_reset=True in config to enable.")

        if self._client is not None:
            try:
                self._client.reset()
            except Exception as e:
                logger.error(f"Error during ChromaDB reset: {e}")
            finally:
                self._collection = None

    def _to_config(self) -> ChromaDBVectorMemoryConfig:
        """Serialize the memory configuration."""

        return self._config

    @classmethod
    def _from_config(cls, config: ChromaDBVectorMemoryConfig) -> Self:
        """Deserialize the memory configuration."""

        return cls(config=config)

# From chromadb/_chromadb.py
def collection_name(self) -> str:
        """Get the name of the ChromaDB collection."""
        return self._config.collection_name

from typing import Callable
from typing_extensions import Annotated

# From chromadb/_chroma_configs.py
class DefaultEmbeddingFunctionConfig(BaseModel):
    """Configuration for the default ChromaDB embedding function.

    Uses ChromaDB's default embedding function (Sentence Transformers all-MiniLM-L6-v2).

    .. versionadded:: v0.4.1
       Support for custom embedding functions in ChromaDB memory.
    """

    function_type: Literal["default"] = "default"

# From chromadb/_chroma_configs.py
class SentenceTransformerEmbeddingFunctionConfig(BaseModel):
    """Configuration for SentenceTransformer embedding functions.

    Allows specifying a custom SentenceTransformer model for embeddings.

    .. versionadded:: v0.4.1
       Support for custom embedding functions in ChromaDB memory.

    Args:
        model_name (str): Name of the SentenceTransformer model to use.
            Defaults to "all-MiniLM-L6-v2".

    Example:
        .. code-block:: python

            from autogen_ext.memory.chromadb import SentenceTransformerEmbeddingFunctionConfig

            _ = SentenceTransformerEmbeddingFunctionConfig(model_name="paraphrase-multilingual-mpnet-base-v2")
    """

    function_type: Literal["sentence_transformer"] = "sentence_transformer"
    model_name: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer model name to use")

# From chromadb/_chroma_configs.py
class OpenAIEmbeddingFunctionConfig(BaseModel):
    """Configuration for OpenAI embedding functions.

    Uses OpenAI's embedding API for generating embeddings.

    .. versionadded:: v0.4.1
       Support for custom embedding functions in ChromaDB memory.

    Args:
        api_key (str): OpenAI API key. If empty, will attempt to use environment variable.
        model_name (str): OpenAI embedding model name. Defaults to "text-embedding-ada-002".

    Example:
        .. code-block:: python

            from autogen_ext.memory.chromadb import OpenAIEmbeddingFunctionConfig

            _ = OpenAIEmbeddingFunctionConfig(api_key="sk-...", model_name="text-embedding-3-small")
    """

    function_type: Literal["openai"] = "openai"
    api_key: str = Field(default="", description="OpenAI API key")
    model_name: str = Field(default="text-embedding-ada-002", description="OpenAI embedding model name")

# From chromadb/_chroma_configs.py
class CustomEmbeddingFunctionConfig(BaseModel):
    """Configuration for custom embedding functions.

    Allows using a custom function that returns a ChromaDB-compatible embedding function.

    .. versionadded:: v0.4.1
       Support for custom embedding functions in ChromaDB memory.

    .. warning::
       Configurations containing custom functions are not serializable.

    Args:
        function (Callable): Function that returns a ChromaDB-compatible embedding function.
        params (Dict[str, Any]): Parameters to pass to the function.
    """

    function_type: Literal["custom"] = "custom"
    function: Callable[..., Any] = Field(description="Function that returns an embedding function")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the function")

# From chromadb/_chroma_configs.py
class ChromaDBVectorMemoryConfig(BaseModel):
    """Base configuration for ChromaDB-based memory implementation.

    .. versionchanged:: v0.4.1
       Added support for custom embedding functions via embedding_function_config.
    """

    client_type: Literal["persistent", "http"]
    collection_name: str = Field(default="memory_store", description="Name of the ChromaDB collection")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity search")
    k: int = Field(default=3, description="Number of results to return in queries")
    score_threshold: float | None = Field(default=None, description="Minimum similarity score threshold")
    allow_reset: bool = Field(default=False, description="Whether to allow resetting the ChromaDB client")
    tenant: str = Field(default="default_tenant", description="Tenant to use")
    database: str = Field(default="default_database", description="Database to use")
    embedding_function_config: EmbeddingFunctionConfig = Field(
        default_factory=DefaultEmbeddingFunctionConfig, description="Configuration for the embedding function"
    )

# From chromadb/_chroma_configs.py
class PersistentChromaDBVectorMemoryConfig(ChromaDBVectorMemoryConfig):
    """Configuration for persistent ChromaDB memory."""

    client_type: Literal["persistent", "http"] = "persistent"
    persistence_path: str = Field(default="./chroma_db", description="Path for persistent storage")

# From chromadb/_chroma_configs.py
class HttpChromaDBVectorMemoryConfig(ChromaDBVectorMemoryConfig):
    """Configuration for HTTP ChromaDB memory."""

    client_type: Literal["persistent", "http"] = "http"
    host: str = Field(default="localhost", description="Host of the remote server")
    port: int = Field(default=8000, description="Port of the remote server")
    ssl: bool = Field(default=False, description="Whether to use HTTPS")
    headers: Dict[str, str] | None = Field(default=None, description="Headers to send to the server")

import os
import pickle
from dataclasses import dataclass
from typing import Tuple
from _string_similarity_map import StringSimilarityMap
from utils.page_logger import PageLogger

# From task_centric_memory/_memory_bank.py
class Memo:
    """
    Represents an atomic unit of memory that can be stored in a memory bank and later retrieved.
    """

    task: str | None  # The task description, if any.
    insight: str

# From task_centric_memory/_memory_bank.py
class MemoryBankConfig(TypedDict, total=False):
    path: str
    relevance_conversion_threshold: float
    n_results: int
    distance_threshold: int

# From task_centric_memory/_memory_bank.py
class MemoryBank:
    """
    Stores task-completion insights as memories in a vector DB for later retrieval.

    Args:
        reset: True to clear the DB before starting.
        config: An optional dict that can be used to override the following values:

            - path: The path to the directory where the memory bank files are stored.
            - relevance_conversion_threshold: The threshold used to normalize relevance.
            - n_results: The maximum number of most relevant results to return for any given topic.
            - distance_threshold: The maximum string-pair distance for a memo to be retrieved.

        logger: An optional logger. If None, no logging will be performed.
    """

    def __init__(
        self,
        reset: bool,
        config: MemoryBankConfig | None = None,
        logger: PageLogger | None = None,
    ) -> None:
        if logger is None:
            logger = PageLogger()  # Nothing will be logged by this object.
        self.logger = logger
        self.logger.enter_function()

        # Apply default settings and any config overrides.
        memory_dir_path = "./memory_bank/default"
        self.relevance_conversion_threshold = 1.7
        self.n_results = 25
        self.distance_threshold = 100
        if config is not None:
            memory_dir_path = config.get("path", memory_dir_path)
            self.relevance_conversion_threshold = config.get(
                "relevance_conversion_threshold", self.relevance_conversion_threshold
            )
            self.n_results = config.get("n_results", self.n_results)
            self.distance_threshold = config.get("distance_threshold", self.distance_threshold)

        memory_dir_path = os.path.expanduser(memory_dir_path)
        self.logger.info("\nMEMORY BANK DIRECTORY  {}".format(memory_dir_path))
        path_to_db_dir = os.path.join(memory_dir_path, "string_map")
        self.path_to_dict = os.path.join(memory_dir_path, "uid_memo_dict.pkl")

        self.string_map = StringSimilarityMap(reset=reset, path_to_db_dir=path_to_db_dir, logger=self.logger)

        # Load or create the associated memo dict on disk.
        self.uid_memo_dict: Dict[str, Memo] = {}
        self.last_memo_id = 0
        if (not reset) and os.path.exists(self.path_to_dict):
            self.logger.info("\nLOADING MEMOS FROM DISK  at {}".format(self.path_to_dict))
            with open(self.path_to_dict, "rb") as f:
                self.uid_memo_dict = pickle.load(f)
                self.last_memo_id = len(self.uid_memo_dict)
                self.logger.info("\n{} MEMOS LOADED".format(len(self.uid_memo_dict)))

        # Clear the DB if requested.
        if reset:
            self._reset_memos()

        self.logger.leave_function()

    def reset(self) -> None:
        """
        Forces immediate deletion of all contents, in memory and on disk.
        """
        self.string_map.reset_db()
        self._reset_memos()

    def _reset_memos(self) -> None:
        """
        Forces immediate deletion of the memos, in memory and on disk.
        """
        self.logger.info("\nCLEARING MEMOS")
        self.uid_memo_dict = {}
        self.save_memos()

    def save_memos(self) -> None:
        """
        Saves the current memo structures (possibly empty) to disk.
        """
        self.string_map.save_string_pairs()
        with open(self.path_to_dict, "wb") as file:
            self.logger.info("\nSAVING MEMOS TO DISK  at {}".format(self.path_to_dict))
            pickle.dump(self.uid_memo_dict, file)

    def contains_memos(self) -> bool:
        """
        Returns True if the memory bank contains any memo.
        """
        return len(self.uid_memo_dict) > 0

    def _map_topics_to_memo(self, topics: List[str], memo_id: str, memo: Memo) -> None:
        """
        Adds a mapping in the vec DB from each topic to the memo.
        """
        self.logger.enter_function()
        self.logger.info("\nINSIGHT\n{}".format(memo.insight))
        for topic in topics:
            self.logger.info("\n TOPIC = {}".format(topic))
            self.string_map.add_input_output_pair(topic, memo_id)
        self.uid_memo_dict[memo_id] = memo
        self.save_memos()
        self.logger.leave_function()

    def add_memo(self, insight_str: str, topics: List[str], task_str: Optional[str] = None) -> None:
        """
        Adds an insight to the memory bank, given topics related to the insight, and optionally the task.
        """
        self.logger.enter_function()
        self.last_memo_id += 1
        id_str = str(self.last_memo_id)
        insight = Memo(insight=insight_str, task=task_str)
        self._map_topics_to_memo(topics, id_str, insight)
        self.logger.leave_function()

    def add_task_with_solution(self, task: str, solution: str, topics: List[str]) -> None:
        """
        Adds a task-solution pair to the memory bank, to be retrieved together later as a combined insight.
        This is useful when the insight is a demonstration of how to solve a given type of task.
        """
        self.logger.enter_function()
        self.last_memo_id += 1
        id_str = str(self.last_memo_id)
        # Prepend the insight to the task description for context.
        insight_str = "Example task:\n\n{}\n\nExample solution:\n\n{}".format(task, solution)
        memo = Memo(insight=insight_str, task=task)
        self._map_topics_to_memo(topics, id_str, memo)
        self.logger.leave_function()

    def get_relevant_memos(self, topics: List[str]) -> List[Memo]:
        """
        Returns any memos from the memory bank that appear sufficiently relevant to the input topics.
        """
        self.logger.enter_function()

        # Retrieve all topic matches, and gather them into a single list.
        matches: List[Tuple[str, str, float]] = []  # Each match is a tuple: (topic, memo_id, distance)
        for topic in topics:
            matches.extend(self.string_map.get_related_string_pairs(topic, self.n_results, self.distance_threshold))

        # Build a dict of memo-relevance pairs from the matches.
        memo_relevance_dict: Dict[str, float] = {}
        for match in matches:
            relevance = self.relevance_conversion_threshold - match[2]
            memo_id = match[1]
            if memo_id in memo_relevance_dict:
                memo_relevance_dict[memo_id] += relevance
            else:
                memo_relevance_dict[memo_id] = relevance

        # Log the details of all the retrieved memos.
        self.logger.info("\n{} POTENTIALLY RELEVANT MEMOS".format(len(memo_relevance_dict)))
        for memo_id, relevance in memo_relevance_dict.items():
            memo = self.uid_memo_dict[memo_id]
            details = ""
            if memo.task is not None:
                details += "\n  TASK: {}\n".format(memo.task)
            details += "\n  INSIGHT: {}\n\n  RELEVANCE: {:.3f}\n".format(memo.insight, relevance)
            self.logger.info(details)

        # Sort the memo-relevance pairs by relevance, in descending order.
        memo_relevance_dict = dict(sorted(memo_relevance_dict.items(), key=lambda item: item[1], reverse=True))

        # Compose the list of sufficiently relevant memos to return.
        memo_list: List[Memo] = []
        for memo_id in memo_relevance_dict:
            if memo_relevance_dict[memo_id] >= 0:
                memo_list.append(self.uid_memo_dict[memo_id])

        self.logger.leave_function()
        return memo_list

# From task_centric_memory/_memory_bank.py
def reset(self) -> None:
        """
        Forces immediate deletion of all contents, in memory and on disk.
        """
        self.string_map.reset_db()
        self._reset_memos()

# From task_centric_memory/_memory_bank.py
def save_memos(self) -> None:
        """
        Saves the current memo structures (possibly empty) to disk.
        """
        self.string_map.save_string_pairs()
        with open(self.path_to_dict, "wb") as file:
            self.logger.info("\nSAVING MEMOS TO DISK  at {}".format(self.path_to_dict))
            pickle.dump(self.uid_memo_dict, file)

# From task_centric_memory/_memory_bank.py
def contains_memos(self) -> bool:
        """
        Returns True if the memory bank contains any memo.
        """
        return len(self.uid_memo_dict) > 0

# From task_centric_memory/_memory_bank.py
def add_memo(self, insight_str: str, topics: List[str], task_str: Optional[str] = None) -> None:
        """
        Adds an insight to the memory bank, given topics related to the insight, and optionally the task.
        """
        self.logger.enter_function()
        self.last_memo_id += 1
        id_str = str(self.last_memo_id)
        insight = Memo(insight=insight_str, task=task_str)
        self._map_topics_to_memo(topics, id_str, insight)
        self.logger.leave_function()

# From task_centric_memory/_memory_bank.py
def add_task_with_solution(self, task: str, solution: str, topics: List[str]) -> None:
        """
        Adds a task-solution pair to the memory bank, to be retrieved together later as a combined insight.
        This is useful when the insight is a demonstration of how to solve a given type of task.
        """
        self.logger.enter_function()
        self.last_memo_id += 1
        id_str = str(self.last_memo_id)
        # Prepend the insight to the task description for context.
        insight_str = "Example task:\n\n{}\n\nExample solution:\n\n{}".format(task, solution)
        memo = Memo(insight=insight_str, task=task)
        self._map_topics_to_memo(topics, id_str, memo)
        self.logger.leave_function()

# From task_centric_memory/_memory_bank.py
def get_relevant_memos(self, topics: List[str]) -> List[Memo]:
        """
        Returns any memos from the memory bank that appear sufficiently relevant to the input topics.
        """
        self.logger.enter_function()

        # Retrieve all topic matches, and gather them into a single list.
        matches: List[Tuple[str, str, float]] = []  # Each match is a tuple: (topic, memo_id, distance)
        for topic in topics:
            matches.extend(self.string_map.get_related_string_pairs(topic, self.n_results, self.distance_threshold))

        # Build a dict of memo-relevance pairs from the matches.
        memo_relevance_dict: Dict[str, float] = {}
        for match in matches:
            relevance = self.relevance_conversion_threshold - match[2]
            memo_id = match[1]
            if memo_id in memo_relevance_dict:
                memo_relevance_dict[memo_id] += relevance
            else:
                memo_relevance_dict[memo_id] = relevance

        # Log the details of all the retrieved memos.
        self.logger.info("\n{} POTENTIALLY RELEVANT MEMOS".format(len(memo_relevance_dict)))
        for memo_id, relevance in memo_relevance_dict.items():
            memo = self.uid_memo_dict[memo_id]
            details = ""
            if memo.task is not None:
                details += "\n  TASK: {}\n".format(memo.task)
            details += "\n  INSIGHT: {}\n\n  RELEVANCE: {:.3f}\n".format(memo.insight, relevance)
            self.logger.info(details)

        # Sort the memo-relevance pairs by relevance, in descending order.
        memo_relevance_dict = dict(sorted(memo_relevance_dict.items(), key=lambda item: item[1], reverse=True))

        # Compose the list of sufficiently relevant memos to return.
        memo_list: List[Memo] = []
        for memo_id in memo_relevance_dict:
            if memo_relevance_dict[memo_id] >= 0:
                memo_list.append(self.uid_memo_dict[memo_id])

        self.logger.leave_function()
        return memo_list

from typing import TYPE_CHECKING
from typing import Awaitable
from autogen_core.models import ChatCompletionClient
from _memory_bank import Memo
from _memory_bank import MemoryBank
from _prompter import Prompter
from utils.grader import Grader
from _memory_bank import MemoryBankConfig

# From task_centric_memory/memory_controller.py
class MemoryControllerConfig(TypedDict, total=False):
    generalize_task: bool
    revise_generalized_task: bool
    generate_topics: bool
    validate_memos: bool
    max_memos_to_retrieve: int
    max_train_trials: int
    max_test_trials: int
    MemoryBank: "MemoryBankConfig"

# From task_centric_memory/memory_controller.py
class MemoryController:
    """
    (EXPERIMENTAL, RESEARCH IN PROGRESS)

    Implements fast, memory-based learning, and manages the flow of information to and from a memory bank.

    Args:
        reset: True to empty the memory bank before starting.
        client: The model client to use internally.
        task_assignment_callback: An optional callback used to assign a task to any agent managed by the caller.
        config: An optional dict that can be used to override the following values:

            - generalize_task: Whether to rewrite tasks in more general terms.
            - revise_generalized_task: Whether to critique then rewrite the generalized task.
            - generate_topics: Whether to base retrieval directly on tasks, or on topics extracted from tasks.
            - validate_memos: Whether to apply a final validation stage to retrieved memos.
            - max_memos_to_retrieve: The maximum number of memos to return from retrieve_relevant_memos().
            - max_train_trials: The maximum number of learning iterations to attempt when training on a task.
            - max_test_trials: The total number of attempts made when testing for failure on a task.
            - MemoryBank: A config dict passed to MemoryBank.

        logger: An optional logger. If None, a default logger will be created.

    Example:

        The `task-centric-memory` extra first needs to be installed:

        .. code-block:: bash

            pip install "autogen-ext[task-centric-memory]"

        The following code snippet shows how to use this class for the most basic storage and retrieval of memories.:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_ext.experimental.task_centric_memory import MemoryController
            from autogen_ext.experimental.task_centric_memory.utils import PageLogger


            async def main() -> None:
                client = OpenAIChatCompletionClient(model="gpt-4o")
                logger = PageLogger(config={"level": "DEBUG", "path": "./pagelogs/quickstart"})  # Optional, but very useful.
                memory_controller = MemoryController(reset=True, client=client, logger=logger)

                # Add a few task-insight pairs as memories, where an insight can be any string that may help solve the task.
                await memory_controller.add_memo(task="What color do I like?", insight="Deep blue is my favorite color")
                await memory_controller.add_memo(task="What's another color I like?", insight="I really like cyan")
                await memory_controller.add_memo(task="What's my favorite food?", insight="Halibut is my favorite")

                # Retrieve memories for a new task that's related to only two of the stored memories.
                memos = await memory_controller.retrieve_relevant_memos(task="What colors do I like most?")
                print("{} memories retrieved".format(len(memos)))
                for memo in memos:
                    print("- " + memo.insight)


            asyncio.run(main())
    """

    def __init__(
        self,
        reset: bool,
        client: ChatCompletionClient,
        task_assignment_callback: Callable[[str], Awaitable[Tuple[str, str]]] | None = None,
        config: MemoryControllerConfig | None = None,
        logger: PageLogger | None = None,
    ) -> None:
        if logger is None:
            logger = PageLogger({"level": "DEBUG"})
        self.logger = logger
        self.logger.enter_function()

        # Apply default settings and any config overrides.
        self.generalize_task = True
        self.revise_generalized_task = True
        self.generate_topics = True
        self.validate_memos = True
        self.max_memos_to_retrieve = 10
        self.max_train_trials = 10
        self.max_test_trials = 3
        memory_bank_config = None
        if config is not None:
            self.generalize_task = config.get("generalize_task", self.generalize_task)
            self.revise_generalized_task = config.get("revise_generalized_task", self.revise_generalized_task)
            self.generate_topics = config.get("generate_topics", self.generate_topics)
            self.validate_memos = config.get("validate_memos", self.validate_memos)
            self.max_memos_to_retrieve = config.get("max_memos_to_retrieve", self.max_memos_to_retrieve)
            self.max_train_trials = config.get("max_train_trials", self.max_train_trials)
            self.max_test_trials = config.get("max_test_trials", self.max_test_trials)
            memory_bank_config = config.get("MemoryBank", memory_bank_config)

        self.client = client
        self.task_assignment_callback = task_assignment_callback
        self.prompter = Prompter(client, logger)
        self.memory_bank = MemoryBank(reset=reset, config=memory_bank_config, logger=logger)
        self.grader = Grader(client, logger)
        self.logger.leave_function()

    def reset_memory(self) -> None:
        """
        Empties the memory bank in RAM and on disk.
        """
        self.memory_bank.reset()

    async def train_on_task(self, task: str, expected_answer: str) -> None:
        """
        Repeatedly assigns a task to the agent, and tries to learn from failures by creating useful insights as memories.
        """
        self.logger.enter_function()
        self.logger.info("Iterate on the task, possibly discovering a useful new insight.\n")
        _, insight = await self._iterate_on_task(task, expected_answer)
        if insight is None:
            self.logger.info("No useful insight was discovered.\n")
        else:
            self.logger.info("A new insight was created:\n{}".format(insight))
            await self.add_memo(insight, task)
        self.logger.leave_function()

    async def test_on_task(self, task: str, expected_answer: str, num_trials: int = 1) -> Tuple[str, int, int]:
        """
        Assigns a task to the agent, along with any relevant memos retrieved from memory.
        """
        self.logger.enter_function()
        assert self.task_assignment_callback is not None
        response = ""
        num_successes = 0

        for trial in range(num_trials):
            self.logger.info("\n-----  TRIAL {}  -----\n".format(trial + 1))
            task_plus_insights = task

            # Try to retrieve any relevant memories from the DB.
            filtered_memos = await self.retrieve_relevant_memos(task)
            filtered_insights = [memo.insight for memo in filtered_memos]
            if len(filtered_insights) > 0:
                self.logger.info("Relevant insights were retrieved from memory.\n")
                memory_section = self._format_memory_section(filtered_insights)
                if len(memory_section) > 0:
                    task_plus_insights = task + "\n\n" + memory_section

            # Attempt to solve the task.
            self.logger.info("Try to solve the task.\n")
            response, _ = await self.task_assignment_callback(task_plus_insights)

            # Check if the response is correct.
            response_is_correct, extracted_answer = await self.grader.is_response_correct(
                task, response, expected_answer
            )
            self.logger.info("Extracted answer:  {}".format(extracted_answer))
            if response_is_correct:
                self.logger.info("Answer is CORRECT.\n")
                num_successes += 1
            else:
                self.logger.info("Answer is INCORRECT.\n")

        # Calculate the success rate as a percentage, rounded to the nearest whole number.
        self.logger.info("\nSuccess rate:  {}%\n".format(round((num_successes / num_trials) * 100)))
        self.logger.leave_function()
        return response, num_successes, num_trials

    async def add_memo(self, insight: str, task: None | str = None, index_on_both: bool = True) -> None:
        """
        Adds one insight to the memory bank, using the task (if provided) as context.
        """
        self.logger.enter_function()

        generalized_task = ""
        if task is not None:
            self.logger.info("\nGIVEN TASK:")
            self.logger.info(task)
            if self.generalize_task:
                generalized_task = await self.prompter.generalize_task(task, revise=self.revise_generalized_task)
            else:
                generalized_task = task

        self.logger.info("\nGIVEN INSIGHT:")
        self.logger.info(insight)

        # Get a list of topics from the insight and the task (if provided).
        if task is None:
            text_to_index = insight
            self.logger.info("\nTOPICS EXTRACTED FROM INSIGHT:")
        else:
            if index_on_both:
                text_to_index = generalized_task.strip() + "\n(Hint:  " + insight + ")"
                self.logger.info("\nTOPICS EXTRACTED FROM TASK AND INSIGHT COMBINED:")
            else:
                text_to_index = task
                self.logger.info("\nTOPICS EXTRACTED FROM TASK:")

        if self.generate_topics:
            topics = await self.prompter.find_index_topics(text_to_index)
        else:
            topics = [text_to_index]
        self.logger.info("\n".join(topics))
        self.logger.info("")

        # Add the insight to the memory bank.
        self.memory_bank.add_memo(insight, topics, task)
        self.logger.leave_function()

    async def add_task_solution_pair_to_memory(self, task: str, solution: str) -> None:
        """
        Adds a task-solution pair to the memory bank, to be retrieved together later as a combined insight.
        This is useful when the task-solution pair is an exemplar of solving a task related to some other task.
        """
        self.logger.enter_function()

        self.logger.info("\nEXAMPLE TASK:")
        self.logger.info(task)

        self.logger.info("\nEXAMPLE SOLUTION:")
        self.logger.info(solution)

        # Get a list of topics from the task.
        if self.generate_topics:
            topics = await self.prompter.find_index_topics(task.strip())
        else:
            topics = [task.strip()]
        self.logger.info("\nTOPICS EXTRACTED FROM TASK:")
        self.logger.info("\n".join(topics))
        self.logger.info("")

        # Add the task and solution (as a combined insight) to the memory bank.
        self.memory_bank.add_task_with_solution(task=task, solution=solution, topics=topics)
        self.logger.leave_function()

    async def retrieve_relevant_memos(self, task: str) -> List[Memo]:
        """
        Retrieves any memos from memory that seem relevant to the task.
        """
        self.logger.enter_function()

        if self.memory_bank.contains_memos():
            self.logger.info("\nCURRENT TASK:")
            self.logger.info(task)

            # Get a list of topics from the generalized task.
            if self.generalize_task:
                generalized_task = await self.prompter.generalize_task(task, revise=self.revise_generalized_task)
            else:
                generalized_task = task
            if self.generate_topics:
                task_topics = await self.prompter.find_index_topics(generalized_task)
            else:
                task_topics = [generalized_task]
            self.logger.info("\nTOPICS EXTRACTED FROM TASK:")
            self.logger.info("\n".join(task_topics))
            self.logger.info("")

            # Retrieve relevant memos from the memory bank.
            memo_list = self.memory_bank.get_relevant_memos(topics=task_topics)

            # Apply a final validation stage to keep only the memos that the LLM concludes are sufficiently relevant.
            validated_memos: List[Memo] = []
            for memo in memo_list:
                if len(validated_memos) >= self.max_memos_to_retrieve:
                    break
                if (not self.validate_memos) or await self.prompter.validate_insight(memo.insight, task):
                    validated_memos.append(memo)

            self.logger.info("\n{} VALIDATED MEMOS".format(len(validated_memos)))
            for memo in validated_memos:
                if memo.task is not None:
                    self.logger.info("\n  TASK: {}".format(memo.task))
                self.logger.info("\n  INSIGHT: {}".format(memo.insight))
        else:
            self.logger.info("\nNO SUFFICIENTLY RELEVANT MEMOS WERE FOUND IN MEMORY")
            validated_memos = []

        self.logger.leave_function()
        return validated_memos

    def _format_memory_section(self, memories: List[str]) -> str:
        """
        Formats a list of memories as a section for appending to a task description.
        """
        memory_section = ""
        if len(memories) > 0:
            memory_section = "## Important insights that may help solve tasks like this\n"
            for mem in memories:
                memory_section += "- " + mem + "\n"
        return memory_section

    async def _test_for_failure(
        self, task: str, task_plus_insights: str, expected_answer: str
    ) -> Tuple[bool, str, str]:
        """
        Attempts to solve the given task multiple times to find a failure case to learn from.
        """
        self.logger.enter_function()
        self.logger.info("\nTask description, including any insights:  {}".format(task_plus_insights))
        self.logger.info("\nExpected answer:  {}\n".format(expected_answer))

        assert self.task_assignment_callback is not None
        failure_found = False
        response, work_history = "", ""

        for trial in range(self.max_test_trials):
            self.logger.info("\n-----  TRIAL {}  -----\n".format(trial + 1))

            # Attempt to solve the task.
            self.logger.info("Try to solve the task.")
            response, work_history = await self.task_assignment_callback(task_plus_insights)

            response_is_correct, extracted_answer = await self.grader.is_response_correct(
                task, response, expected_answer
            )
            self.logger.info("Extracted answer:  {}".format(extracted_answer))
            if response_is_correct:
                self.logger.info("Answer is CORRECT.\n")
            else:
                self.logger.info("Answer is INCORRECT.\n  Stop testing, and return the details of the failure.\n")
                failure_found = True
                break

        self.logger.leave_function()
        return failure_found, response, work_history

    async def _iterate_on_task(self, task: str, expected_answer: str) -> Tuple[str, None | str]:
        """
        Repeatedly assigns a task to the agent, and tries to learn from failures by creating useful insights as memories.
        """
        self.logger.enter_function()
        self.logger.info("\nTask description:  {}".format(task))
        self.logger.info("\nExpected answer:  {}\n".format(expected_answer))

        final_response = ""
        old_memos = await self.retrieve_relevant_memos(task)
        old_insights = [memo.insight for memo in old_memos]
        new_insights: List[str] = []
        last_insight = None
        insight = None
        successful_insight = None

        # Loop until success (or timeout) while learning from failures.
        for trial in range(1, self.max_train_trials + 1):
            self.logger.info("\n-----  TRAIN TRIAL {}  -----\n".format(trial))
            task_plus_insights = task

            # Add any new insights we've accumulated so far.
            if last_insight is not None:
                memory_section = self._format_memory_section(old_insights + [last_insight])
            else:
                memory_section = self._format_memory_section(old_insights)
            if len(memory_section) > 0:
                task_plus_insights += "\n\n" + memory_section

            # Can we find a failure case to learn from?
            failure_found, response, work_history = await self._test_for_failure(
                task, task_plus_insights, expected_answer
            )
            if not failure_found:
                # No. Time to exit the loop.
                self.logger.info("\nResponse is CORRECT.\n  Stop looking for insights.\n")
                # Was this the first trial?
                if trial == 1:
                    # Yes. We should return the successful response, and no insight.
                    final_response = response
                else:
                    # No. We learned a successful insight, which should be returned.
                    successful_insight = insight
                break

            # Will we try again?
            if trial == self.max_train_trials:
                # No. We're out of training trials.
                self.logger.info("\nNo more trials will be attempted.\n")
                break

            # Try to learn from this failure.
            self.logger.info("\nResponse is INCORRECT. Try to learn from this failure.\n")
            insight = await self.prompter.learn_from_failure(
                task, memory_section, response, expected_answer, work_history
            )
            self.logger.info("\nInsight:  {}\n".format(insight))
            new_insights.append(insight)
            last_insight = insight

        # Return the answer from the last loop.
        self.logger.info("\n{}\n".format(final_response))
        self.logger.leave_function()
        return final_response, successful_insight

    async def _append_any_relevant_memories(self, task: str) -> str:
        """
        Appends any relevant memories to the task description.
        """
        self.logger.enter_function()

        filtered_memos = await self.retrieve_relevant_memos(task)
        filtered_insights = [memo.insight for memo in filtered_memos]
        if len(filtered_insights) > 0:
            self.logger.info("Relevant insights were retrieved from memory.\n")
            memory_section = self._format_memory_section(filtered_insights)
            if len(memory_section) > 0:
                task = task + "\n\n" + memory_section

        self.logger.leave_function()
        return task

    async def assign_task(self, task: str, use_memory: bool = True, should_await: bool = True) -> str:
        """
        Assigns a task to some agent through the task_assignment_callback, along with any relevant memories.
        """
        self.logger.enter_function()

        assert self.task_assignment_callback is not None

        if use_memory:
            task = await self._append_any_relevant_memories(task)

        # Attempt to solve the task.
        self.logger.info("Try to solve the task.\n")
        assert should_await
        response, _ = await self.task_assignment_callback(task)

        self.logger.leave_function()
        return response

    async def consider_memo_storage(self, text: str) -> str | None:
        """
        Tries to extract any advice from the given text and add it to memory.
        """
        self.logger.enter_function()

        advice = await self.prompter.extract_advice(text)
        self.logger.info("Advice:  {}".format(advice))
        if advice is not None:
            await self.add_memo(insight=advice)

        self.logger.leave_function()
        return advice

    async def handle_user_message(self, text: str, should_await: bool = True) -> str:
        """
        Handles a user message by extracting any advice as an insight to be stored in memory, and then calling assign_task().
        """
        self.logger.enter_function()

        # Check for advice.
        advice = await self.consider_memo_storage(text)

        # Assign the task through the task_assignment_callback, using memory only if no advice was just provided.
        response = await self.assign_task(text, use_memory=(advice is None), should_await=should_await)

        self.logger.leave_function()
        return response

# From task_centric_memory/memory_controller.py
def reset_memory(self) -> None:
        """
        Empties the memory bank in RAM and on disk.
        """
        self.memory_bank.reset()

import random
import time
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.messages import TextMessage
from autogen_core.models import LLMMessage
from autogen_core.models import UserMessage
from page_logger import PageLogger
from memory_controller import MemoryControllerConfig
from memory_controller import MemoryController
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.teams import MagenticOneGroupChat

# From utils/apprentice.py
class ApprenticeConfig(TypedDict, total=False):
    name_of_agent_or_team: str
    disable_prefix_caching: bool
    MemoryController: "MemoryControllerConfig"

# From utils/apprentice.py
class Apprentice:
    """
    A minimal wrapper combining task-centric memory with an agent or team.
    Applications may use the Apprentice class, or they may directly instantiate
    and call the Memory Controller using this class as an example.

    Args:
        client: The client to call the model.
        config: An optional dict that can be used to override the following values:

            - name_of_agent_or_team: The name of the target agent or team for assigning tasks to.
            - disable_prefix_caching: True to disable prefix caching by prepending random ints to the first message.
            - MemoryController: A config dict passed to MemoryController.

        logger: An optional logger. If None, a default logger will be created.
    """

    def __init__(
        self,
        client: ChatCompletionClient,
        config: ApprenticeConfig | None = None,
        logger: PageLogger | None = None,
    ) -> None:
        if logger is None:
            logger = PageLogger({"level": "DEBUG"})
        self.logger = logger

        # Apply default settings and any config overrides.
        self.name_of_agent_or_team = "AssistantAgent"
        self.disable_prefix_caching = False
        memory_controller_config = None
        if config is not None:
            self.name_of_agent_or_team = config.get("name_of_agent_or_team", self.name_of_agent_or_team)
            self.disable_prefix_caching = config.get("disable_prefix_caching", self.disable_prefix_caching)
            memory_controller_config = config.get("MemoryController", memory_controller_config)

        self.client = client
        if self.disable_prefix_caching:
            self.rand = random.Random()
            self.rand.seed(int(time.time() * 1000))

        # Create the MemoryController, which creates the MemoryBank.
        from ..memory_controller import MemoryController

        self.memory_controller = MemoryController(
            reset=True,
            client=self.client,
            task_assignment_callback=self.assign_task_to_agent_or_team,
            config=memory_controller_config,
            logger=self.logger,
        )

    def reset_memory(self) -> None:
        """
        Resets the memory bank.
        """
        self.memory_controller.reset_memory()

    async def handle_user_message(self, text: str, should_await: bool = True) -> str:
        """
        Handles a user message, extracting any advice and assigning a task to the agent.
        """
        self.logger.enter_function()

        # Pass the user message through to the memory controller.
        response = await self.memory_controller.handle_user_message(text, should_await)

        self.logger.leave_function()
        return response

    async def add_task_solution_pair_to_memory(self, task: str, solution: str) -> None:
        """
        Adds a task-solution pair to the memory bank, to be retrieved together later as a combined insight.
        This is useful when the insight is a demonstration of how to solve a given type of task.
        """
        self.logger.enter_function()

        # Pass the task and solution through to the memory controller.
        await self.memory_controller.add_task_solution_pair_to_memory(task, solution)

        self.logger.leave_function()

    async def assign_task(self, task: str, use_memory: bool = True, should_await: bool = True) -> str:
        """
        Assigns a task to the agent, along with any relevant insights/memories.
        """
        self.logger.enter_function()

        # Pass the task through to the memory controller.
        response = await self.memory_controller.assign_task(task, use_memory, should_await)

        self.logger.leave_function()
        return response

    async def train_on_task(self, task: str, expected_answer: str) -> None:
        """
        Repeatedly assigns a task to the completion agent, and tries to learn from failures by creating useful insights as memories.
        """
        self.logger.enter_function()

        # Pass the task through to the memory controller.
        await self.memory_controller.train_on_task(task, expected_answer)

        self.logger.leave_function()

    async def assign_task_to_agent_or_team(self, task: str) -> Tuple[str, str]:
        """
        Passes the given task to the target agent or team.
        """
        self.logger.enter_function()

        # Pass the task through.
        if self.name_of_agent_or_team == "MagenticOneGroupChat":
            response, work_history = await self._assign_task_to_magentic_one(task)
        elif self.name_of_agent_or_team == "AssistantAgent":
            response, work_history = await self._assign_task_to_assistant_agent(task)
        else:
            raise AssertionError("Invalid base agent")

        self.logger.leave_function()
        return response, work_history

    async def _assign_task_to_assistant_agent(self, task: str) -> Tuple[Any, Any]:
        """
        Passes the given task to a newly created AssistantAgent with a generic 6-step system prompt.
        """
        self.logger.enter_function()
        self.logger.info(task)

        system_message_content = """You are a helpful and thoughtful assistant.
In responding to every user message, you follow the same multi-step process given here:
1. Explain your understanding of the user message in detail, covering all the important points.
2. List as many possible responses as you can think of.
3. Carefully list and weigh the pros and cons (if any) of each possible response.
4. Critique the pros and cons above, looking for any flaws in your reasoning. But don't make up flaws that don't exist.
5. Decide on the best response, looping back to step 1 if none of the responses are satisfactory.
6. Finish by providing your final response in the particular format requested by the user."""

        if self.disable_prefix_caching:
            # Prepend a random int to disable prefix caching.
            random_str = "({})\n\n".format(self.rand.randint(0, 1000000))
            system_message_content = random_str + system_message_content

        system_message: LLMMessage
        if self.client.model_info["family"] == "o1":
            # No system message allowed, so pass it as the first user message.
            system_message = UserMessage(content=system_message_content, source="User")
        else:
            # System message allowed.
            system_message = SystemMessage(content=system_message_content)

        user_message: LLMMessage = UserMessage(content=task, source="User")
        system_message_list: List[LLMMessage] = [system_message]
        user_message_list: List[LLMMessage] = [user_message]
        input_messages: List[LLMMessage] = system_message_list + user_message_list

        assistant_agent = AssistantAgent(
            "assistant_agent",
            self.client,
            system_message=system_message_content,
        )

        # Get the agent's response to the task.
        task_result: TaskResult = await assistant_agent.run(task=TextMessage(content=task, source="User"))
        messages: Sequence[BaseAgentEvent | BaseChatMessage] = task_result.messages
        message: BaseAgentEvent | BaseChatMessage = messages[-1]
        response_str = message.to_text()

        # Log the model call
        self.logger.log_model_task(
            summary="Ask the model to complete the task", input_messages=input_messages, task_result=task_result
        )
        self.logger.info("\n-----  RESPONSE  -----\n\n{}\n".format(response_str))

        # Use the response as the work history as well.
        work_history = response_str

        self.logger.leave_function()
        return response_str, work_history

    async def _assign_task_to_magentic_one(self, task: str) -> Tuple[str, str]:
        """
        Instantiates a MagenticOneGroupChat team, and passes the given task to it.
        """
        self.logger.enter_function()
        self.logger.info(task)

        general_agent = AssistantAgent(
            "general_agent",
            self.client,
            description="A general GPT-4o AI assistant capable of performing a variety of tasks.",
        )

        from autogen_ext.agents.web_surfer import MultimodalWebSurfer

        web_surfer = MultimodalWebSurfer(
            name="web_surfer",
            model_client=self.client,
            downloads_folder="logs",
            debug_dir="logs",
            to_save_screenshots=True,
        )

        from autogen_agentchat.teams import MagenticOneGroupChat

        team = MagenticOneGroupChat(
            [general_agent, web_surfer],
            model_client=self.client,
            max_turns=20,
        )

        # Get the team's response to the task.
        task_result: TaskResult = await team.run(task=task)

        assert isinstance(task_result, TaskResult)
        messages = task_result.messages

        response_str_list: List[str] = []
        for message in messages:
            response_str_list.append(message.to_text())
        response_str = "\n".join(response_str_list)

        self.logger.info("\n-----  RESPONSE  -----\n\n{}\n".format(response_str))

        # MagenticOne's response is the chat history, which we use here as the work history.
        work_history = response_str

        self.logger.leave_function()
        return response_str, work_history

from typing import Type
from agentforge.utils.logger import Logger
from agentforge.config import Config
from agentforge.config_structs import CogConfig
from agentforge.storage.memory import Memory
from agentforge.storage.chat_history_memory import ChatHistoryMemory

# From core/memory_manager.py
class MemoryManager:
    """
    Manages memory nodes for a Cog, handling memory resolution, querying, and updating.
    Mirrors the AgentRegistry pattern for memory-specific concerns.
    """

    def __init__(self, cog_config: CogConfig, cog_name: str) -> None:
        """
        Initialize the MemoryManager with cog configuration and name.
        Orchestrates logger, config, persona, memory node, and agent-memory map setup.
        """
        self.cog_config = cog_config
        self.cog_name = cog_name
        self.logger = Logger(self.cog_name, "mem_mgr")
        self.config = Config() 
        self._resolve_persona()
        self._initialize_memory_nodes()
        self._initialize_agent_memory_maps()
        
        self.logger.debug(f"Initialized MemoryManager for cog='{self.cog_name}', persona='{self.persona}' with {len(self.memory_nodes)} memory nodes.")

    def _resolve_persona(self) -> None:
        """
        Resolve persona using precedence: Cog > Agent > default.
        """
        persona_data = self.config.resolve_persona(
            cog_config={"cog": {"persona": self.cog_config.cog.persona}}
        )
        self.persona = persona_data.get("name") if persona_data else None

    def _initialize_memory_nodes(self) -> None:
        """
        Build memory nodes from the cog configuration and add chat history node if enabled.
        """
        self.memory_nodes = self._build_memory_nodes()
        self._initialize_chat_memory()
        
    def _initialize_chat_memory(self) -> None:
        """
        Initialize the chat history memory node if enabled.
        """
        # Use the dataclass field for chat_memory_enabled (default True if None)
        chat_enabled = self.cog_config.cog.chat_memory_enabled
        self._chat_memory_enabled = chat_enabled if chat_enabled is not None else True

        if not self._chat_memory_enabled:
            return
        
        max_results = self.cog_config.cog.chat_history_max_results
        self._chat_history_max_results = max_results if max_results is not None and max_results >= 0 else 20

        # Pull chat_history_max_retrieval from cog config, defaulting to 20 if missing or negative
        max_retrieval = getattr(self.cog_config.cog, "chat_history_max_retrieval", None)
        self._chat_history_max_retrieval = max_retrieval if max_retrieval is not None and max_retrieval >= 0 else 20

        self.memory_nodes["chat_history"] = {
            "instance": ChatHistoryMemory(self.cog_name, self.persona),
            "config": None,
        }

    def _initialize_agent_memory_maps(self) -> None:
        """
        Build agent-to-memory node maps for query and update triggers.
        """
        self.query_before_map = self._map_agents_to_memory_nodes(trigger="query_before")
        self.update_after_map = self._map_agents_to_memory_nodes(trigger="update_after")

    # -----------------------------------------------------------------
    # Public Interface Methods
    # -----------------------------------------------------------------

    def query_before(self, agent_id: str, _ctx: dict, _state: dict) -> None:
        """
        Query memory nodes configured to run before the specified agent.
        Calls _query_memory_node for each relevant node.
        """
        self.logger.info(f"Querying memory nodes before agent: {agent_id}")
        queried = 0
        results_found = 0
        for mem_id in self.query_before_map.get(agent_id, []):
            if self._query_memory_node(mem_id, agent_id, _ctx, _state):
                results_found += 1
            queried += 1
        self.logger.info(f"Queried {queried} memory node(s) before agent '{agent_id}'; {results_found} returned results.")

    def update_after(self, agent_id: str, _ctx: dict, _state: dict) -> None:
        """
        Update memory nodes configured to run after the specified agent.
        Calls _update_memory_node for each relevant node.
        """
        self.logger.info(f"Updating memory nodes after agent: {agent_id}")
        updated = 0
        for mem_id in self.update_after_map.get(agent_id, []):
            self._update_memory_node(mem_id, agent_id, _ctx, _state)
            updated += 1
        self.logger.info(f"Updated {updated} memory node(s) after agent '{agent_id}'.")

    def build_mem(self) -> Dict[str, Any]:
        """
        Return a mapping of memory node IDs to their current store for agent execution context.
        Extension point: override to customize memory context building.
        Returns:
            Dict[str, Any]: Mapping of memory node IDs to their store dicts.
        """
        self.logger.debug("Building memory context for agent execution.")
        return {mid: m["instance"].store for mid, m in self.memory_nodes.items()}

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    def _build_memory_nodes(self) -> Dict[str, Any]:
        """
        Build memory nodes from the cog configuration.
        Returns:
            Dict[str, Any]: Mapping of memory node IDs to their instance/config dicts.
        """
        memories = {}
        memory_list = self.cog_config.cog.memory
        for mem_def in memory_list:
            mem_id = mem_def.id
            mem_obj = self._create_memory_node(mem_def)
            memories[mem_id] = {
                "instance": mem_obj,
                "config": mem_def,
            }
        self.logger.debug(f"Built {len(memories)} memory node(s) from configuration.")
        return memories

    def _create_memory_node(self, mem_def: Any) -> Memory:
        """
        Instantiate a single memory node from its definition.
        Extension point: override to customize memory node instantiation.
        """
        mem_class = self._get_memory_node_class(mem_def)
        collection_id = mem_def.collection_id or mem_def.id
        return mem_class(cog_name=self.cog_name, persona=self.persona, collection_id=collection_id)

    def _get_memory_node_class(self, mem_def: Any) -> Type[Memory]:
        """
        Resolve and return the memory class for a given memory definition.
        Extension point: override to customize class resolution.
        """
        return Config.resolve_class(
            mem_def.type,
            default_class=Memory,
            context=f"memory '{mem_def.id}'"
        )

    def _map_agents_to_memory_nodes(self, trigger: str) -> Dict[str, List[str]]:
        """
        Build a mapping from agent_id to memory node IDs for a given trigger ("query_before" or "update_after").
        Calls _get_agents_for_trigger and _add_agent_to_map for each node.
        Returns:
            Dict[str, List[str]]: Mapping of agent IDs to lists of memory node IDs.
        """
        agent_map: Dict[str, List[str]] = {}
        for mem_id, mem_data in self.memory_nodes.items():
            agents = self._get_agents_for_trigger(mem_data["config"], trigger)
            if not agents:
                continue
            for agent_id in agents:
                self._add_agent_to_map(agent_map, agent_id, mem_id)
        self.logger.debug(f"Built agent-to-memory map for trigger '{trigger}' with {len(agent_map)} entries.")
        return agent_map

    def _get_agents_for_trigger(self, mem_config: Any, trigger: str) -> List[str]:
        """
        Extract agent IDs for a given trigger from a memory node config.
        """
        agents = getattr(mem_config, trigger, None)
        if not agents:
            return []
        if isinstance(agents, str):
            return [agents]
        return list(agents)

    def _add_agent_to_map(self, agent_map: Dict[str, List[str]], agent_id: str, mem_id: str) -> None:
        """
        Add a memory node to the agent map for a given agent.
        """
        if agent_id not in agent_map:
            agent_map[agent_id] = []
        agent_map[agent_id].append(mem_id)

    def _query_memory_node(self, mem_id: str, agent_id: str, _ctx: dict, _state: dict) -> bool:
        """
        Query a single memory node before agent execution.
        Extension point: override to customize query logic.
        Returns True if the memory node's store is non-empty.
        """
        mem_data = self.memory_nodes[mem_id]
        cfg = mem_data["config"]
        mem_obj = mem_data["instance"]
        self.logger.debug(f"Querying memory '{mem_id}' before agent '{agent_id}'")
        mem_obj.query_memory(cfg.query_keys, _ctx, _state)
        return bool(mem_obj.store)

    def _update_memory_node(self, mem_id: str, agent_id: str, _ctx: dict, _state: dict) -> None:
        """
        Update a single memory node after agent execution.
        Extension point: override to customize update logic.
        """
        mem_data = self.memory_nodes[mem_id]
        cfg = mem_data["config"]
        mem_obj = mem_data["instance"]
        self.logger.debug(f"Updating memory '{mem_id}' after agent '{agent_id}'")
        mem_obj.update_memory(cfg.update_keys, _ctx, _state) 

    # -----------------------------------------------------------------
    # Chat History Methods
    # -----------------------------------------------------------------

    def record_chat(self, _ctx, output):
        """
        Record a chat turn in the chat history memory node, if enabled.
        """
        if not self._chat_memory_enabled:
            return
        
        chat_node = self.memory_nodes.get("chat_history").get("instance")
        chat_node.update_memory(_ctx, output)

    def load_chat(self, _ctx: dict = None, _state: dict = None):
        """
        Query the chat history node and load the most recent N messages into its store.
        N is determined by chat_history_max_results in the cog config (default 10, 0 means no limit).
        """
        if not self._chat_memory_enabled or "chat_history" not in self.memory_nodes:
            return
      
        chat_node = self.memory_nodes["chat_history"]["instance"]
        chat_node.query_memory(
            num_results=self._chat_history_max_results,
            max_retrieval=self._chat_history_max_retrieval,
            _ctx=_ctx or {},
            _state=_state or {},
        )

# From core/memory_manager.py
def query_before(self, agent_id: str, _ctx: dict, _state: dict) -> None:
        """
        Query memory nodes configured to run before the specified agent.
        Calls _query_memory_node for each relevant node.
        """
        self.logger.info(f"Querying memory nodes before agent: {agent_id}")
        queried = 0
        results_found = 0
        for mem_id in self.query_before_map.get(agent_id, []):
            if self._query_memory_node(mem_id, agent_id, _ctx, _state):
                results_found += 1
            queried += 1
        self.logger.info(f"Queried {queried} memory node(s) before agent '{agent_id}'; {results_found} returned results.")

# From core/memory_manager.py
def update_after(self, agent_id: str, _ctx: dict, _state: dict) -> None:
        """
        Update memory nodes configured to run after the specified agent.
        Calls _update_memory_node for each relevant node.
        """
        self.logger.info(f"Updating memory nodes after agent: {agent_id}")
        updated = 0
        for mem_id in self.update_after_map.get(agent_id, []):
            self._update_memory_node(mem_id, agent_id, _ctx, _state)
            updated += 1
        self.logger.info(f"Updated {updated} memory node(s) after agent '{agent_id}'.")

# From core/memory_manager.py
def build_mem(self) -> Dict[str, Any]:
        """
        Return a mapping of memory node IDs to their current store for agent execution context.
        Extension point: override to customize memory context building.
        Returns:
            Dict[str, Any]: Mapping of memory node IDs to their store dicts.
        """
        self.logger.debug("Building memory context for agent execution.")
        return {mid: m["instance"].store for mid, m in self.memory_nodes.items()}

# From core/memory_manager.py
def record_chat(self, _ctx, output):
        """
        Record a chat turn in the chat history memory node, if enabled.
        """
        if not self._chat_memory_enabled:
            return
        
        chat_node = self.memory_nodes.get("chat_history").get("instance")
        chat_node.update_memory(_ctx, output)

# From core/memory_manager.py
def load_chat(self, _ctx: dict = None, _state: dict = None):
        """
        Query the chat history node and load the most recent N messages into its store.
        N is determined by chat_history_max_results in the cog config (default 10, 0 means no limit).
        """
        if not self._chat_memory_enabled or "chat_history" not in self.memory_nodes:
            return
      
        chat_node = self.memory_nodes["chat_history"]["instance"]
        chat_node.query_memory(
            num_results=self._chat_history_max_results,
            max_retrieval=self._chat_history_max_retrieval,
            _ctx=_ctx or {},
            _state=_state or {},
        )

from agentforge.utils.prompt_processor import PromptProcessor

# From storage/chat_history_memory.py
class ChatHistoryMemory(Memory):

    ALLOW_META = {"iso_timestamp", "id", }

    def __init__(self, cog_name, persona=None, collection_id="chat_history"):
        super().__init__(cog_name, persona, collection_id, logger_name="ChatHistoryMemory")
        self.prompt_processor = PromptProcessor()

    def update_memory(self, ctx, output):
        turn_id = str(uuid.uuid4())

        def make_meta(role):
            return {
                "role": role,
                "turn_id": turn_id,
            }

        docs  = [self.prompt_processor.value_to_markdown(ctx), self.prompt_processor.value_to_markdown(output)]
        metas = [make_meta("user"), make_meta("assistant")]

        # Let Chroma auto-increment the integer IDs (or keep your own counter),
        # but *do not* copy the partner's text into metadata.
        self.storage.save_to_storage(self.collection_name,
                                    data=docs,
                                    metadata=metas)

    # ------------------------
    # Helpers
    # ------------------------
    def _sort_records(self, records):
        """Return records sorted oldest→newest using iso_timestamp then id."""
        def sort_key(rec):
            meta = rec["meta"]
            ts = meta.get("iso_timestamp")
            return ts if ts is not None else meta.get("id", 0)
        return sorted(records, key=sort_key)

    def _format_records(self, records):
        """Convert raw records into the structure expected by prompts."""
        formatted = []
        for rec in records:
            role = rec["meta"]["role"]
            formatted.append({
                role: [
                    rec["content"],
                    f"timestamp: {rec['meta'].get('iso_timestamp', '')}\n"
                ]
            })
        return formatted

    def _get_recency_records(self, num_results):
        raw = self.storage.get_last_x_entries(
            self.collection_name,
            num_results,
            include=["documents", "metadatas"],
        )
        records = [
            {"content": d, "meta": m}
            for d, m in zip(raw["documents"], raw["metadatas"])
        ]
        return self._sort_records(records)

    def _get_semantic_records(self, query_texts, max_retrieval, recency_records):
        # Calculate a filter to avoid fetching items already in the recency slice
        min_id = None
        if recency_records:
            try:
                min_id = min(r["meta"]["id"] for r in recency_records)
            except Exception:
                pass

        filter_condition = {"id": {"$lt": min_id}} if min_id is not None else None

        # Ask for more than we need to account for post-deduplication
        overshoot = max_retrieval + len(recency_records) * 2

        raw_semantic = self.storage.query_storage(
            collection_name=self.collection_name,
            query=query_texts,
            filter_condition=filter_condition,
            num_results=overshoot,
        )

        if not raw_semantic or not raw_semantic.get("documents"):
            return []

        seen_ids = {r["meta"]["id"] for r in recency_records}
        seen_turns = {r["meta"].get("turn_id") for r in recency_records}

        semantic_records = []
        for d, m in zip(raw_semantic["documents"], raw_semantic["metadatas"]):
            if m["id"] in seen_ids or m.get("turn_id") in seen_turns:
                continue
            semantic_records.append({"content": d, "meta": m})

        # Sort and limit to the requested max number
        semantic_records = self._sort_records(semantic_records)[:max_retrieval]
        return semantic_records

    # ------------------------
    # Public
    # ------------------------
    def query_memory(self, num_results=20, max_retrieval=20, query_keys=None, _ctx=None, _state=None, **kwargs):
        """Populate self.store with 'history' (recency) and optionally 'relevant' (semantic)."""
        # Phase 1 – recency slice
        recency_records = self._get_recency_records(num_results)
        self.store["history"] = self._format_records(recency_records)

        # Phase 2 – semantic slice
        if max_retrieval > 0:
            query_texts = self._build_queries_from_keys_and_context(query_keys, _ctx, _state)
            # Fallback: if no query text provided, use the most recent user message
            if not query_texts:
                # find last record authored by user (search from end)
                for rec in reversed(recency_records):
                    if rec["meta"].get("role") == "user":
                        query_texts = rec["content"]
                        break
                # If still empty, take last assistant message as last resort
                if not query_texts and recency_records:
                    query_texts = recency_records[-1]["content"]
            if query_texts:
                semantic_records = self._get_semantic_records(query_texts, max_retrieval, recency_records)
                if not semantic_records:
                    semantic_records = [{"content": "No relevant records found in memory", "meta": {"role": "memory_system"}}]

                self.store["relevant"] = self._format_records(semantic_records)

# From storage/chat_history_memory.py
def update_memory(self, ctx, output):
        turn_id = str(uuid.uuid4())

        def make_meta(role):
            return {
                "role": role,
                "turn_id": turn_id,
            }

        docs  = [self.prompt_processor.value_to_markdown(ctx), self.prompt_processor.value_to_markdown(output)]
        metas = [make_meta("user"), make_meta("assistant")]

        # Let Chroma auto-increment the integer IDs (or keep your own counter),
        # but *do not* copy the partner's text into metadata.
        self.storage.save_to_storage(self.collection_name,
                                    data=docs,
                                    metadata=metas)

# From storage/chat_history_memory.py
def query_memory(self, num_results=20, max_retrieval=20, query_keys=None, _ctx=None, _state=None, **kwargs):
        """Populate self.store with 'history' (recency) and optionally 'relevant' (semantic)."""
        # Phase 1 – recency slice
        recency_records = self._get_recency_records(num_results)
        self.store["history"] = self._format_records(recency_records)

        # Phase 2 – semantic slice
        if max_retrieval > 0:
            query_texts = self._build_queries_from_keys_and_context(query_keys, _ctx, _state)
            # Fallback: if no query text provided, use the most recent user message
            if not query_texts:
                # find last record authored by user (search from end)
                for rec in reversed(recency_records):
                    if rec["meta"].get("role") == "user":
                        query_texts = rec["content"]
                        break
                # If still empty, take last assistant message as last resort
                if not query_texts and recency_records:
                    query_texts = recency_records[-1]["content"]
            if query_texts:
                semantic_records = self._get_semantic_records(query_texts, max_retrieval, recency_records)
                if not semantic_records:
                    semantic_records = [{"content": "No relevant records found in memory", "meta": {"role": "memory_system"}}]

                self.store["relevant"] = self._format_records(semantic_records)

# From storage/chat_history_memory.py
def make_meta(role):
            return {
                "role": role,
                "turn_id": turn_id,
            }

# From storage/chat_history_memory.py
def sort_key(rec):
            meta = rec["meta"]
            ts = meta.get("iso_timestamp")
            return ts if ts is not None else meta.get("id", 0)

import re
from agentforge.utils.parsing_processor import ParsingProcessor
from agentforge.agent import Agent

# From storage/scratchpad.py
class ScratchPad(Memory):
    """
    ScratchPad memory storage that inherits from the base Memory class.
    Provides functionality for storing, retrieving, and updating user scratchpads.
    
    This class maintains two collections:
    1. The main scratchpad which contains the current summarized knowledge
    2. A log collection that stores individual entries before they're consolidated
    """

    def __init__(self, cog_name: str, persona: Optional[str] = None, collection_id: Optional[str] = None):
        """
        Initialize the ScratchPad memory.
        
        Args:
            cog_name (str): Name of the cog to which this memory belongs.
            persona (Optional[str]): Optional persona name for further partitioning.
            collection_id (Optional[str]): Identifier for the main scratchpad collection.
        """
        super().__init__(cog_name, persona, collection_id)
        self.logger = Logger('Memory')
        self.parser = ParsingProcessor()
        self.prompt_processor = PromptProcessor()
        
        # Define the log collection name based on the main collection
        self.log_collection_name = f"scratchpad_log_{self.collection_name}"
        self.log_collection_name = self.parser.format_string(self.log_collection_name)

    def query_memory(self, query_keys: Optional[List[str]], _ctx: dict, _state: dict, num_results: int = 5) -> dict[str, Any]:
        """
        Query memory storage for relevant entries based on provided context and state.

        Args:
            query_keys (Optional[List[str]]): Keys to construct the query from context/state.
            _ctx (dict): External context data.
            _state (dict): Internal state data.
            num_results (int): Number of results to retrieve.
        """
        # For scratchpads, we don't use semantic search but instead just retrieve the content
        result = self.storage.load_collection(collection_name=self.collection_name)
        self.logger.debug(f"Retrieved scratchpad: {result}")
        
        if result and result.get('documents') and len(result['documents']) > 0:
            self.store.update({"readable": result.get('documents')})
            self.logger.debug(f"Query returned {len(result.get('ids', []))} results.")
            return
        
        # Return default message if no scratchpad exists
        default_msg = "No information available yet. This scratchpad will be updated as we learn more about the user."
        self.store.update({"readable": default_msg})


    def update_memory(self, update_keys: Optional[List[str]], _ctx: dict, _state: dict,
                      ids: Optional[Union[str, list[str]]] = None,
                      metadata: Optional[list[dict]] = None) -> None:
        """
        Update memory with new data using update_keys to extract from context/state.

        Args:
            update_keys (Optional[List[str]]): Keys to extract for update, or None to use all data.
            _ctx (dict): External context data.
            _state (dict): Internal state data.
            ids (Union[str, list[str]], optional): The IDs for the documents.
            metadata (list[dict], optional): Custom metadata for the documents (overrides generated metadata).
        """

        content = self.prompt_processor.value_to_markdown(val=_ctx)
        state = self.prompt_processor.value_to_markdown(val=_state)
        
        if not content:
            self.logger.warning("No content provided for scratchpad update")
            return
            
        # Save to the log collection
        self._save_scratchpad_log(content)
        self._save_scratchpad_log(state)
        
        # Check if it's time to consolidate the log
        self.check_scratchpad()

    def _save_scratchpad_log(self, content: str) -> None:
        """
        Save an entry to the scratchpad log.
        
        Args:
            content (str): The content to save in the log.
        """
        collection_size = self.storage.count_collection(self.log_collection_name)
        memory_id = [str(collection_size + 1)]
        
        self.logger.debug(
            f"Saving to Scratchpad Log: {self.log_collection_name}\nContent: {content}\nID: {memory_id}", 
        )
        
        self.storage.save_to_storage(
            collection_name=self.log_collection_name,
            data=[content],
            ids=memory_id,
            metadata=[{}]  # No special metadata needed for log entries
        )

    def _save_main_scratchpad(self, content: str) -> None:
        """
        Save or update the main scratchpad.
        
        Args:
            content (str): The content to save in the scratchpad.
        """
        self.logger.debug(
            f"Updating main scratchpad: {self.collection_name}\nContent: {content[:100]}...", 
        )
        
        self.storage.save_to_storage(
            collection_name=self.collection_name,
            data=[content],
            ids=["1"],  # Always use ID "1" to ensure we replace the existing entry
            metadata=[{}]  # No special metadata needed for main scratchpad
        )

    def _get_scratchpad_log(self) -> list:
        """
        Retrieve the scratchpad log entries.
        
        Returns:
            list: The scratchpad log entries as a list or an empty list if not found.
        """
        result = self.storage.load_collection(collection_name=self.log_collection_name)
        self.logger.debug(f"Scratchpad Log: {result}")
        
        if result and result.get('documents'):
            return result['documents']
        return []

    def check_scratchpad(self) -> Optional[str]:
        """
        Check if it's time to update the scratchpad based on the log entries.
        If there are enough log entries, consolidate them into the main scratchpad.
        
        Returns:
            Optional[str]: Updated scratchpad content if updated, None otherwise.
        """
        scratchpad_log = self._get_scratchpad_log()
        log_count = len(scratchpad_log)
        
        self.logger.debug(f"Checking scratchpad log. Number of entries: {log_count}")
        
        # If we have enough log entries, consolidate them
        if log_count >= 20:
            self.logger.debug(f"Scratchpad log count >= 20, updating scratchpad")
            
            # Create an agent to summarize the log
            scratchpad_agent = Agent(agent_name="scratchpad_agent")
            
            # Get the current scratchpad content
            current_scratchpad = self.store.get('readable', '')
            
            # Join all log entries
            scratchpad_log_content = "\n".join(scratchpad_log)
            
            # Run the agent to generate a new scratchpad
            agent_vars = {
                "scratchpad_log": scratchpad_log_content,
                "scratchpad": current_scratchpad
            }
            scratchpad_result = scratchpad_agent.run(**agent_vars)
            
            # Extract the updated scratchpad from the agent's response
            updated_scratchpad = self._extract_updated_scratchpad(scratchpad_result)
            
            # Save the updated scratchpad
            self._save_main_scratchpad(updated_scratchpad)
            
            # Clear the log after processing
            self.storage.delete_collection(self.log_collection_name)
            self.logger.debug(f"Cleared scratchpad log")
            
            return updated_scratchpad
            
        return None

    def _extract_updated_scratchpad(self, scratchpad_result: str) -> str:
        """
        Extract the updated scratchpad content from the ScratchpadAgent's output.
        
        Parameters:
            scratchpad_result (str): The full output from the ScratchpadAgent.
            
        Returns:
            str: The extracted updated scratchpad content.
        """
        pattern = r'<updated_scratchpad>(.*?)</updated_scratchpad>'
        match = re.search(pattern, scratchpad_result, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            self.logger.warning("No updated scratchpad content found in the result.")
            return "No updated scratchpad content could be extracted."

    def delete(self, ids: Union[str, list[str]] = None) -> None:
        """
        Delete the scratchpad and its log.
        
        Args:
            ids (Union[str, list[str]], optional): Not used for scratchpads.
        """
        # Delete both the main scratchpad and the log
        self.storage.delete_collection(self.collection_name)
        self.storage.delete_collection(self.log_collection_name)
        self.logger.debug(f"Deleted scratchpad and log collections")

# From storage/scratchpad.py
def check_scratchpad(self) -> Optional[str]:
        """
        Check if it's time to update the scratchpad based on the log entries.
        If there are enough log entries, consolidate them into the main scratchpad.
        
        Returns:
            Optional[str]: Updated scratchpad content if updated, None otherwise.
        """
        scratchpad_log = self._get_scratchpad_log()
        log_count = len(scratchpad_log)
        
        self.logger.debug(f"Checking scratchpad log. Number of entries: {log_count}")
        
        # If we have enough log entries, consolidate them
        if log_count >= 20:
            self.logger.debug(f"Scratchpad log count >= 20, updating scratchpad")
            
            # Create an agent to summarize the log
            scratchpad_agent = Agent(agent_name="scratchpad_agent")
            
            # Get the current scratchpad content
            current_scratchpad = self.store.get('readable', '')
            
            # Join all log entries
            scratchpad_log_content = "\n".join(scratchpad_log)
            
            # Run the agent to generate a new scratchpad
            agent_vars = {
                "scratchpad_log": scratchpad_log_content,
                "scratchpad": current_scratchpad
            }
            scratchpad_result = scratchpad_agent.run(**agent_vars)
            
            # Extract the updated scratchpad from the agent's response
            updated_scratchpad = self._extract_updated_scratchpad(scratchpad_result)
            
            # Save the updated scratchpad
            self._save_main_scratchpad(updated_scratchpad)
            
            # Clear the log after processing
            self.storage.delete_collection(self.log_collection_name)
            self.logger.debug(f"Cleared scratchpad log")
            
            return updated_scratchpad
            
        return None

# From storage/scratchpad.py
def delete(self, ids: Union[str, list[str]] = None) -> None:
        """
        Delete the scratchpad and its log.
        
        Args:
            ids (Union[str, list[str]], optional): Not used for scratchpads.
        """
        # Delete both the main scratchpad and the log
        self.storage.delete_collection(self.collection_name)
        self.storage.delete_collection(self.log_collection_name)
        self.logger.debug(f"Deleted scratchpad and log collections")


# From storage/persona_memory.py
class PersonaMemory(Memory):
    """
    Specialized memory for managing persona-related facts and narratives.
    Integrates with retrieval, narrative, and update agents for dynamic persona management.
    """
    def __init__(self, cog_name: str, persona: Optional[str] = None, collection_id: Optional[str] = None):
        """
        Initialize PersonaMemory with specialized agents for persona management.
        Args:
            cog_name (str): Name of the cog to which this memory belongs.
            persona (Optional[str]): Optional persona name for further partitioning.
            collection_id (Optional[str]): Identifier for the collection.
        """
        super().__init__(cog_name, persona, collection_id, logger_name="PersonaMemory")
        self.prompt_processor = PromptProcessor()
        self._initialize_agents()
        self.narrative: Optional[str] = None

    def _initialize_agents(self) -> None:
        """
        Initialize the specialized agents used by PersonaMemory.
        """
        try:
            self.retrieval_agent = Agent(agent_name="persona_retrieval_agent")
            self.narrative_agent = Agent(agent_name="persona_narrative_agent")
            self.update_agent = Agent(agent_name="persona_update_agent")
            self.logger.debug("Initialized persona memory agents")
        except Exception as e:
            self.logger.error(f"Failed to initialize persona agents: {e}")
            raise

    def _build_collection_name(self) -> None:
        """
        Build a collection name specific to persona memory.
        Sets self.collection_name to 'persona_facts_{persona|cog_name}'.
        """
        base_name = self.collection_id or "persona_facts"
        storage_suffix = self._resolve_storage_id()
        self.collection_name = f"{base_name}_{storage_suffix}"

    # -----------------------------------------------------------------
    # Context Preparation and Fact Retrieval
    # -----------------------------------------------------------------
    def _load_context(self, keys: Optional[List[str]], _ctx: dict, _state: dict, num_results: int) -> Tuple[str, List[str]]:
        static_persona = self._get_static_persona_markdown()
        query_keys = self._extract_query_keys(keys, _ctx, _state)
        initial_facts = self._retrieve_initial_facts(query_keys, num_results)
        return static_persona, initial_facts

    def _extract_query_keys(self, keys: Optional[List[str]], _ctx: dict, _state: dict) -> List[str]:
        """
        Extract query keys from context and state.
        Args:
            keys: Keys to extract.
            _ctx: External context data.
            _state: Internal state data.
        Returns:
            List of extracted query keys.
        """
        if not keys:
            return []
        try:
            extracted = self._extract_keys(keys, _ctx, _state)
            if isinstance(extracted, dict):
                return list(extracted.values())
            if isinstance(extracted, list):
                return extracted
            return [str(extracted)]
        except Exception as e:
            self.logger.warning(f"Failed to extract query keys: {e}")
            return []

    def _retrieve_initial_facts(self, query_keys: List[str], num_results: int) -> List[str]:
        """
        Perform initial storage lookup for persona facts.
        Args:
            query_keys: List of query keys.
            num_results: Number of results to retrieve.
        Returns:
            List of retrieved facts.
        """
        if not query_keys:
            return []
        results = self.storage.query_storage(
            collection_name=self.collection_name,
            query=query_keys,
            num_results=num_results
        )
        return results.get('documents', [])

    # -----------------------------------------------------------------
    # Semantic Retrieval
    # -----------------------------------------------------------------
    def _retrieve_semantic_facts(self, _ctx: dict, _state: dict, static_persona: str, initial_facts: List[str], num_results: int) -> List[str]:
        queries = self._generate_semantic_queries(_ctx, _state, static_persona, initial_facts, num_results)
        semantic_facts = self._perform_semantic_search(queries, num_results)
        return self._deduplicate_facts(initial_facts + semantic_facts)

    def _generate_semantic_queries(self, _ctx: dict, _state: dict, static_persona: str, retrieved_facts: List[str], num_results: int) -> List[str]:
        retrieval_response = self.retrieval_agent.run(
            _ctx=_ctx,
            _state=_state,
            persona_static=static_persona,
            retrieved_facts=retrieved_facts,
            num_results=num_results
        )
        if not isinstance(retrieval_response, dict) or 'queries' not in retrieval_response:
            self.logger.warning("Retrieval agent returned invalid response format")
            return []
        queries = retrieval_response.get('queries', [])
        if not queries:
            self.logger.warning("No queries generated by retrieval agent")
        return queries

    def _perform_semantic_search(self, queries: List[str], num_results: int) -> List[str]:
        """
        Perform semantic search using generated queries.
        Args:
            queries: List of queries.
            num_results: Number of results to retrieve.
        Returns:
            List of found facts.
        """
        if not queries:
            return []
        results = self.storage.query_storage(
            collection_name=self.collection_name,
            query=queries,
            num_results=num_results
        )
        return results.get('documents', [])

    def _deduplicate_facts(self, facts: List[str]) -> List[str]:
        """
        Deduplicate facts while preserving order.
        Args:
            facts: List of facts.
        Returns:
            List of unique facts.
        """
        unique_facts = []
        seen = set()
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)
        return unique_facts

    # -----------------------------------------------------------------
    # Narrative Generation
    # -----------------------------------------------------------------
    def _generate_narrative(self, _ctx: dict, _state: dict, static_persona: str, facts: List[str]) -> str:
        narrative_response = self.narrative_agent.run(
            _ctx=_ctx,
            _state=_state,
            persona_static=static_persona,
            retrieved_facts=facts
        )
        if narrative_response and 'narrative' in narrative_response:
            return narrative_response['narrative']
        return self._generate_static_only_narrative(static_persona)

    def _generate_static_only_narrative(self, static_persona: str) -> str:
        """
        Generate a narrative using only static persona when no dynamic facts are available.
        Updates self.store with the static narrative instead of returning data.
        
        Args:
            static_persona: The static persona information.
        Returns:
            The static narrative.
        """
        return f"Based on the static persona information: {static_persona}"

    def _get_static_persona_markdown(self) -> str:
        """
        Get the static persona information formatted as markdown.
        Returns:
            Markdown formatted static persona data.
        """
        from agentforge.config import Config
        config = Config()
        cog_config = None
        if hasattr(self, 'cog_name') and self.cog_name:
            try:
                cog_data = config.load_cog_data(self.cog_name)
                cog_config = cog_data.get('cog', {}) if cog_data else None
            except Exception:
                pass
        persona_data = config.resolve_persona(cog_config=cog_config, agent_config=None)
        if not persona_data or 'static' not in persona_data:
            return "No static persona information available."
        static_content = persona_data.get('static', {})
        persona_settings = config.settings.system.persona
        persona_md = self.prompt_processor.build_persona_markdown(static_content, persona_settings)
        return persona_md or "No static persona information available."

    # -----------------------------------------------------------------
    # Memory Updating
    # -----------------------------------------------------------------
    def _determine_update_action(self, _ctx: dict, _state: dict, static_persona: str, facts: List[str]) -> Tuple[str, List[dict]]:
        update_response = self.update_agent.run(
            _ctx=_ctx,
            _state=_state,
            persona_static=static_persona,
            retrieved_facts=facts
        )
        if update_response and 'action' in update_response:
            return update_response['action'], update_response.get('new_facts', [])
        return 'none', []

    def _apply_update_action(self, action: str, new_facts: List[dict]) -> None:
        for fact_data in new_facts:
            if not isinstance(fact_data, dict) or 'fact' not in fact_data:
                self.logger.warning("Invalid fact format in update response")
                continue
            new_fact = fact_data.get('fact')
            supersedes = fact_data.get('supersedes', [])
            if not new_fact:
                self.logger.warning("Empty fact provided")
                continue
            if action == 'add':
                if self._is_duplicate_fact(new_fact):
                    self.logger.debug(f"Skipping duplicate fact: {new_fact}")
                    continue
                self.logger.info(f"Adding new persona fact: {new_fact}")
                fact_metadata_list = [{
                    'type': 'persona_fact',
                    'source': 'update_agent',
                    'superseded': False
                }]
                self.storage.save_to_storage(
                    collection_name=self.collection_name,
                    data=[new_fact],
                    metadata=fact_metadata_list
                )
            elif action == 'update':
                self.logger.info(f"Updating persona with new fact: {new_fact}")
                fact_metadata_list = [{
                    'type': 'persona_fact',
                    'source': 'update_agent',
                    'superseded': False,
                    'supersedes': ','.join(supersedes) if supersedes else ''
                }]
                self.storage.save_to_storage(
                    collection_name=self.collection_name,
                    data=[new_fact],
                    metadata=fact_metadata_list
                )
                self.logger.debug(f"Marked facts as superseded: {supersedes}")

    def _is_duplicate_fact(self, new_fact: str) -> bool:
        """
        Check if an exact duplicate of the new fact already exists in storage.
        Args:
            new_fact (str): The fact to check for duplicates.
        Returns:
            bool: True if an exact duplicate exists, False otherwise.
        """
        try:
            results = self.storage.query_storage(
                collection_name=self.collection_name,
                query=new_fact,
                num_results=10
            )
            if not results or not results.get('documents'):
                return False
            for document in results['documents']:
                if document.strip() == new_fact.strip():
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking for duplicate facts: {e}")
            return False

    # -----------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------
    
    def query_memory(self, query_keys: Optional[List[str]], _ctx: dict, _state: dict, num_results: int = 5) -> None:
        """
        Query persona memory using retrieval and narrative agents.
        Updates self.store with narrative and retrieved facts instead of returning data.
        
        Args:
            query_keys: Keys to extract from context/state, or None to use all data.
            _ctx: External context data.
            _state: Internal state data.
            num_results: Number of results to retrieve per query.
        """
        try:
            static_persona, initial_facts = self._load_context(query_keys, _ctx, _state, num_results)
            semantic_facts = self._retrieve_semantic_facts(_ctx, _state, static_persona, initial_facts, num_results)
            narrative = self._generate_narrative(_ctx, _state, static_persona, semantic_facts)
            self.narrative = narrative
            self.store = {
                '_narrative': narrative,
                '_static': static_persona,
                '_retrieved_facts': semantic_facts,
                'raw_facts': semantic_facts
            }
            self.logger.debug(f"Successfully generated narrative and stored {len(semantic_facts)} facts")
        except Exception as e:
            self.logger.error(f"Error in query_memory: {e}")
            raise Exception(f"Error in query_memory: {e}")

    def update_memory(self, update_keys: Optional[List[str]], _ctx: dict, _state: dict,
                     ids: Optional[Union[str, List[str]]] = None,
                     metadata: Optional[List[dict]] = None,
                     num_results: int = 5) -> None:
        """
        Update persona memory using retrieval and update agents.
        Args:
            update_keys: Keys to extract from context/state, or None to use all data.
            _ctx: External context data.
            _state: Internal state data.
            ids: Ignored; IDs are auto-generated for persona facts.
            metadata: Ignored; custom metadata is generated for persona facts.
            num_results: Number of results to retrieve per query.
        """
        try:
            static_persona, initial_facts = self._load_context(update_keys, _ctx, _state, num_results)
            semantic_facts = self._retrieve_semantic_facts(_ctx, _state, static_persona, initial_facts, num_results)
            action, new_facts = self._determine_update_action(_ctx, _state, static_persona, semantic_facts)
            if action != 'none':
                self._apply_update_action(action, new_facts)
        except Exception as e:
            self.logger.error(f"Error in update_memory: {e}")
            raise Exception(f"Error in update_memory: {e}")

from chroma_storage import ChromaStorage

# From storage/memory.py
def wipe_memory(self) -> None:
        """
        Wipe all memory, removing all collections and their data.
        Use with caution: this will permanently delete all data within the storage.
        This is a template method that delegates to extensible steps for customization.
        """
        self._prepare_wipe()
        try:
            self._execute_wipe()
            self._post_wipe()
        except Exception as e:
            self._handle_wipe_error(e)

# From storage/memory.py
def format_memory_results(raw_results: dict) -> str:
        """
        Format raw query results into a human-readable string.
        Subclasses can override this method to customize formatting.

        Args:
            raw_results (dict): The raw result from storage query.
        Returns:
            str: Human-readable formatted string of memory results.
        """
        ids = raw_results.get("ids", [])
        documents = raw_results.get("documents", [])
        metadatas = raw_results.get("metadatas", [])

        records = [
            {"id": id_, "content": d, "meta": m}
            for id_, d, m in list(zip(ids, documents, metadatas))
        ]

        def sort_key(rec):
            meta = rec["meta"]
            # Prefer iso_timestamp if present, else fallback to id
            # If both missing, fallback to 0
            ts = meta.get("iso_timestamp")
            if ts is not None:
                return ts
            return meta.get("id", 0)

        records = sorted(records, key=sort_key)

        history = []

        for rec in records:
            current_record = {}
            rec_id = rec["id"]
            
            current_record[rec_id] = [
                f"{rec['content']}\n",
                rec["meta"]
            ]

            history.append(current_record)

        return records

from __future__ import annotations
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from pinecone import Index
from reworkd_platform.settings import settings
from reworkd_platform.timer import timed_function
from reworkd_platform.web.api.memory.memory import AgentMemory

# From pinecone/pinecone.py
class Row(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any] = {}

# From pinecone/pinecone.py
class QueryResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = {}

# From pinecone/pinecone.py
class PineconeMemory(AgentMemory):
    """
    Wrapper around pinecone
    """

    def __init__(self, index_name: str, namespace: str = ""):
        self.index = Index(settings.pinecone_index_name)
        self.namespace = namespace or index_name

    @timed_function(level="DEBUG")
    def __enter__(self) -> AgentMemory:
        self.embeddings: Embeddings = OpenAIEmbeddings(
            client=None,  # Meta private value but mypy will complain its missing
            openai_api_key=settings.openai_api_key,
        )

        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @timed_function(level="DEBUG")
    def reset_class(self) -> None:
        self.index.delete(delete_all=True, namespace=self.namespace)

    @timed_function(level="DEBUG")
    def add_tasks(self, tasks: List[str]) -> List[str]:
        if len(tasks) == 0:
            return []

        embeds = self.embeddings.embed_documents(tasks)

        if len(tasks) != len(embeds):
            raise ValueError("Embeddings and tasks are not the same length")

        rows = [
            Row(values=vector, metadata={"text": tasks[i]}, id=str(uuid.uuid4()))
            for i, vector in enumerate(embeds)
        ]

        self.index.upsert(
            vectors=[row.dict() for row in rows], namespace=self.namespace
        )

        return [row.id for row in rows]

    @timed_function(level="DEBUG")
    def get_similar_tasks(
        self, text: str, score_threshold: float = 0.95
    ) -> List[QueryResult]:
        # Get similar tasks
        vector = self.embeddings.embed_query(text)
        results = self.index.query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            include_values=True,
            namespace=self.namespace,
        )

        return [
            QueryResult(id=row.id, score=row.score, metadata=row.metadata)
            for row in getattr(results, "matches", [])
            if row.score > score_threshold
        ]

    @staticmethod
    def should_use() -> bool:
        return False

# From pinecone/pinecone.py
def reset_class(self) -> None:
        self.index.delete(delete_all=True, namespace=self.namespace)

# From pinecone/pinecone.py
def add_tasks(self, tasks: List[str]) -> List[str]:
        if len(tasks) == 0:
            return []

        embeds = self.embeddings.embed_documents(tasks)

        if len(tasks) != len(embeds):
            raise ValueError("Embeddings and tasks are not the same length")

        rows = [
            Row(values=vector, metadata={"text": tasks[i]}, id=str(uuid.uuid4()))
            for i, vector in enumerate(embeds)
        ]

        self.index.upsert(
            vectors=[row.dict() for row in rows], namespace=self.namespace
        )

        return [row.id for row in rows]

# From pinecone/pinecone.py
def get_similar_tasks(
        self, text: str, score_threshold: float = 0.95
    ) -> List[QueryResult]:
        # Get similar tasks
        vector = self.embeddings.embed_query(text)
        results = self.index.query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            include_values=True,
            namespace=self.namespace,
        )

        return [
            QueryResult(id=row.id, score=row.score, metadata=row.metadata)
            for row in getattr(results, "matches", [])
            if row.score > score_threshold
        ]

# From pinecone/pinecone.py
def should_use() -> bool:
        return False

import pytest
from reworkd_platform.web.api.memory.memory_with_fallback import MemoryWithFallback

# From memory/memory_with_fallback_test.py
def test_memory_primary(mocker, method_name: str, args) -> None:
    primary = mocker.Mock()
    secondary = mocker.Mock()
    memory_with_fallback = MemoryWithFallback(primary, secondary)

    # Use getattr() to call the method on the object with args
    getattr(memory_with_fallback, method_name)(*args)
    getattr(primary, method_name).assert_called_once_with(*args)
    getattr(secondary, method_name).assert_not_called()

# From memory/memory_with_fallback_test.py
def test_memory_fallback(mocker, method_name: str, args) -> None:
    primary = mocker.Mock()
    secondary = mocker.Mock()
    memory_with_fallback = MemoryWithFallback(primary, secondary)

    getattr(primary, method_name).side_effect = Exception("Primary Failed")

    # Call the method again, this time it should fall back to secondary
    getattr(memory_with_fallback, method_name)(*args)
    getattr(primary, method_name).assert_called_once_with(*args)
    getattr(secondary, method_name).assert_called_once_with(*args)

from loguru import logger

# From memory/memory_with_fallback.py
class MemoryWithFallback(AgentMemory):
    """
    Wrap a primary AgentMemory provider and use a fallback in the case that it fails
    We do this because we've had issues with Weaviate crashing and causing memory to randomly fail
    """

    def __init__(self, primary: AgentMemory, secondary: AgentMemory):
        self.primary = primary
        self.secondary = secondary

    def __enter__(self) -> AgentMemory:
        try:
            return self.primary.__enter__()
        except Exception as e:
            logger.exception(e)
            return self.secondary.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            self.primary.__exit__(exc_type, exc_value, traceback)
        except Exception as e:
            logger.exception(e)
            self.secondary.__exit__(exc_type, exc_value, traceback)

    def add_tasks(self, tasks: List[str]) -> List[str]:
        try:
            return self.primary.add_tasks(tasks)
        except Exception as e:
            logger.exception(e)
            return self.secondary.add_tasks(tasks)

    def get_similar_tasks(self, query: str, score_threshold: float = 0) -> List[str]:
        try:
            return self.primary.get_similar_tasks(query)
        except Exception as e:
            logger.exception(e)
            return self.secondary.get_similar_tasks(query)

    def reset_class(self) -> None:
        try:
            self.primary.reset_class()
        except Exception as e:
            logger.exception(e)
            self.secondary.reset_class()


# From memory/memory.py
class AgentMemory(ABC):
    """
    Base class for AgentMemory
    Expose __enter__ and __exit__ to ensure connections get closed within requests
    """

    @abstractmethod
    def __enter__(self) -> "AgentMemory":
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_tasks(self, tasks: List[str]) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_similar_tasks(self, query: str, score_threshold: float = 0.95) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def reset_class(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def should_use() -> bool:
        return True

from agent import Agent
from python.helpers.vector_db import VectorDB
from python.helpers.vector_db import Document
from python.helpers import files
from python.helpers.tool import Tool
from python.helpers.tool import Response
from python.helpers.print_style import PrintStyle

# From tools/memory_tool.py
def search(agent:Agent, query:str, count:int=5, threshold:float=0.1):
    initialize(agent)
    docs = db.search_similarity_threshold(query,count,threshold) # type: ignore
    if len(docs)==0: return files.read_file("./prompts/fw.memories_not_found.md", query=query)
    else: return str(docs)

# From tools/memory_tool.py
def save(agent:Agent, text:str):
    initialize(agent)
    id = db.insert_document(text) # type: ignore
    return files.read_file("./prompts/fw.memory_saved.md", memory_id=id)

# From tools/memory_tool.py
def forget(agent:Agent, query:str):
    initialize(agent)
    deleted = db.delete_documents_by_query(query) # type: ignore
    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)

# From tools/memory_tool.py
def initialize(agent:Agent):
    global db
    if not db:
        dir = os.path.join("memory",agent.config.memory_subdir)
        db = VectorDB(embeddings_model=agent.config.embeddings_model, in_memory=False, cache_dir=dir)

# From tools/memory_tool.py
def extract_guids(text):
    pattern = r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
    return re.findall(pattern, text)

# From tools/memory_tool.py
def execute(self,**kwargs):
        result=""
        
        if "query" in kwargs:
            if "threshold" in kwargs: threshold = float(kwargs["threshold"]) 
            else: threshold = 0.1
            if "count" in kwargs: count = int(kwargs["count"]) 
            else: count = 5
            result = search(self.agent, kwargs["query"], count, threshold)
        elif "memorize" in kwargs:
            result = save(self.agent, kwargs["memorize"])
        elif "forget" in kwargs:
            result = forget(self.agent, kwargs["forget"])
        elif "delete" in kwargs:
            result = delete(self.agent, kwargs["delete"])
                        
        # result = process_query(self.agent, self.args["memory"],self.args["action"], result_count=self.agent.config.auto_memory_count)
        return Response(message=result, break_loop=False)

