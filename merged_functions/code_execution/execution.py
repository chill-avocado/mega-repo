# Merged file for code_execution/execution
# This file contains code merged from multiple repositories

from automata.core.base import AutomataError

# From eval/eval_error.py
class EvalLoadingError(AutomataError):
    """Exception raised when there's an issue with loading evaluations."""

    pass

# From eval/eval_error.py
class EvalExecutionError(AutomataError):
    """Raised when there's an issue during task execution."""

    pass

import os
import uuid
from abc import ABC
from abc import abstractmethod
from collections.abc import Hashable
from enum import Enum
from typing import Any
from typing import Callable
from typing import Optional
from automata.config import TASK_OUTPUT_PATH

# From tasks/task_base.py
class TaskStatus(Enum):
    CREATED = "created"
    REGISTERED = "registered"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    COMMITTED = "committed"
    FAILED = "failed"
    RETRYING = "retrying"

# From tasks/task_base.py
class Task:
    """
    A generic `Task` object used by the `TaskExecutor`.

    A `Task` is responsible for storing the task id, priority, and max retries.
    It receives kwargs that are passed to the task function when the task is executed.
    It also exposes a method to generate a deterministic task id based on the hash of the hashable kwargs.
    """

    TASK_LOG_NAME = "task_SESSION_ID.log"
    TASK_LOG_REL_DIR = "logs"

    def __init__(self, *args, **kwargs) -> None:
        # sourcery skip: remove-redundant-if
        """
        Initializes a task object.

        Keyword Args:
            generate_deterministic_id (bool): Whether to generate a deterministic task id or not.
              In the case of a deterministic task id, the task id is generated based on the hash of
              the hashable kwargs. Defaults to False.
            priority (int): The priority of the task. Defaults to 0.
            max_retries (int): The maximum number of retries for the task. Defaults to 3.
        """
        if (
            "generate_deterministic_id" in kwargs
            and "session_id" not in kwargs
        ):
            self.session_id = self._deterministic_session_id(**kwargs)
        elif (
            "session_id" in kwargs
            and "generate_deterministic_id" not in kwargs
        ):
            self.session_id = kwargs["session_id"]
        elif "generate_deterministic_id" in kwargs and "session_id" in kwargs:
            raise ValueError(
                "Values for both session_id and generate_deterministic_id cannot be provided."
            )
        else:
            self.session_id = str(uuid.uuid4())

        self.priority = kwargs.get("priority", 0)
        self.max_retries = kwargs.get("max_retries", 3)
        self._status = TaskStatus.CREATED
        self.retry_count = 0
        self.observer: Optional[Callable] = None
        self.task_dir = self._get_task_dir(
            kwargs.get("task_dir", TASK_OUTPUT_PATH)
        )
        self.result: Optional[str] = None
        self.error: Optional[str] = None

    def __str__(self):
        return f"Task {self.session_id} ({self.status})"

    def notify_observer(self) -> None:
        """
        Used to notify the observer when the task status changes.
        """
        if self.observer:
            self.observer(self)

    @property
    def status(self) -> TaskStatus:
        """
        The status of the task is updated by the task executor as the task progresses through
        different stages of execution.
        """
        return self._status

    @status.setter
    def status(self, new_status: TaskStatus) -> None:
        """
        Sets the status of the `Task`.

        If the new status is RETRYING, the retry count is incremented
        and the task status is set to FAILED if the retry count exceeds the max retries.
        """
        if new_status == TaskStatus.RETRYING:
            self.retry_count += 1
            if self.retry_count == self.max_retries:
                self._status = TaskStatus.FAILED
            else:
                self._status = new_status
        else:
            self._status = new_status
        self.notify_observer()

    def _deterministic_session_id(self, **kwargs) -> str:
        """
        Returns a deterministic session id for the task which is
        generated based on the hash of the hashable kwargs.

        Keyword Args:
            kwargs (dict): The keyword arguments passed to the task.
        """
        # Generate the hash of the hashable kwargs
        hashable_items = sorted(
            [item for item in kwargs.items() if isinstance(item[1], Hashable)]
        )
        kwargs_hash = hash(tuple(hashable_items))

        # Combine the hashes and use it as a seed for generating a deterministic UUID
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{kwargs_hash}"))

    def _get_task_dir(self, base_dir: str) -> str:
        """
        Gets the output directory for the task.
        Use of the session_id as the directory name ensures that the task directory is unique.
        """
        return os.path.join(base_dir, f"task_{self.session_id}")

    def _get_log_dir(self) -> str:
        """
        Gets the output directory where task logs are saved."""
        return os.path.join(self.task_dir, Task.TASK_LOG_REL_DIR)

# From tasks/task_base.py
class ITaskExecution(ABC):
    """Interface for task execution behaviors."""

    @abstractmethod
    def execute(self, task: Task) -> Any:
        pass

# From tasks/task_base.py
class TaskEnvironment(ABC):
    """An abstract base class for implementing a task task_environment."""

    @abstractmethod
    def setup(self, task: Task):
        """Set up the task_environment."""
        pass

    @abstractmethod
    def teardown(self):
        """Tear down the task_environment."""
        pass

    @abstractmethod
    def validate(self):
        """Validate the task_environment."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the environment to its initial state."""
        pass

# From tasks/task_base.py
def notify_observer(self) -> None:
        """
        Used to notify the observer when the task status changes.
        """
        if self.observer:
            self.observer(self)

# From tasks/task_base.py
def status(self) -> TaskStatus:
        """
        The status of the task is updated by the task executor as the task progresses through
        different stages of execution.
        """
        return self._status

# From tasks/task_base.py
def execute(self, task: Task) -> Any:
        pass

# From tasks/task_base.py
def setup(self, task: Task):
        """Set up the task_environment."""
        pass

# From tasks/task_base.py
def teardown(self):
        """Tear down the task_environment."""
        pass

# From tasks/task_base.py
def validate(self):
        """Validate the task_environment."""
        pass

# From tasks/task_base.py
def reset(self):
        """Reset the environment to its initial state."""
        pass

import logging
import logging.config
import time
from automata.agent import OpenAIAutomataAgent
from automata.config import OpenAIAutomataAgentConfigBuilder
from automata.core.utils import get_logging_config
from automata.memory_store import OpenAIAutomataConversationDatabase
from automata.tasks.automata_task import AutomataTask
from automata.tasks.task_base import ITaskExecution
from automata.tasks.task_base import Task
from automata.tasks.task_base import TaskStatus
from automata.tasks.task_error import TaskGeneralError
from automata.tasks.task_error import TaskStateError

# From tasks/task_executor.py
class IAutomataTaskExecution(ITaskExecution):
    """Class for executing general tasks."""

    def execute(self, task: Task) -> OpenAIAutomataAgent:
        """
        Executes the task by creating and running an AutomataAgent.
        Eachtime the execution fails, the task's retry count is incremented.
        After the maximum number of retries is reached, the task is marked as failed.

        Raises:
            Exception: If the task fails on execution
        """
        if not isinstance(task, AutomataTask):
            raise TaskGeneralError(
                "AutomataTaskEnvironment requires an AutomataTask instance"
            )
        task.status = TaskStatus.RUNNING
        try:
            agent = IAutomataTaskExecution._build_agent(task)
            result = agent.run()
            task.result = result
            task.status = TaskStatus.SUCCESS
            return agent
        except Exception as e:
            logger.exception(f"AutomataTask failed: {e}")
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.retry_count += 1
            raise e

    @staticmethod
    def _build_agent(task: AutomataTask) -> OpenAIAutomataAgent:
        """
        Uses the task's arguments to build an AutomataAgent from
        the OpenAIAutomataAgentConfigBuilder.
        TODO - Consider explicitly passing args to the ConfigFactory
               Instead of passing kwargs to the create_config method.
        """

        agent_config = OpenAIAutomataAgentConfigBuilder.create_from_args(
            session_id=str(task.session_id), **task.kwargs
        )

        agent = OpenAIAutomataAgent(
            task.instructions,
            agent_config,
        )

        if task.record_conversation:
            # TODO - Remove hard coupling of OpenAIProvider to IAutomataTaskExecution
            # then, introduce provider-style workflow if necessary

            # Initialize the OpenAIAutomataConversationDatabase and set it to the agent
            db_provider = OpenAIAutomataConversationDatabase()
            agent.set_database_provider(db_provider)

        return agent

# From tasks/task_executor.py
class AutomataTaskExecutor:
    """A class for using ITaskExecution behavior to execute a task."""

    def __init__(self, execution: ITaskExecution) -> None:
        self.execution = execution

    def execute(self, task: AutomataTask) -> Any:
        """
        Executes the task using the specified execution behavior.

        This method will retry the task if it fails,
        until the maximum number of retries is reached.

        Raises Exception:
            If the task is not status PENDING.
            If the task fails and the maximum number of retries is reached.
        """
        if task.status != TaskStatus.PENDING:
            raise TaskStateError(
                f"Cannot execute task because task is not in PENDING state. Task status = {task.status}"
            )
        for attempt in range(task.max_retries):
            try:
                logger.debug(f"Executing task {task.session_id}")
                task.status = TaskStatus.RUNNING
                result = self.execution.execute(task)
                task.status = TaskStatus.SUCCESS
                logger.info(f"Task {task.session_id} executed successfully.")
                return result
            except Exception as e:
                logging.exception(f"AutomataTask failed: {e}")
                task.status = TaskStatus.RETRYING
                task.error = str(e)

                # If we've used up all retries, re-raise the exception
                if attempt == task.max_retries - 1:
                    raise e

                # Otherwise, wait before retrying
                time.sleep(AutomataTaskExecutor._exponential_backoff(attempt))

    @staticmethod
    def _exponential_backoff(attempt_number: int) -> int:
        return 2**attempt_number

from typing import TYPE_CHECKING
from typing import Sequence
from automata.tools.tool_base import Tool
from automata.llm import FunctionCall

# From tools/tool_executor.py
class IToolExecution(ABC):
    """Interface for executing tools."""

    @abstractmethod
    def execute(self, function_call: "FunctionCall") -> str:
        pass

# From tools/tool_executor.py
class ToolExecution(IToolExecution):
    """Class for executing tools."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def execute(self, function_call: "FunctionCall") -> str:
        if tool := self.tools.get(function_call.name):
            return tool.run(function_call.arguments)
        else:
            raise Exception(
                f"No tool found for function call: {function_call.name}"
            )

# From tools/tool_executor.py
class ToolExecutor:
    """Class for using IToolExecution behavior to execute a tool."""

    def __init__(self, execution: IToolExecution) -> None:
        self.execution = execution

    def execute(self, function_call: "FunctionCall") -> str:
        return self.execution.execute(function_call)

import random
import shutil
from typing import Dict
from typing import Generator
from typing import List
from typing import Set
from unittest.mock import MagicMock
import numpy
import pytest
from automata.agent import AgentToolkitNames
from automata.config import AgentConfigName
from automata.embedding import EmbeddingSimilarityCalculator
from automata.eval import AgentEvaluationHarness
from automata.eval import CodeWritingEval
from automata.eval import OpenAIFunctionEval
from automata.eval import SymbolSearchEval
from automata.eval import ToolEvaluationHarness
from automata.eval.agent.agent_eval_composite import AgentEvalComposite
from automata.eval.agent.agent_eval_database import AgentEvalResultDatabase
from automata.experimental.search import SymbolRankConfig
from automata.experimental.search import SymbolSearch
from automata.experimental.tools import AgentifiedSearchToolkitBuilder
from automata.memory_store import SymbolCodeEmbeddingHandler
from automata.singletons.dependency_factory import dependency_factory
from automata.singletons.github_client import GitHubClient
from automata.symbol import Symbol
from automata.symbol import SymbolGraph
from automata.symbol import parse_symbol
from automata.symbol_embedding import ChromaSymbolEmbeddingVectorDatabase
from automata.symbol_embedding import JSONSymbolEmbeddingVectorDatabase
from automata.symbol_embedding import SymbolCodeEmbedding
from automata.symbol_embedding import SymbolDocEmbedding
from automata.tasks import AutomataAgentTaskDatabase
from automata.tasks import AutomataTask
from automata.tasks import AutomataTaskEnvironment
from automata.tasks import AutomataTaskExecutor
from automata.tasks import AutomataTaskRegistry
from automata.tasks import IAutomataTaskExecution
from automata.tools import Tool
from automata.tools import ToolExecution
from automata.tools import ToolExecutor
from automata.tools.agent_tool_factory import AgentToolFactory
import unittest.mock

# From tests/conftest.py
class TestTool(Tool):
    def run(self, tool_input: Dict[str, str]) -> str:
        return "TestTool response"

# From tests/conftest.py
def temp_output_dir() -> Generator:
    """Creates a temporary output filename which is deleted after the test is run"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "test_output_vec")
    if not os.path.exists(filename):
        os.mkdir(filename)
    yield filename
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except OSError:
        pass

    # The TemporaryDirectory context manager should already clean up the directory,
    # but just in case it doesn't (e.g. due to an error), we'll try removing it manually as well.
    try:
        shutil.rmtree(f"{filename}/")
    except OSError:
        pass

# From tests/conftest.py
def temp_output_filename(temp_output_dir: str) -> str:
    """Creates a temporary output filename which is deleted after the test is run"""
    return os.path.join(temp_output_dir, "test_output.json")

# From tests/conftest.py
def symbols() -> List[Symbol]:
    """
    Mock several realistic symbols for testing

    Note:
        These symbols at one point reflected existing code
        but they are not guaranteed to be up to date.
    """
    return [
        # Symbol with a simple attribute
        parse_symbol(
            "scip-python python automata v0.0.0 `config.automata_agent_config`/AutomataAgentConfig#description."
        ),
        # Symbol with a method with foreign argument
        parse_symbol(
            "scip-python python automata v0.0.0 `config.automata_agent_config`/AutomataAgentConfig#load().(config_name)"
        ),
        # Symbol with a locally defined object
        parse_symbol(
            "scip-python python automata v0.0.0 `core.tasks.automata_task_executor`/logger."
        ),
        # Symbol with a class object and class variable
        parse_symbol(
            "scip-python python automata v0.0.0 `config.automata_agent_config`/AutomataAgentConfig#verbose."
        ),
        # Symbol with a class method
        parse_symbol(
            "scip-python python automata v0.0.0 `evals.eval_helpers`/EvalAction#__init__().(action)"
        ),
        # Symbol with an object
        parse_symbol(
            "scip-python python automata v0.0.0 `core.agent.automata_agent_enums`/ActionIndicator#CODE."
        ),
        # Class Name
        parse_symbol(
            "scip-python python automata v0.0.0 `core.agent.automata_agent_enums`/ActionIndicator#"
        ),
        # Init
        parse_symbol(
            "scip-python python automata v0.0.0 `core.tools.base`/ToolNotFoundError#__init__()."
        ),
    ]

# From tests/conftest.py
def mock_simple_method_symbols() -> List[Symbol]:
    """Returns a list of 100 mock symbols with a simple method"""
    return [
        parse_symbol(
            EXAMPLE_SYMBOL_PREFIX + str(random.random()) + "_uri_ex_method()."
        )
        for _ in range(100)
    ]

# From tests/conftest.py
def embedding_type(request):
    return request.param

# From tests/conftest.py
def embedded_symbol(symbols, request):
    data = [("x", [1, 2, 3]), ("y", [1, 2, 3, 4])][request.param]
    return SymbolCodeEmbedding(symbols[request.param], data[0], data[1])

# From tests/conftest.py
def mock_embedding():
    """Returns a random mock embedding vector"""
    return np.array([random.random() for _ in range(1024)])

# From tests/conftest.py
def embedding_maker(embedding_type):
    def _make_embedding(symbol, document, vector):
        if embedding_type == SymbolDocEmbedding:
            # Add extra fields for SymbolDocEmbedding
            return embedding_type(
                symbol,
                document,
                vector,
                source_code="some code",
                summary="summary",
                context="context",
            )
        else:
            return embedding_type(symbol, document, vector)

    return _make_embedding

# From tests/conftest.py
def symbol_graph_mock(mocker):
    """Mock a SymbolGraph object for cases where we don't need to test the graph itself"""
    return mocker.MagicMock(spec=SymbolGraph)

# From tests/conftest.py
def symbol_search(mocker, symbol_graph_mock):
    """Creates a SymbolSearch object with Mock dependencies for testing"""
    symbol_similarity_mock = mocker.MagicMock(
        spec=EmbeddingSimilarityCalculator
    )
    symbol_similarity_mock.embedding_handler = mocker.MagicMock(
        spec=SymbolCodeEmbeddingHandler
    )

    symbol_code_embedding_handler = mocker.MagicMock(
        spec=SymbolCodeEmbeddingHandler
    )

    symbol_rank_config_mock = mocker.MagicMock(spec=SymbolRankConfig)
    symbol_rank_config_mock.validate_config = mocker.MagicMock()

    return SymbolSearch(
        symbol_graph_mock,
        symbol_rank_config_mock,
        symbol_code_embedding_handler,
        symbol_similarity_mock,
    )

# From tests/conftest.py
def json_vector_db(
    tmpdir_factory,
) -> Generator[JSONSymbolEmbeddingVectorDatabase, Any, Any]:
    """Creates a JSONSymbolEmbeddingVectorDatabase object for testing"""
    db_file = tmpdir_factory.mktemp("data").join("test_json.db")
    yield JSONSymbolEmbeddingVectorDatabase(str(db_file))
    if os.path.exists(str(db_file)):
        os.remove(str(db_file))

# From tests/conftest.py
def chroma_vector_db(embedding_type) -> ChromaSymbolEmbeddingVectorDatabase:
    """Creates a in-memory Chroma Symbol database for testing"""
    return ChromaSymbolEmbeddingVectorDatabase(
        CHROMA_COLLECTION_NAME, factory=embedding_type.from_args
    )

# From tests/conftest.py
def chroma_vector_db_persistent(
    embedding_type,
    tmpdir_factory,
) -> Generator[ChromaSymbolEmbeddingVectorDatabase, Any, Any]:
    db_file = tmpdir_factory.mktemp("data").join("test_json.db")
    db_dir = os.path.dirname(str(db_file))
    """Creates a persistent Chroma Symbol database for testing"""
    yield ChromaSymbolEmbeddingVectorDatabase(
        CHROMA_COLLECTION_NAME,
        factory=embedding_type.from_args,
        persist_directory=db_dir,
    )
    if os.path.exists(str(db_file)):
        os.remove(str(db_file))

# From tests/conftest.py
def conversation_db(
    tmpdir_factory,
) -> Generator[OpenAIAutomataConversationDatabase, Any, Any]:
    db_file = tmpdir_factory.mktemp("data").join("test_conversation.db")
    db = OpenAIAutomataConversationDatabase(str(db_file))
    yield db
    db.close()
    if os.path.exists(str(db_file)):
        os.remove(str(db_file))

# From tests/conftest.py
def task_db(tmpdir_factory) -> Generator[AutomataAgentTaskDatabase, Any, Any]:
    db_file = tmpdir_factory.mktemp("data").join("test_task.db")
    db = AutomataAgentTaskDatabase(str(db_file))
    yield db
    db.close()
    if os.path.exists(str(db_file)):
        os.remove(str(db_file))

# From tests/conftest.py
def eval_db(tmpdir_factory) -> Generator[AgentEvalResultDatabase, Any, Any]:
    db_file = tmpdir_factory.mktemp("data").join("test_eval.db")
    db = AgentEvalResultDatabase(str(db_file))
    yield db
    db.close()
    if os.path.exists(str(db_file)):
        os.remove(str(db_file))

# From tests/conftest.py
def automata_agent_config_builder():
    config_name = AgentConfigName.TEST
    # We must mock the get method on the dependency factory at this location
    # Otherwise, the dependency factory will attempt to actually instantiate the dependencies
    import unittest.mock

    dependency_factory.get = unittest.mock.MagicMock(return_value=None)

    return OpenAIAutomataAgentConfigBuilder.from_name(config_name)

# From tests/conftest.py
def automata_agent(mocker, automata_agent_config_builder):
    """Creates a mock AutomataAgent object for testing"""

    llm_toolkits_list = ["advanced-context-oracle"]
    dependencies: Set[Any] = set()
    for tool in llm_toolkits_list:
        for dependency_name, _ in AgentToolFactory.TOOLKIT_TYPE_TO_ARGS[
            AgentToolkitNames(tool)
        ]:
            dependencies.add(dependency_name)

    kwargs = {
        dependency: dependency_factory.get(dependency)
        for dependency in dependencies
    }
    tools = AgentToolFactory.build_tools(["advanced-context-oracle"], **kwargs)

    instructions = "Test instruction."

    return OpenAIAutomataAgent(
        instructions,
        config=automata_agent_config_builder.with_tools(tools)
        .with_stream(False)
        .with_system_template_formatter({})
        .build(),
    )

# From tests/conftest.py
def tasks():
    repo_manager = MagicMock()
    task_0 = AutomataTask(
        repo_manager,
        # session_id = automata_agent.session_id,
        config_to_load=AgentConfigName.TEST.to_path(),
        instructions="This is a test.",
        session_id=str(uuid.uuid4()),
    )

    task_1 = AutomataTask(
        repo_manager,
        # session_id = automata_agent.session_id,
        config_to_load=AgentConfigName.TEST.to_path(),
        instructions="This is a test2.",
    )
    return [task_0, task_1]

# From tests/conftest.py
def task_w_agent_matched_session(automata_agent):
    repo_manager = MagicMock()
    return AutomataTask(
        repo_manager,
        session_id=automata_agent.session_id,
        config_to_load=AgentConfigName.TEST.to_path(),
        instructions="This is a test.",
    )

# From tests/conftest.py
def task_environment():
    github_mock = MagicMock(spec=GitHubClient)
    return AutomataTaskEnvironment(github_mock)

# From tests/conftest.py
def task_registry(task_db):
    return AutomataTaskRegistry(task_db)

# From tests/conftest.py
def function_evaluator():
    return OpenAIFunctionEval()

# From tests/conftest.py
def code_evaluator():
    return CodeWritingEval(target_variables=["x", "y", "z"])

# From tests/conftest.py
def search_evaluator():
    return SymbolSearchEval()

# From tests/conftest.py
def composite_evaluator(function_evaluator, code_evaluator):
    evaluators = [function_evaluator, code_evaluator]
    return AgentEvalComposite(evaluators)

# From tests/conftest.py
def agent_eval_harness(function_evaluator, code_evaluator):
    database = MagicMock()
    return AgentEvaluationHarness(
        [function_evaluator, code_evaluator], database
    )

# From tests/conftest.py
def tool_eval_harness(search_evaluator):
    return ToolEvaluationHarness([search_evaluator])

# From tests/conftest.py
def matched_setup(
    mocker,
    automata_agent,
    task_w_agent_matched_session,
    task_environment,
    task_registry,
    conversation_db,
):
    # Mock the API response
    mock_openai_chatcompletion_create = mocker.patch(
        "openai.ChatCompletion.create"
    )

    # Register and setup task
    task_registry.register(task_w_agent_matched_session)
    task_environment.setup(task_w_agent_matched_session)

    # Use the agent's set_database_provider method
    automata_agent.set_database_provider(conversation_db)

    execution = IAutomataTaskExecution()
    IAutomataTaskExecution._build_agent = MagicMock(
        return_value=automata_agent
    )
    task_executor = AutomataTaskExecutor(execution)

    return (
        mock_openai_chatcompletion_create,
        automata_agent,
        task_executor,
        task_registry,
    )

# From tests/conftest.py
def test_tool(request) -> TestTool:
    name = request.node.get_closest_marker("tool_name")
    description = request.node.get_closest_marker("tool_description")
    function = request.node.get_closest_marker("tool_function")

    return TestTool(
        name=name.args[0] if name else "TestTool",
        description=description.args[0]
        if description
        else "A test tool for testing purposes",
        function=function.args[0]
        if function
        else (lambda x: "TestTool response"),
    )

# From tests/conftest.py
def test_tool_instantiation(test_tool: TestTool) -> None:
    """Test that the TestTool class can be instantiated."""
    assert test_tool.name == "TestTool"
    assert test_tool.description == "A test tool for testing purposes"
    assert test_tool.function is not None

# From tests/conftest.py
def function_call() -> FunctionCall:
    return FunctionCall(name="TestTool", arguments={"test": "test"})

# From tests/conftest.py
def tool_execution(test_tool) -> ToolExecution:
    return ToolExecution([test_tool])

# From tests/conftest.py
def tool_executor(tool_execution) -> ToolExecutor:
    return ToolExecutor(tool_execution)

# From tests/conftest.py
def setup_tool(mocker, symbols):
    # Mock the API response

    # Create a mock for the SymbolSearch instance
    symbol_search_mock = MagicMock()

    symbol_0 = symbols[0]
    symbol_1 = symbols[1]
    symbol_2 = symbols[2]

    # Mock the get_symbol_rank_results method
    symbol_search_mock.get_symbol_rank_results = MagicMock(
        return_value=[
            (symbol_0, 0.3),
            (symbol_1, 0.2),
            (symbol_2, 0.1),
            (symbols[3], 0.01),  # pad to top k (5)
            (symbols[4], 0.01),
        ]
    )

    # Create tools using the factory
    symbol_search_tools = AgentToolFactory.create_tools_from_builder(
        AgentToolkitNames.SYMBOL_SEARCH, symbol_search=symbol_search_mock
    )

    # Create the tool execution instance with the tools
    tool_execution = ToolExecution(symbol_search_tools)

    return ToolExecutor(execution=tool_execution)

# From tests/conftest.py
def agentified_search_tool_builder(symbols):
    """Returns an agentified search tool builder mock"""
    symbol_search_mock = MagicMock(spec=SymbolSearch)
    symbol_doc_embedding_handler_mock = MagicMock()
    completion_provider_mock = MagicMock()

    # set the return value on the symbol_search_mock
    symbol_search_mock.get_symbol_code_similarity_results = MagicMock(
        return_value=[(symbols[0], 1), (symbols[1], 0.8), (symbols[2], 0.6)]
    )

    agentified_search_tool_builder = AgentifiedSearchToolkitBuilder(
        symbol_search=symbol_search_mock,
        symbol_doc_embedding_handler=symbol_doc_embedding_handler_mock,
        completion_provider=completion_provider_mock,
    )
    agentified_search_tool_builder.completion_provider.standalone_call = (
        MagicMock(return_value=symbols[1].dotpath)
    )

    return agentified_search_tool_builder

# From tests/conftest.py
def run(self, tool_input: Dict[str, str]) -> str:
        return "TestTool response"

import json
from tqdm import tqdm
from automata.eval.eval_base import Action
from automata.eval.eval_base import Payload
from automata.eval.tool.search_eval import SymbolSearchAction
from automata.eval.tool.tool_eval import ToolEval
from automata.eval.tool.tool_eval import ToolEvalResult
from automata.eval.tool.tool_eval_metrics import ToolEvaluationMetrics
from automata.eval import parse_action_from_payload

# From tool/tool_eval_harness.py
class ToolEvalSetLoader:
    """Loads a list of function calls and their expected actions from a JSON file."""

    def __init__(self, filepath: str):
        from automata.eval import parse_action_from_payload

        self.filepath = filepath
        if not filepath.endswith(".json"):
            raise ValueError(
                f"Only JSON files are supported, received filepath {filepath}."
            )
        payloads = self.load_json()
        self.input_functions: List[FunctionCall] = []
        self.expected_actions: List[Action] = []

        for item in payloads:
            template = item["template"]
            entries = item["entries"]

            for entry in entries:
                # TODO - Avoid using type ignore below.
                payload = self.format_values(template, entry)  # type: ignore

                input_func_call = payload.get("input_function")
                expected_action = payload.get("expected_action")

                if not isinstance(input_func_call, dict):
                    raise ValueError("Function call must be a dictionary.")
                if not isinstance(expected_action, dict):
                    raise ValueError("Expected action must be a dictionary.")

                self.input_functions.append(
                    FunctionCall(
                        name=input_func_call["name"],
                        arguments=input_func_call["arguments"],
                    )
                )
                self.expected_actions.append(
                    parse_action_from_payload(expected_action)
                )

    def format_values(self, obj: Any, formatter: Dict[str, str]) -> Any:
        """Recursively apply formatter to all string values in the object."""
        if isinstance(obj, str):
            return obj.format(**formatter)
        elif isinstance(obj, list):
            return [self.format_values(item, formatter) for item in obj]
        elif isinstance(obj, dict):
            return {
                k: self.format_values(v, formatter) for k, v in obj.items()
            }
        else:
            return obj

    def load_json(self) -> List[Payload]:
        """Loads the JSON file."""
        try:
            logging.info(f"Loading json from {self.filepath}...")
            with open(self.filepath, "r") as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} tasks.")
            return data
        except Exception as e:
            raise EvalLoadingError from e

# From tool/tool_eval_harness.py
class ToolEvaluationHarness:
    """A class to evaluate a list of function calls against a list of expected actions."""

    def __init__(self, evals: List[ToolEval], **kwargs):
        self.evals = evals
        self.run_id = str(uuid.uuid4())
        random.seed(kwargs.get("random_seed", 123))

    def evaluate(
        self,
        input_functions: List[FunctionCall],
        expected_actions: List[Action],
        executor: ToolExecution,
    ) -> ToolEvaluationMetrics:
        """Returns the evaluation metrics for the given function calls and expected actions."""

        logging.info(
            f"Starting evaluation of {len(input_functions)} function calls with run_id={self.run_id}..."
        )

        aggregate_results = []
        function_action_zip = list(zip(input_functions, expected_actions))
        random.shuffle(function_action_zip)
        for input_function, expected_action in tqdm(function_action_zip):
            # TODO - Why are we struggling with initializers
            # We should root out the issue, rather than skipping.
            if (
                isinstance(expected_action, SymbolSearchAction)
                and expected_action.search_results
                and "__init__" in expected_action.search_results[0]
            ):
                continue
            try:
                for eval in self.evals:
                    result = eval.generate_eval_result(
                        input_function,
                        expected_action,
                        executor,
                        run_id=self.run_id,
                    )
                    if not isinstance(result, ToolEvalResult):
                        raise ValueError(
                            "Evaluators must return a ToolEvalResult."
                        )
                    # TODO - Re-enable this once we have a database
                    # self.database.write_result(result)
                    aggregate_results.append(result)
            except Exception as e:
                logging.error(f"Error during function call execution: {e}")
                raise EvalExecutionError from e
        return ToolEvaluationMetrics(aggregate_results)

# From tool/tool_eval_harness.py
def format_values(self, obj: Any, formatter: Dict[str, str]) -> Any:
        """Recursively apply formatter to all string values in the object."""
        if isinstance(obj, str):
            return obj.format(**formatter)
        elif isinstance(obj, list):
            return [self.format_values(item, formatter) for item in obj]
        elif isinstance(obj, dict):
            return {
                k: self.format_values(v, formatter) for k, v in obj.items()
            }
        else:
            return obj

# From tool/tool_eval_harness.py
def load_json(self) -> List[Payload]:
        """Loads the JSON file."""
        try:
            logging.info(f"Loading json from {self.filepath}...")
            with open(self.filepath, "r") as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} tasks.")
            return data
        except Exception as e:
            raise EvalLoadingError from e

# From tool/tool_eval_harness.py
def evaluate(
        self,
        input_functions: List[FunctionCall],
        expected_actions: List[Action],
        executor: ToolExecution,
    ) -> ToolEvaluationMetrics:
        """Returns the evaluation metrics for the given function calls and expected actions."""

        logging.info(
            f"Starting evaluation of {len(input_functions)} function calls with run_id={self.run_id}..."
        )

        aggregate_results = []
        function_action_zip = list(zip(input_functions, expected_actions))
        random.shuffle(function_action_zip)
        for input_function, expected_action in tqdm(function_action_zip):
            # TODO - Why are we struggling with initializers
            # We should root out the issue, rather than skipping.
            if (
                isinstance(expected_action, SymbolSearchAction)
                and expected_action.search_results
                and "__init__" in expected_action.search_results[0]
            ):
                continue
            try:
                for eval in self.evals:
                    result = eval.generate_eval_result(
                        input_function,
                        expected_action,
                        executor,
                        run_id=self.run_id,
                    )
                    if not isinstance(result, ToolEvalResult):
                        raise ValueError(
                            "Evaluators must return a ToolEvalResult."
                        )
                    # TODO - Re-enable this once we have a database
                    # self.database.write_result(result)
                    aggregate_results.append(result)
            except Exception as e:
                logging.error(f"Error during function call execution: {e}")
                raise EvalExecutionError from e
        return ToolEvaluationMetrics(aggregate_results)

import jsonpickle
from automata.eval.agent.agent_eval import AgentEval
from automata.eval.eval_base import register_action
from automata.llm import LLMChatMessage
from automata.llm import OpenAIChatMessage

# From agent/code_writing_eval.py
class CodeExecutionError(AutomataError):
    """Exception raised when there's an error executing the code."""

    pass

# From agent/code_writing_eval.py
class VariableNotFoundError(AutomataError):
    """Exception raised when the target variable is not found."""

    pass

# From agent/code_writing_eval.py
class CodeWritingAction(Action):
    """An concrete action representing written code."""

    BACKWARD_LANGUAGE_MARKER_POSITION = 1
    FORWARD_LANGUAGE_MARKER_POSITION = 0

    def __init__(
        self,
        py_object: Optional[Any],
        error: Optional[str] = None,
    ):  # sourcery skip: docstrings-for-functions
        self.py_object = py_object
        self.error = error

    def __eq__(self, other):  # sourcery skip: docstrings-for-functions
        if not isinstance(other, CodeWritingAction):
            return False
        return self.py_object == other.py_object

    def __hash__(self):
        return hash((jsonpickle.dumps(self.py_object),))

    def __repr__(self):
        return f"CodeWritingAction(py_object={self.py_object}, error={self.error})"

    def to_payload(self) -> Payload:
        """Converts a CodeWritingAction into a payload for storing."""

        return {
            "type": "CodeWritingAction",
            "py_object": str(jsonpickle.encode(self.py_object)),
            "error": self.error or "None",
        }

    @classmethod
    def from_payload(cls, payload: Payload) -> "CodeWritingAction":
        """Converts a payload CodeWritingAction into underlying payload."""

        if not isinstance(payload["py_object"], (str, dict)):
            raise ValueError(
                f"Object types of type={type(object)} received, instead of str."
            )
        # Extract the string from the payload
        json_string = payload["py_object"]

        if not isinstance(json_string, str):
            raise ValueError(
                "Expected a string, but received a non-string object."
            )

        # Replace single quotes with double quotes
        json_string_with_double_quotes = json_string.replace("'", '"')

        # Replace None with null
        json_string_with_double_quotes = (
            json_string_with_double_quotes.replace("None", "null")
        )

        # Decode the corrected JSON string using jsonpickle
        py_object = jsonpickle.decode(json_string_with_double_quotes)

        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(
                f"Object types of type={type(error)} received, instead of str."
            )

        return cls(
            py_object=py_object,
            error=error,
        )

    @staticmethod
    def _extract_snippet(
        snippet: str, expected_language: str = "python"
    ) -> str:
        """Extracts a code snippet from a markdown string."""

        return snippet.split(f"```{expected_language}")[
            CodeWritingAction.BACKWARD_LANGUAGE_MARKER_POSITION
        ].split("```")[CodeWritingAction.FORWARD_LANGUAGE_MARKER_POSITION]

# From agent/code_writing_eval.py
class CodeWritingEval(AgentEval):
    """A class for evaluating an LLM's code writing ability."""

    def __init__(
        self,
        target_variables: List[str] = ["x"],
        *args,
        **kwargs,
    ):
        self.target_variables = target_variables

    def extract_action(self, message: LLMChatMessage) -> List[Action]:
        """Extracts the coding action explicitly"""

        actions: List[Action] = []

        if (
            not isinstance(message, OpenAIChatMessage)
            or not message.function_call
            or message.function_call.name != "call-termination"
        ):
            return actions

        arguments = message.function_call.arguments

        if "result" not in arguments:
            return actions

        parsed_snippets = self._parse_code_snippet(arguments["result"])

        for snippet in parsed_snippets:
            action = CodeWritingAction(
                py_object=snippet.get("py_object"),
                error=snippet.get("error"),
            )
            actions.append(action)

        return actions

    def _parse_code_snippet(self, raw_content: str) -> List[Dict[str, Any]]:
        """Parses a code snippet and extracts the object value and type at the specified variable."""

        if "```" not in raw_content:
            return []
        # Isolate the exec environment to a separate dictionary
        isolated_locals: Dict[str, Any] = {}
        try:
            code_snippet = CodeWritingAction._extract_snippet(raw_content)
            # Execute the code snippet
            try:
                exec(code_snippet, None, isolated_locals)
            except Exception as e:
                return [
                    {
                        "error": str(
                            CodeExecutionError(
                                f"Error executing code: {str(e)}"
                            )
                        ),
                    }
                ]

            if targets := [
                isolated_locals.get(target_variable)
                for target_variable in self.target_variables
                if target_variable in isolated_locals
            ]:
                return [{"py_object": py_object} for py_object in targets]

            else:
                raise VariableNotFoundError(
                    f"Variables '{self.target_variables}' not found in the executed code."
                )
        except Exception as e:
            # If there's an error executing the code, return that.
            raise CodeExecutionError(f"Error executing code: {str(e)}") from e

    def _filter_actions(self, actions: List[Action]) -> List[Action]:
        """Filters out non-CodeWritingActions."""

        return [
            action
            for action in actions
            if isinstance(action, CodeWritingAction)
        ]

# From agent/code_writing_eval.py
def to_payload(self) -> Payload:
        """Converts a CodeWritingAction into a payload for storing."""

        return {
            "type": "CodeWritingAction",
            "py_object": str(jsonpickle.encode(self.py_object)),
            "error": self.error or "None",
        }

# From agent/code_writing_eval.py
def from_payload(cls, payload: Payload) -> "CodeWritingAction":
        """Converts a payload CodeWritingAction into underlying payload."""

        if not isinstance(payload["py_object"], (str, dict)):
            raise ValueError(
                f"Object types of type={type(object)} received, instead of str."
            )
        # Extract the string from the payload
        json_string = payload["py_object"]

        if not isinstance(json_string, str):
            raise ValueError(
                "Expected a string, but received a non-string object."
            )

        # Replace single quotes with double quotes
        json_string_with_double_quotes = json_string.replace("'", '"')

        # Replace None with null
        json_string_with_double_quotes = (
            json_string_with_double_quotes.replace("None", "null")
        )

        # Decode the corrected JSON string using jsonpickle
        py_object = jsonpickle.decode(json_string_with_double_quotes)

        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(
                f"Object types of type={type(error)} received, instead of str."
            )

        return cls(
            py_object=py_object,
            error=error,
        )

# From agent/code_writing_eval.py
def extract_action(self, message: LLMChatMessage) -> List[Action]:
        """Extracts the coding action explicitly"""

        actions: List[Action] = []

        if (
            not isinstance(message, OpenAIChatMessage)
            or not message.function_call
            or message.function_call.name != "call-termination"
        ):
            return actions

        arguments = message.function_call.arguments

        if "result" not in arguments:
            return actions

        parsed_snippets = self._parse_code_snippet(arguments["result"])

        for snippet in parsed_snippets:
            action = CodeWritingAction(
                py_object=snippet.get("py_object"),
                error=snippet.get("error"),
            )
            actions.append(action)

        return actions

import ast
import contextlib
import io
import signal
from typing import Tuple
from automata.agent import AgentToolkitBuilder
from automata.agent import OpenAIAgentToolkitBuilder
from automata.config import LLMProvider
from automata.llm import OpenAITool
from automata.singletons.toolkit_registry import OpenAIAutomataAgentToolkitRegistry

# From builders/py_interpreter.py
class PyInterpreter:
    """This class provides an execution environment for the agent."""

    SUCCESS_STRING = "Execution successful."
    DEFAULT_CODE_CONTEXT = "from typing import *\nfrom collections import *\nimport numpy as np\nimport sympy as sp\n"
    DEFAULT_TEST_CONTEXT = ""

    def __init__(self):
        self.code_context: List[
            str
        ] = PyInterpreter.DEFAULT_CODE_CONTEXT.split("\n")

        self.test_context: List[
            str
        ] = PyInterpreter.DEFAULT_TEST_CONTEXT.split("\n")

    def __repr__(self) -> str:
        return f"PyInterpreter(code_context={self.code_context}, test_context={self.test_context})"

    def _attempt_execution(self, code: str) -> Tuple[bool, str]:
        """Attempts to execute the provided code."""
        output_buffer = io.StringIO()
        try:
            return self._execute_code(
                "\n".join(self.code_context) + "\n" + code, output_buffer
            )
        except Exception as e:
            error_message = str(e) or "Unknown error occurred."
            if error_output := output_buffer.getvalue().strip():
                error_message += f"\nOutput before error:\n{error_output}"
            return False, f"Execution failed with error = {error_message}"

    def _execute_code(
        self, code: str, output_buffer: io.StringIO
    ) -> Tuple[bool, str]:
        """Attempts to execute the provided code."""
        exec_payload = "try:\n" + "\n".join(
            [f"    {line}" for line in code.split("\n") + ["pass"]]
        )
        exec_payload += "\nexcept AssertionError as ae:\n"
        exec_payload += "    global_exception = 'AssertionError on line ' + str(ae.__traceback__.tb_lineno) + ': ' + str(ae)\n"
        exec_payload += "    raise ValueError(f'An assertion error occurred on line {ae.__traceback__.tb_lineno}')"

        exec_payload += "\nexcept Exception as e:\n"
        exec_payload += "    global_exception = e\n"
        exec_payload += "    raise e"

        def handler(signum, frame) -> TimeoutError:
            raise TimeoutError("Execution timed out")

        signal.signal(signal.SIGALRM, handler)
        # TODO - move to decorator to enable cross-platform compatibility
        # external dependency `timeout_decorator` is one potential choice

        signal.alarm(5)  # Set a 5-second alarm

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(exec_payload, {**globals()})
            if execution_output := output_buffer.getvalue().strip():
                return (
                    True,
                    f"{PyInterpreter.SUCCESS_STRING}\nOutput:\n{execution_output}",
                )
            else:
                return True, PyInterpreter.SUCCESS_STRING
        except Exception as e:
            return (
                False,
                f"Execution failed with error '{e}' after outputting {output_buffer.getvalue().strip() or None}",
            )
        finally:
            signal.alarm(0)  # Disable the alarm

    def set_tests(self, code: str, overwrite: bool = True) -> str:
        """Sets up the provided code and persists the context to the local execution buffer."""
        # Add extra handling for string input
        if isinstance(overwrite, str):
            overwrite = overwrite.lower() == "true"
        if overwrite:
            self.test_context = []
        code = self._clean_markdown(code)
        try:
            result: Optional[str] = None
            ast.parse(code)
            if self.code_context != PyInterpreter.DEFAULT_CODE_CONTEXT.split(
                "\n"
            ):
                code = "\n".join(self.code_context) + "\n" + code
                status, result = self._attempt_execution(code)
                if not status:
                    return result
            self.test_context.extend(code.split("\n"))
            return (
                f"{PyInterpreter.SUCCESS_STRING}\nresult = {result}"
                if result is not None
                else PyInterpreter.SUCCESS_STRING
            )
        except Exception as e:
            return f"Execution failed with error '{e}'."

    def set_code(self, code: str, overwrite: bool = True) -> Tuple[bool, str]:
        """Sets up the provided code and persists the context to the local execution buffer."""
        # Add extra handling for string input
        if isinstance(overwrite, str):
            overwrite = overwrite.lower() == "true"
        if overwrite:
            self.code_context = [
                str(ele)
                for ele in PyInterpreter.DEFAULT_CODE_CONTEXT.split("\n")
            ]

        code = self._clean_markdown(code)
        status, result = self._attempt_execution(code)
        if status:
            self.code_context.extend(code.split("\n"))
        return status, result

    def set_code_and_run_tests(self, code: str, overwrite: bool = True) -> str:
        """Set the code and then run the local tests"""
        status, result = self.set_code(code, overwrite)
        result = f"Code Exec Result:\n{result}"
        if status:
            result += "\n" + f"Test Exec Result:\n{self._run_tests()}"
        return result

    def _run_tests(self) -> str:
        """Runs the internal test code."""
        code = "\n".join(self.test_context)
        _, result = self._attempt_execution(code)
        return result

    @staticmethod
    def _clean_markdown(code: str) -> str:
        """Clean the markdown code to be executable."""
        if "```python" in code:
            code = code.split("```python")[1]
        return code.split("```")[0]

# From builders/py_interpreter.py
class PyInterpreterToolkitBuilder(AgentToolkitBuilder):
    """A builder for tools which provide an execution environment for the agent"""

    def __init__(self, *args, **kwargs) -> None:
        self.py_interpreter = PyInterpreter()

    def build(self) -> List[Tool]:
        """Builds the tools for the interpreter."""
        return [
            Tool(
                name="py-set-and-run-tests",
                function=self.py_interpreter.set_tests,
                description="Sets up the provided Python markdown snippet in the test environment and executes the tests and previously provided code from `py-set-code-and-run-tests`. Sympy and Numpy have already been imported as `sp` and `np`, respectively.\n\nAny provided code will be parsed and persisted across interactions. If `overwrite` is set to true then the existing test code is overwritten. The user must call `print(...)` on any output they would like to see returned from the environment. For instance, if your execution terminates with a variable x, to find out what x evaluates to you should end the code with `print(x)`.",
            ),
            Tool(
                name="py-set-code-and-run-tests",
                function=self.py_interpreter.set_code_and_run_tests,
                description="Sets up the provided Python markdown snippet in the local source environment. The code is executed and its context is persisted in the source environment across interactions. After successfully executing the provided code, the provided tests are then ran. If `overwrite` is set to true then existing source code environment is overwritten (but not the tests).",
            ),
        ]

# From builders/py_interpreter.py
class PyInterpreterOpenAIToolkitBuilder(
    PyInterpreterToolkitBuilder, OpenAIAgentToolkitBuilder
):
    TOOL_NAME = AgentToolkitNames.PY_INTERPRETER
    LLM_PROVIDER = LLMProvider.OPENAI

    def build_for_open_ai(self) -> List[OpenAITool]:
        """Builds the tools associated with the Python interpreter for the OpenAI API."""
        tools = super().build()

        # Predefined properties and required parameters
        properties = {
            "code": {
                "type": "string",
                "description": "The given Python code to execute, formatted as a markdown snippet, e.g. ```python\\n[CODE]``` and with newlines separated by the double-escaped newline char '\\n'.",
            },
            "overwrite": {
                "type": "string",
                "description": "Specifies whether or not the given code should overwrite the existing code in the interpreter.",
                "default": "True",
            },
        }
        required = ["code"]

        return [
            OpenAITool(
                function=tool.function,
                name=tool.name,
                description=tool.description,
                properties=properties,
                required=required,
            )
            for tool in tools
        ]

# From builders/py_interpreter.py
def set_tests(self, code: str, overwrite: bool = True) -> str:
        """Sets up the provided code and persists the context to the local execution buffer."""
        # Add extra handling for string input
        if isinstance(overwrite, str):
            overwrite = overwrite.lower() == "true"
        if overwrite:
            self.test_context = []
        code = self._clean_markdown(code)
        try:
            result: Optional[str] = None
            ast.parse(code)
            if self.code_context != PyInterpreter.DEFAULT_CODE_CONTEXT.split(
                "\n"
            ):
                code = "\n".join(self.code_context) + "\n" + code
                status, result = self._attempt_execution(code)
                if not status:
                    return result
            self.test_context.extend(code.split("\n"))
            return (
                f"{PyInterpreter.SUCCESS_STRING}\nresult = {result}"
                if result is not None
                else PyInterpreter.SUCCESS_STRING
            )
        except Exception as e:
            return f"Execution failed with error '{e}'."

# From builders/py_interpreter.py
def set_code(self, code: str, overwrite: bool = True) -> Tuple[bool, str]:
        """Sets up the provided code and persists the context to the local execution buffer."""
        # Add extra handling for string input
        if isinstance(overwrite, str):
            overwrite = overwrite.lower() == "true"
        if overwrite:
            self.code_context = [
                str(ele)
                for ele in PyInterpreter.DEFAULT_CODE_CONTEXT.split("\n")
            ]

        code = self._clean_markdown(code)
        status, result = self._attempt_execution(code)
        if status:
            self.code_context.extend(code.split("\n"))
        return status, result

# From builders/py_interpreter.py
def set_code_and_run_tests(self, code: str, overwrite: bool = True) -> str:
        """Set the code and then run the local tests"""
        status, result = self.set_code(code, overwrite)
        result = f"Code Exec Result:\n{result}"
        if status:
            result += "\n" + f"Test Exec Result:\n{self._run_tests()}"
        return result

# From builders/py_interpreter.py
def build(self) -> List[Tool]:
        """Builds the tools for the interpreter."""
        return [
            Tool(
                name="py-set-and-run-tests",
                function=self.py_interpreter.set_tests,
                description="Sets up the provided Python markdown snippet in the test environment and executes the tests and previously provided code from `py-set-code-and-run-tests`. Sympy and Numpy have already been imported as `sp` and `np`, respectively.\n\nAny provided code will be parsed and persisted across interactions. If `overwrite` is set to true then the existing test code is overwritten. The user must call `print(...)` on any output they would like to see returned from the environment. For instance, if your execution terminates with a variable x, to find out what x evaluates to you should end the code with `print(x)`.",
            ),
            Tool(
                name="py-set-code-and-run-tests",
                function=self.py_interpreter.set_code_and_run_tests,
                description="Sets up the provided Python markdown snippet in the local source environment. The code is executed and its context is persisted in the source environment across interactions. After successfully executing the provided code, the provided tests are then ran. If `overwrite` is set to true then existing source code environment is overwritten (but not the tests).",
            ),
        ]

# From builders/py_interpreter.py
def build_for_open_ai(self) -> List[OpenAITool]:
        """Builds the tools associated with the Python interpreter for the OpenAI API."""
        tools = super().build()

        # Predefined properties and required parameters
        properties = {
            "code": {
                "type": "string",
                "description": "The given Python code to execute, formatted as a markdown snippet, e.g. ```python\\n[CODE]``` and with newlines separated by the double-escaped newline char '\\n'.",
            },
            "overwrite": {
                "type": "string",
                "description": "Specifies whether or not the given code should overwrite the existing code in the interpreter.",
                "default": "True",
            },
        }
        required = ["code"]

        return [
            OpenAITool(
                function=tool.function,
                name=tool.name,
                description=tool.description,
                properties=properties,
                required=required,
            )
            for tool in tools
        ]

# From builders/py_interpreter.py
def handler(signum, frame) -> TimeoutError:
            raise TimeoutError("Execution timed out")

import queue
import re
import subprocess
import threading
import traceback
from subprocess_language import SubprocessLanguage

# From languages/java.py
class Java(SubprocessLanguage):
    file_extension = "java"
    name = "Java"

    def __init__(self):
        super().__init__()
        self.start_cmd = None  # We will handle the start command in the run method

    def preprocess_code(self, code):
        return preprocess_java(code)

    def line_postprocessor(self, line):
        # Clean up output from javac and java
        return line.strip()

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line

    def run(self, code):
        try:
            # Extract the class name from the code
            match = re.search(r'class\s+(\w+)', code)
            if not match:
                yield {
                    "type": "console",
                    "format": "output",
                    "content": "Error: No class definition found in the provided code."
                }
                return

            class_name = match.group(1)
            file_name = f"{class_name}.java"

            # Write the Java code to a file, preserving newlines
            with open(file_name, "w", newline='\n') as file:
                file.write(code)

            # Compile the Java code
            compile_process = subprocess.Popen(
                ["javac", file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = compile_process.communicate()

            if compile_process.returncode != 0:
                yield {
                    "type": "console",
                    "format": "output",
                    "content": f"Compilation Error:\n{stderr}"
                }
                return

            # Run the compiled Java code
            run_process = subprocess.Popen(
                ["java", class_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout_thread = threading.Thread(
                target=self.handle_stream_output,
                args=(run_process.stdout, False),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self.handle_stream_output,
                args=(run_process.stderr, True),
                daemon=True,
            )

            stdout_thread.start()
            stderr_thread.start()

            stdout_thread.join()
            stderr_thread.join()

            run_process.wait()
            self.done.set()

            while True:
                if not self.output_queue.empty():
                    yield self.output_queue.get()
                else:
                    time.sleep(0.1)
                try:
                    output = self.output_queue.get(timeout=0.3)
                    yield output
                except queue.Empty:
                    if self.done.is_set():
                        for _ in range(3):
                            if not self.output_queue.empty():
                                yield self.output_queue.get()
                            time.sleep(0.2)
                        break

        except Exception as e:
            yield {
                "type": "console",
                "format": "output",
                "content": f"{traceback.format_exc()}"
            }
        finally:
            # Clean up the generated Java files
            if os.path.exists(file_name):
                os.remove(file_name)
            class_file = file_name.replace(".java", ".class")
            if os.path.exists(class_file):
                os.remove(class_file)

# From languages/java.py
def preprocess_java(code):
    """
    Add active line markers
    Add end of execution marker
    """
    lines = code.split("\n")
    processed_lines = []

    for i, line in enumerate(lines, 1):
        # Add active line print
        processed_lines.append(f'System.out.println("##active_line{i}##");')
        processed_lines.append(line)

    # Join lines to form the processed code
    code = "\n".join(processed_lines)

    # Add end of execution marker
    code += '\nSystem.out.println("##end_of_execution##");'
    return code

# From languages/java.py
def preprocess_code(self, code):
        return preprocess_java(code)

# From languages/java.py
def line_postprocessor(self, line):
        # Clean up output from javac and java
        return line.strip()

# From languages/java.py
def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

# From languages/java.py
def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line

from base_language import BaseLanguage

# From languages/subprocess_language.py
class SubprocessLanguage(BaseLanguage):
    def __init__(self):
        self.start_cmd = []
        self.process = None
        self.verbose = False
        self.output_queue = queue.Queue()
        self.done = threading.Event()

    def detect_active_line(self, line):
        return None

    def detect_end_of_execution(self, line):
        return None

    def line_postprocessor(self, line):
        return line

    def preprocess_code(self, code):
        """
        This needs to insert an end_of_execution marker of some kind,
        which can be detected by detect_end_of_execution.

        Optionally, add active line markers for detect_active_line.
        """
        return code

    def terminate(self):
        if self.process:
            self.process.terminate()
            self.process.stdin.close()
            self.process.stdout.close()

    def start_process(self):
        if self.process:
            self.terminate()

        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        self.process = subprocess.Popen(
            self.start_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            universal_newlines=True,
            env=my_env,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stdout, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stderr, True),
            daemon=True,
        ).start()

    def run(self, code):
        retry_count = 0
        max_retries = 3

        # Setup
        try:
            code = self.preprocess_code(code)
            if not self.process:
                self.start_process()
        except:
            yield {
                "type": "console",
                "format": "output",
                "content": traceback.format_exc(),
            }
            return

        while retry_count <= max_retries:
            if self.verbose:
                print(f"(after processing) Running processed code:\n{code}\n---")

            self.done.clear()

            try:
                self.process.stdin.write(code + "\n")
                self.process.stdin.flush()
                break
            except:
                if retry_count != 0:
                    # For UX, I like to hide this if it happens once. Obviously feels better to not see errors
                    # Most of the time it doesn't matter, but we should figure out why it happens frequently with:
                    # applescript
                    yield {
                        "type": "console",
                        "format": "output",
                        "content": f"{traceback.format_exc()}\nRetrying... ({retry_count}/{max_retries})\nRestarting process.",
                    }

                self.start_process()

                retry_count += 1
                if retry_count > max_retries:
                    yield {
                        "type": "console",
                        "format": "output",
                        "content": "Maximum retries reached. Could not execute code.",
                    }
                    return

        while True:
            if not self.output_queue.empty():
                yield self.output_queue.get()
            else:
                time.sleep(0.1)
            try:
                output = self.output_queue.get(timeout=0.3)  # Waits for 0.3 seconds
                yield output
            except queue.Empty:
                if self.done.is_set():
                    # Try to yank 3 more times from it... maybe there's something in there...
                    # (I don't know if this actually helps. Maybe we just need to yank 1 more time)
                    for _ in range(3):
                        if not self.output_queue.empty():
                            yield self.output_queue.get()
                        time.sleep(0.2)
                    break

    def handle_stream_output(self, stream, is_error_stream):
        try:
            for line in iter(stream.readline, ""):
                if self.verbose:
                    print(f"Received output line:\n{line}\n---")

                line = self.line_postprocessor(line)

                if line is None:
                    continue  # `line = None` is the postprocessor's signal to discard completely

                if self.detect_active_line(line):
                    active_line = self.detect_active_line(line)
                    self.output_queue.put(
                        {
                            "type": "console",
                            "format": "active_line",
                            "content": active_line,
                        }
                    )
                    # Sometimes there's a little extra on the same line, so be sure to send that out
                    line = re.sub(r"##active_line\d+##", "", line)
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                elif self.detect_end_of_execution(line):
                    # Sometimes there's a little extra on the same line, so be sure to send that out
                    line = line.replace("##end_of_execution##", "").strip()
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                    self.done.set()
                elif is_error_stream and "KeyboardInterrupt" in line:
                    self.output_queue.put(
                        {
                            "type": "console",
                            "format": "output",
                            "content": "KeyboardInterrupt",
                        }
                    )
                    time.sleep(0.1)
                    self.done.set()
                else:
                    self.output_queue.put(
                        {"type": "console", "format": "output", "content": line}
                    )
        except ValueError as e:
            if "operation on closed file" in str(e):
                if self.verbose:
                    print("Stream closed while reading.")
            else:
                raise e

# From languages/subprocess_language.py
def terminate(self):
        if self.process:
            self.process.terminate()
            self.process.stdin.close()
            self.process.stdout.close()

# From languages/subprocess_language.py
def start_process(self):
        if self.process:
            self.terminate()

        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        self.process = subprocess.Popen(
            self.start_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            universal_newlines=True,
            env=my_env,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stdout, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self.handle_stream_output,
            args=(self.process.stderr, True),
            daemon=True,
        ).start()

# From languages/subprocess_language.py
def handle_stream_output(self, stream, is_error_stream):
        try:
            for line in iter(stream.readline, ""):
                if self.verbose:
                    print(f"Received output line:\n{line}\n---")

                line = self.line_postprocessor(line)

                if line is None:
                    continue  # `line = None` is the postprocessor's signal to discard completely

                if self.detect_active_line(line):
                    active_line = self.detect_active_line(line)
                    self.output_queue.put(
                        {
                            "type": "console",
                            "format": "active_line",
                            "content": active_line,
                        }
                    )
                    # Sometimes there's a little extra on the same line, so be sure to send that out
                    line = re.sub(r"##active_line\d+##", "", line)
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                elif self.detect_end_of_execution(line):
                    # Sometimes there's a little extra on the same line, so be sure to send that out
                    line = line.replace("##end_of_execution##", "").strip()
                    if line:
                        self.output_queue.put(
                            {"type": "console", "format": "output", "content": line}
                        )
                    self.done.set()
                elif is_error_stream and "KeyboardInterrupt" in line:
                    self.output_queue.put(
                        {
                            "type": "console",
                            "format": "output",
                            "content": "KeyboardInterrupt",
                        }
                    )
                    time.sleep(0.1)
                    self.done.set()
                else:
                    self.output_queue.put(
                        {"type": "console", "format": "output", "content": line}
                    )
        except ValueError as e:
            if "operation on closed file" in str(e):
                if self.verbose:
                    print("Stream closed while reading.")
            else:
                raise e


# From languages/applescript.py
class AppleScript(SubprocessLanguage):
    file_extension = "applescript"
    name = "AppleScript"

    def __init__(self):
        super().__init__()
        self.start_cmd = [os.environ.get("SHELL", "/bin/zsh")]

    def preprocess_code(self, code):
        """
        Inserts an end_of_execution marker and adds active line indicators.
        """
        # Add active line indicators to the code
        code = self.add_active_line_indicators(code)

        # Escape double quotes
        code = code.replace('"', r"\"")

        # Wrap in double quotes
        code = '"' + code + '"'

        # Prepend start command for AppleScript
        code = "osascript -e " + code

        # Append end of execution indicator
        code += '; echo "##end_of_execution##"'

        return code

    def add_active_line_indicators(self, code):
        """
        Adds log commands to indicate the active line of execution in the AppleScript.
        """
        modified_lines = []
        lines = code.split("\n")

        for idx, line in enumerate(lines):
            # Add log command to indicate the line number
            if line.strip():  # Only add if line is not empty
                modified_lines.append(f'log "##active_line{idx + 1}##"')
            modified_lines.append(line)

        return "\n".join(modified_lines)

    def detect_active_line(self, line):
        """
        Detects active line indicator in the output.
        """
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        """
        Detects end of execution marker in the output.
        """
        return "##end_of_execution##" in line

# From languages/applescript.py
def add_active_line_indicators(self, code):
        """
        Adds log commands to indicate the active line of execution in the AppleScript.
        """
        modified_lines = []
        lines = code.split("\n")

        for idx, line in enumerate(lines):
            # Add log command to indicate the line number
            if line.strip():  # Only add if line is not empty
                modified_lines.append(f'log "##active_line{idx + 1}##"')
            modified_lines.append(line)

        return "\n".join(modified_lines)


# From languages/javascript.py
class JavaScript(SubprocessLanguage):
    file_extension = "js"
    name = "JavaScript"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["node", "-i"]

    def preprocess_code(self, code):
        return preprocess_javascript(code)

    def line_postprocessor(self, line):
        # Node's interactive REPL outputs a billion things
        # So we clean it up:
        if "Welcome to Node.js" in line:
            return None
        if line.strip() in ["undefined", 'Type ".help" for more information.']:
            return None
        line = line.strip(". \n")
        # Remove trailing ">"s
        line = re.sub(r"^\s*(>\s*)+", "", line)
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line

# From languages/javascript.py
def preprocess_javascript(code):
    """
    Add active line markers
    Wrap in a try catch
    Add end of execution marker
    """

    # Detect if nothing in the code is multiline. (This is waaaay to false-positive-y but it works)
    nothing_multiline = not any(char in code for char in ["{", "}", "[", "]"])

    if nothing_multiline:
        # Split code into lines
        lines = code.split("\n")
        processed_lines = []
        for i, line in enumerate(lines, 1):
            # Add active line print
            processed_lines.append(f'console.log("##active_line{i}##");')
            processed_lines.append(line)

        # Join lines to form the processed code
        code = "\n".join(processed_lines)

    # Wrap in a try-catch and add end of execution marker
    code = f"""
try {{
{code}
}} catch (e) {{
    console.log(e);
}}
console.log("##end_of_execution##");
"""

    return code

from pathlib import Path

# From languages/ruby.py
class Ruby(SubprocessLanguage):
    file_extension = "rb"
    name = "Ruby"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["irb"] 

    def preprocess_code(self, code):
        """
        Add active line markers
        Wrap in a tryCatch for better error handling 
        Add end of execution marker
        """

        lines = code.split("\n")
        processed_lines = []

        for i, line in enumerate(lines, 1):
            # Add active line print
            processed_lines.append(f'puts "##active_line{i}##"')
            processed_lines.append(line)
        # Join lines to form the processed code
        processed_code = "\n".join(processed_lines)

        # Wrap in a tryCatch for error handling and add end of execution marker
        processed_code = f"""
begin
  {processed_code}
rescue => e
  puts "##execution_error##\\n" + e.message
ensure
  puts "##end_of_execution##\\n"
end
"""
        self.code_line_count = len(processed_code.split("\n"))
        #print(processed_code)
        return processed_code

    def line_postprocessor(self, line):
        # If the line count attribute is set and non-zero, decrement and skip the line
        if hasattr(self, "code_line_count") and self.code_line_count > 0:
            self.code_line_count -= 1
            return None
        if "nil" in line:
           return None
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line or "##execution_error##" in line

import platform

# From languages/powershell.py
class PowerShell(SubprocessLanguage):
    file_extension = "ps1"
    name = "PowerShell"

    def __init__(self):
        super().__init__()

        # Determine the start command based on the platform (use "powershell" for Windows)
        if platform.system() == "Windows":
            self.start_cmd = ["powershell.exe"]
            # self.start_cmd = os.environ.get('SHELL', 'powershell.exe')
        else:
            # On non-Windows platforms, prefer pwsh (PowerShell Core) if available, or fall back to bash
            self.start_cmd = ["pwsh"] if shutil.which("pwsh") else ["bash"]

    def preprocess_code(self, code):
        return preprocess_powershell(code)

    def line_postprocessor(self, line):
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line

# From languages/powershell.py
def preprocess_powershell(code):
    """
    Add active line markers
    Wrap in try-catch block
    Add end of execution marker
    """
    # Add commands that tell us what the active line is
    code = add_active_line_prints(code)

    # Wrap in try-catch block for error handling
    code = wrap_in_try_catch(code)

    # Add end marker (we'll be listening for this to know when it ends)
    code += '\nWrite-Output "##end_of_execution##"'

    return code

# From languages/powershell.py
def add_active_line_prints(code):
    """
    Add Write-Output statements indicating line numbers to a PowerShell script.
    """
    lines = code.split("\n")
    for index, line in enumerate(lines):
        # Insert the Write-Output command before the actual line
        lines[index] = f'Write-Output "##active_line{index + 1}##"\n{line}'
    return "\n".join(lines)

# From languages/powershell.py
def wrap_in_try_catch(code):
    """
    Wrap PowerShell code in a try-catch block to catch errors and display them.
    """
    try_catch_code = """
try {
    $ErrorActionPreference = "Stop"
"""
    return try_catch_code + code + "\n} catch {\n    Write-Error $_\n}\n"


# From languages/shell.py
class Shell(SubprocessLanguage):
    file_extension = "sh"
    name = "Shell"
    aliases = ["bash", "sh", "zsh", "batch", "bat"]

    def __init__(
        self,
    ):
        super().__init__()

        # Determine the start command based on the platform
        if platform.system() == "Windows":
            self.start_cmd = ["cmd.exe"]
        else:
            self.start_cmd = [os.environ.get("SHELL", "bash")]

    def preprocess_code(self, code):
        return preprocess_shell(code)

    def line_postprocessor(self, line):
        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line

# From languages/shell.py
def preprocess_shell(code):
    """
    Add active line markers
    Wrap in a try except (trap in shell)
    Add end of execution marker
    """

    # Add commands that tell us what the active line is
    # if it's multiline, just skip this. soon we should make it work with multiline
    if (
        not has_multiline_commands(code)
        and os.environ.get("INTERPRETER_ACTIVE_LINE_DETECTION", "True").lower()
        == "true"
    ):
        code = add_active_line_prints(code)

    # Add end command (we'll be listening for this so we know when it ends)
    code += '\necho "##end_of_execution##"'

    return code

# From languages/shell.py
def has_multiline_commands(script_text):
    # Patterns that indicate a line continues
    continuation_patterns = [
        r"\\$",  # Line continuation character at the end of the line
        r"\|$",  # Pipe character at the end of the line indicating a pipeline continuation
        r"&&\s*$",  # Logical AND at the end of the line
        r"\|\|\s*$",  # Logical OR at the end of the line
        r"<\($",  # Start of process substitution
        r"\($",  # Start of subshell
        r"{\s*$",  # Start of a block
        r"\bif\b",  # Start of an if statement
        r"\bwhile\b",  # Start of a while loop
        r"\bfor\b",  # Start of a for loop
        r"do\s*$",  # 'do' keyword for loops
        r"then\s*$",  # 'then' keyword for if statements
    ]

    # Check each line for multiline patterns
    for line in script_text.splitlines():
        if any(re.search(pattern, line.rstrip()) for pattern in continuation_patterns):
            return True

    return False


# From languages/r.py
class R(SubprocessLanguage):
    file_extension = "r"
    name = "R"

    def __init__(self):
        super().__init__()
        self.start_cmd = ["R", "-q", "--vanilla"]  # Start R in quiet and vanilla mode

    def preprocess_code(self, code):
        """
        Add active line markers
        Wrap in a tryCatch for better error handling in R
        Add end of execution marker
        """

        lines = code.split("\n")
        processed_lines = []

        for i, line in enumerate(lines, 1):
            # Add active line print
            processed_lines.append(f'cat("##active_line{i}##\\n");{line}')

        # Join lines to form the processed code
        processed_code = "\n".join(processed_lines)

        # Wrap in a tryCatch for error handling and add end of execution marker
        processed_code = f"""
tryCatch({{
{processed_code}
}}, error=function(e){{
    cat("##execution_error##\\n", conditionMessage(e), "\\n");
}})
cat("##end_of_execution##\\n");
"""
        # Count the number of lines of processed_code
        # (R echoes all code back for some reason, but we can skip it if we track this!)
        self.code_line_count = len(processed_code.split("\n")) - 1

        return processed_code

    def line_postprocessor(self, line):
        # If the line count attribute is set and non-zero, decrement and skip the line
        if hasattr(self, "code_line_count") and self.code_line_count > 0:
            self.code_line_count -= 1
            return None

        if re.match(r"^(\s*>>>\s*|\s*\.\.\.\s*|\s*>\s*|\s*\+\s*|\s*)$", line):
            return None
        if "R version" in line:  # Startup message
            return None
        if line.strip().startswith('[1] "') and line.endswith(
            '"'
        ):  # For strings, trim quotation marks
            return line[5:-1].strip()
        if line.strip().startswith(
            "[1]"
        ):  # Normal R output prefix for non-string outputs
            return line[4:].strip()

        return line

    def detect_active_line(self, line):
        if "##active_line" in line:
            return int(line.split("##active_line")[1].split("##")[0])
        return None

    def detect_end_of_execution(self, line):
        return "##end_of_execution##" in line or "##execution_error##" in line

