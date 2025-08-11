# Merged file for agent_frameworks/execution
# This file contains code merged from multiple repositories

from dataclasses import dataclass
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import Annotated
from  import FunctionCall
from  import Image

# From models/_types.py
class SystemMessage(BaseModel):
    """System message contains instructions for the model coming from the developer.

    .. note::

        Open AI is moving away from using 'system' role in favor of 'developer' role.
        See `Model Spec <https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions>`_ for more details.
        However, the 'system' role is still allowed in their API and will be automatically converted to 'developer' role
        on the server side.
        So, you can use `SystemMessage` for developer messages.

    """

    content: str
    """The content of the message."""

    type: Literal["SystemMessage"] = "SystemMessage"

# From models/_types.py
class UserMessage(BaseModel):
    """User message contains input from end users, or a catch-all for data provided to the model."""

    content: Union[str, List[Union[str, Image]]]
    """The content of the message."""

    source: str
    """The name of the agent that sent this message."""

    type: Literal["UserMessage"] = "UserMessage"

# From models/_types.py
class AssistantMessage(BaseModel):
    """Assistant message are sampled from the language model."""

    content: Union[str, List[FunctionCall]]
    """The content of the message."""

    thought: str | None = None
    """The reasoning text for the completion if available. Used for reasoning model and additional text content besides function calls."""

    source: str
    """The name of the agent that sent this message."""

    type: Literal["AssistantMessage"] = "AssistantMessage"

# From models/_types.py
class FunctionExecutionResult(BaseModel):
    """Function execution result contains the output of a function call."""

    content: str
    """The output of the function call."""

    name: str
    """(New in v0.4.8) The name of the function that was called."""

    call_id: str
    """The ID of the function call. Note this ID may be empty for some models."""

    is_error: bool | None = None
    """Whether the function call resulted in an error."""

# From models/_types.py
class FunctionExecutionResultMessage(BaseModel):
    """Function execution result message contains the output of multiple function calls."""

    content: List[FunctionExecutionResult]

    type: Literal["FunctionExecutionResultMessage"] = "FunctionExecutionResultMessage"

# From models/_types.py
class RequestUsage:
    prompt_tokens: int
    completion_tokens: int

# From models/_types.py
class TopLogprob:
    logprob: float
    bytes: Optional[List[int]] = None

# From models/_types.py
class ChatCompletionTokenLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List[TopLogprob] | None] = None
    bytes: Optional[List[int]] = None

# From models/_types.py
class CreateResult(BaseModel):
    """Create result contains the output of a model completion."""

    finish_reason: FinishReasons
    """The reason the model finished generating the completion."""

    content: Union[str, List[FunctionCall]]
    """The output of the model completion."""

    usage: RequestUsage
    """The usage of tokens in the prompt and completion."""

    cached: bool
    """Whether the completion was generated from a cached response."""

    logprobs: Optional[List[ChatCompletionTokenLogprob] | None] = None
    """The logprobs of the tokens in the completion."""

    thought: Optional[str] = None
    """The reasoning text for the completion if available. Used for reasoning models
    and additional text content besides function calls."""

import asyncio
from collections import Counter
from collections import deque
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Set
from autogen_core import AgentRuntime
from autogen_core import Component
from autogen_core import ComponentModel
from pydantic import model_validator
from typing_extensions import Self
from autogen_agentchat.base import ChatAgent
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.messages import BaseAgentEvent
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.messages import MessageFactory
from autogen_agentchat.messages import StopMessage
from autogen_agentchat.state import BaseGroupChatManagerState
from autogen_agentchat.teams import BaseGroupChat
from _group_chat._base_group_chat_manager import BaseGroupChatManager
from _group_chat._events import GroupChatTermination

# From _graph/_digraph_group_chat.py
class DiGraphEdge(BaseModel):
    """Represents a directed edge in a :class:`DiGraph`, with an optional execution condition.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    .. warning::

        If the condition is a callable, it will not be serialized in the model.

    """

    target: str  # Target node name
    condition: Union[str, Callable[[BaseChatMessage], bool], None] = Field(default=None)
    """(Experimental) Condition to execute this edge.
    If None, the edge is unconditional.
    If a string, the edge is conditional on the presence of that string in the last agent chat message.
    If a callable, the edge is conditional on the callable returning True when given the last message.
    """

    # Using Field to exclude the condition in serialization if it's a callable
    condition_function: Callable[[BaseChatMessage], bool] | None = Field(default=None, exclude=True)
    activation_group: str = Field(default="")
    """Group identifier for forward dependencies.

    When multiple edges point to the same target node, they are grouped by this field.
    This allows distinguishing between different cycles or dependency patterns.

    Example: In a graph containing a cycle like A->B->C->B, the two edges pointing to B (A->B and C->B)
    can be in different activation groups to control how B is activated.
    Defaults to the target node name if not specified.
    """
    activation_condition: Literal["all", "any"] = "all"
    """Determines how forward dependencies within the same activation_group are evaluated.

    - "all": All edges in this activation group must be satisfied before the target node can execute
    - "any": Any single edge in this activation group being satisfied allows the target node to execute

    This is used to handle complex dependency patterns in cyclic graphs where multiple
    paths can lead to the same target node.
    """

    @model_validator(mode="after")
    def _validate_condition(self) -> "DiGraphEdge":
        # Store callable in a separate field and set condition to None for serialization
        if callable(self.condition):
            self.condition_function = self.condition
            # For serialization purposes, we'll set the condition to None
            # when storing as a pydantic model/dict
            object.__setattr__(self, "condition", None)

        # Set activation_group to target if not already set
        if not self.activation_group:
            self.activation_group = self.target

        return self

    def check_condition(self, message: BaseChatMessage) -> bool:
        """Check if the edge condition is satisfied for the given message.

        Args:
            message: The message to check the condition against.

        Returns:
            True if condition is satisfied (None condition always returns True),
            False otherwise.
        """
        if self.condition_function is not None:
            return self.condition_function(message)
        elif isinstance(self.condition, str):
            # If it's a string, check if the string is in the message content
            return self.condition in message.to_model_text()
        return True

# From _graph/_digraph_group_chat.py
class DiGraphNode(BaseModel):
    """Represents a node (agent) in a :class:`DiGraph`, with its outgoing edges and activation type.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    """

    name: str  # Agent's name
    edges: List[DiGraphEdge] = []  # Outgoing edges
    activation: Literal["all", "any"] = "all"

# From _graph/_digraph_group_chat.py
class DiGraph(BaseModel):
    """Defines a directed graph structure with nodes and edges.
    :class:`GraphFlow` uses this to determine execution order and conditions.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    """

    nodes: Dict[str, DiGraphNode]  # Node name → DiGraphNode mapping
    default_start_node: str | None = None  # Default start node name
    _has_cycles: bool | None = None  # Cyclic graph flag

    def get_parents(self) -> Dict[str, List[str]]:
        """Compute a mapping of each node to its parent nodes."""
        parents: Dict[str, List[str]] = {node: [] for node in self.nodes}
        for node in self.nodes.values():
            for edge in node.edges:
                parents[edge.target].append(node.name)
        return parents

    def get_start_nodes(self) -> Set[str]:
        """Return the nodes that have no incoming edges (entry points)."""
        if self.default_start_node:
            return {self.default_start_node}

        parents = self.get_parents()
        return set([node_name for node_name, parent_list in parents.items() if not parent_list])

    def get_leaf_nodes(self) -> Set[str]:
        """Return nodes that have no outgoing edges (final output nodes)."""
        return set([name for name, node in self.nodes.items() if not node.edges])

    def has_cycles_with_exit(self) -> bool:
        """
        Check if the graph has any cycles and validate that each cycle has at least one conditional edge.

        Returns:
            bool: True if there is at least one cycle and all cycles have an exit condition.
                False if there are no cycles.

        Raises:
            ValueError: If there is a cycle without any conditional edge.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            for edge in self.nodes[node_name].edges:
                target = edge.target
                if target not in visited:
                    if dfs(target):
                        return True
                elif target in rec_stack:
                    # Found a cycle → extract the cycle
                    cycle_start_index = path.index(target)
                    cycle_nodes = path[cycle_start_index:]
                    cycle_edges: List[DiGraphEdge] = []
                    for n in cycle_nodes:
                        cycle_edges.extend(self.nodes[n].edges)
                    if all(edge.condition is None and edge.condition_function is None for edge in cycle_edges):
                        raise ValueError(
                            f"Cycle detected without exit condition: {' -> '.join(cycle_nodes + cycle_nodes[:1])}"
                        )
                    return True  # Found cycle, but it has an exit condition

            rec_stack.remove(node_name)
            path.pop()
            return False

        has_cycle = False
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    has_cycle = True

        return has_cycle

    def get_has_cycles(self) -> bool:
        """Indicates if the graph has at least one cycle (with valid exit conditions)."""
        if self._has_cycles is None:
            self._has_cycles = self.has_cycles_with_exit()

        return self._has_cycles

    def graph_validate(self) -> None:
        """Validate graph structure and execution rules."""
        if not self.nodes:
            raise ValueError("Graph has no nodes.")

        if not self.get_start_nodes():
            raise ValueError("Graph must have at least one start node")

        if not self.get_leaf_nodes():
            raise ValueError("Graph must have at least one leaf node")

        # Outgoing edge condition validation (per node)
        for node in self.nodes.values():
            # Check that if a node has an outgoing conditional edge, then all outgoing edges are conditional
            has_condition = any(
                edge.condition is not None or edge.condition_function is not None for edge in node.edges
            )
            has_unconditioned = any(edge.condition is None and edge.condition_function is None for edge in node.edges)
            if has_condition and has_unconditioned:
                raise ValueError(f"Node '{node.name}' has a mix of conditional and unconditional edges.")

        # Validate activation conditions across all edges in the graph
        self._validate_activation_conditions()

        self._has_cycles = self.has_cycles_with_exit()

    def _validate_activation_conditions(self) -> None:
        """Validate that all edges pointing to the same target node have consistent activation_condition values.

        Raises:
            ValueError: If edges pointing to the same target have different activation_condition values
        """
        target_activation_conditions: Dict[str, Dict[str, str]] = {}  # target_node -> {activation_group -> condition}

        for node in self.nodes.values():
            for edge in node.edges:
                target = edge.target  # The target node this edge points to
                activation_group = edge.activation_group

                if target not in target_activation_conditions:
                    target_activation_conditions[target] = {}

                if activation_group in target_activation_conditions[target]:
                    if target_activation_conditions[target][activation_group] != edge.activation_condition:
                        # Find the source node that has the conflicting condition
                        conflicting_source = self._find_edge_source_by_target_and_group(
                            target, activation_group, target_activation_conditions[target][activation_group]
                        )
                        raise ValueError(
                            f"Conflicting activation conditions for target '{target}' group '{activation_group}': "
                            f"'{target_activation_conditions[target][activation_group]}' (from node '{conflicting_source}') "
                            f"and '{edge.activation_condition}' (from node '{node.name}')"
                        )
                else:
                    target_activation_conditions[target][activation_group] = edge.activation_condition

    def _find_edge_source_by_target_and_group(
        self, target: str, activation_group: str, activation_condition: str
    ) -> str:
        """Find the source node that has an edge pointing to the given target with the given activation_group and activation_condition."""
        for node_name, node in self.nodes.items():
            for edge in node.edges:
                if (
                    edge.target == target
                    and edge.activation_group == activation_group
                    and edge.activation_condition == activation_condition
                ):
                    return node_name
        return "unknown"

    def get_remaining_map(self) -> Dict[str, Dict[str, int]]:
        """Get the remaining map that tracks how many edges point to each target node with each activation group.

        Returns:
            Dictionary mapping target nodes to their activation groups and remaining counts
        """

        remaining_map: Dict[str, Dict[str, int]] = {}

        for node in self.nodes.values():
            for edge in node.edges:
                target = edge.target
                activation_group = edge.activation_group

                if target not in remaining_map:
                    remaining_map[target] = {}

                if activation_group not in remaining_map[target]:
                    remaining_map[target][activation_group] = 0

                remaining_map[target][activation_group] += 1

        return remaining_map

# From _graph/_digraph_group_chat.py
class GraphFlowManagerState(BaseGroupChatManagerState):
    """Tracks active execution state for DAG-based execution."""

    active_nodes: List[str] = []  # Currently executing nodes
    type: str = "GraphManagerState"

# From _graph/_digraph_group_chat.py
class GraphFlowManager(BaseGroupChatManager):
    """Manages execution of agents using a Directed Graph execution model."""

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
        graph: DiGraph,
    ) -> None:
        """Initialize the graph-based execution manager."""
        super().__init__(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
            max_turns=max_turns,
            message_factory=message_factory,
        )
        graph.graph_validate()
        if graph.get_has_cycles() and self._termination_condition is None and self._max_turns is None:
            raise ValueError("A termination condition is required for cyclic graphs without a maximum turn limit.")
        self._graph = graph
        # Lookup table for incoming edges for each node.
        self._parents = graph.get_parents()
        # Lookup table for outgoing edges for each node.
        self._edges: Dict[str, List[DiGraphEdge]] = {n: node.edges for n, node in graph.nodes.items()}

        # Build activation and enqueued_any lookup tables by collecting all edges and grouping by target node
        self._build_lookup_tables(graph)

        # Track which activation groups were triggered for each node
        self._triggered_activation_groups: Dict[str, Set[str]] = {}
        # === Mutable states for the graph execution ===
        # Count the number of remaining parents to activate each node.
        self._remaining: Dict[str, Counter[str]] = {
            target: Counter(groups) for target, groups in graph.get_remaining_map().items()
        }
        # cache for remaining
        self._origin_remaining: Dict[str, Dict[str, int]] = {
            target: Counter(groups) for target, groups in self._remaining.items()
        }

        # Ready queue for nodes that are ready to execute, starting with the start nodes.
        self._ready: Deque[str] = deque([n for n in graph.get_start_nodes()])

    def _build_lookup_tables(self, graph: DiGraph) -> None:
        """Build activation and enqueued_any lookup tables by collecting all edges and grouping by target node.

        Args:
            graph: The directed graph
        """
        self._activation: Dict[str, Dict[str, Literal["any", "all"]]] = {}
        self._enqueued_any: Dict[str, Dict[str, bool]] = {}

        for node in graph.nodes.values():
            for edge in node.edges:
                target = edge.target
                activation_group = edge.activation_group

                # Build activation lookup
                if target not in self._activation:
                    self._activation[target] = {}
                if activation_group not in self._activation[target]:
                    self._activation[target][activation_group] = edge.activation_condition

                # Build enqueued_any lookup
                if target not in self._enqueued_any:
                    self._enqueued_any[target] = {}
                if activation_group not in self._enqueued_any[target]:
                    self._enqueued_any[target][activation_group] = False

    async def update_message_thread(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> None:
        await super().update_message_thread(messages)

        # Find the node that ran in the current turn.
        message = messages[-1]
        if message.source not in self._graph.nodes:
            # Ignore messages from sources outside of the graph.
            return
        assert isinstance(message, BaseChatMessage)
        source = message.source

        # Propagate the update to the children of the node.
        for edge in self._edges[source]:
            # Use the new check_condition method that handles both string and callable conditions
            if not edge.check_condition(message):
                continue

            target = edge.target
            activation_group = edge.activation_group

            if self._activation[target][activation_group] == "all":
                self._remaining[target][activation_group] -= 1
                if self._remaining[target][activation_group] == 0:
                    # If all parents are done, add to the ready queue.
                    self._ready.append(target)
                    # Track which activation group was triggered
                    self._save_triggered_activation_group(target, activation_group)
            else:
                # If activation is any, add to the ready queue if not already enqueued.
                if not self._enqueued_any[target][activation_group]:
                    self._ready.append(target)
                    self._enqueued_any[target][activation_group] = True
                    # Track which activation group was triggered
                    self._save_triggered_activation_group(target, activation_group)

    def _save_triggered_activation_group(self, target: str, activation_group: str) -> None:
        """Save which activation group was triggered for a target node.

        Args:
            target: The target node that was triggered
            activation_group: The activation group that caused the trigger
        """
        if target not in self._triggered_activation_groups:
            self._triggered_activation_groups[target] = set()
        self._triggered_activation_groups[target].add(activation_group)

    def _reset_triggered_activation_groups(self, speaker: str) -> None:
        """Reset the bookkeeping for the specific activation groups that were triggered for a speaker.

        Args:
            speaker: The speaker node to reset activation groups for
        """
        if speaker not in self._triggered_activation_groups:
            return

        for activation_group in self._triggered_activation_groups[speaker]:
            if self._activation[speaker][activation_group] == "any":
                self._enqueued_any[speaker][activation_group] = False
            else:
                # Reset the remaining count for this activation group using the graph's original count
                if speaker in self._remaining and activation_group in self._remaining[speaker]:
                    self._remaining[speaker][activation_group] = self._origin_remaining[speaker][activation_group]

        # Clear the triggered activation groups for this speaker
        self._triggered_activation_groups[speaker].clear()

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
        # Drain the ready queue for the next set of speakers.
        speakers: List[str] = []
        while self._ready:
            speaker = self._ready.popleft()
            speakers.append(speaker)

            # Reset the bookkeeping for the specific activation groups that were triggered
            self._reset_triggered_activation_groups(speaker)

        return speakers

    async def validate_group_state(self, messages: List[BaseChatMessage] | None) -> None:
        pass

    async def _apply_termination_condition(
        self, delta: Sequence[BaseAgentEvent | BaseChatMessage], increment_turn_count: bool = False
    ) -> bool:
        """Apply termination condition including graph-specific completion logic.

        First checks if graph execution is complete, then checks standard termination conditions.

        Args:
            delta: The message delta to check termination conditions against
            increment_turn_count: Whether to increment the turn count

        Returns:
            True if the conversation should be terminated, False otherwise
        """
        # Check if the graph execution is complete (no ready speakers) - prioritize this check
        if not self._ready:
            stop_message = StopMessage(
                content=_DIGRAPH_STOP_MESSAGE,
                source=self._name,
            )
            # Reset the execution state when the graph has naturally completed
            self._reset_execution_state()
            # Reset the termination conditions and turn count.
            if self._termination_condition is not None:
                await self._termination_condition.reset()
            self._current_turn = 0
            # Signal termination to the caller of the team.
            await self._signal_termination(stop_message)
            return True

        # Apply the standard termination conditions from the base class
        return await super()._apply_termination_condition(delta, increment_turn_count)

    def _reset_execution_state(self) -> None:
        """Reset the graph execution state to the initial state."""
        self._remaining = {target: Counter(groups) for target, groups in self._graph.get_remaining_map().items()}
        self._enqueued_any = {n: {g: False for g in self._enqueued_any[n]} for n in self._enqueued_any}
        self._ready = deque([n for n in self._graph.get_start_nodes()])

    async def save_state(self) -> Mapping[str, Any]:
        """Save the execution state."""
        state = {
            "message_thread": [message.dump() for message in self._message_thread],
            "current_turn": self._current_turn,
            "remaining": {target: dict(counter) for target, counter in self._remaining.items()},
            "enqueued_any": dict(self._enqueued_any),
            "ready": list(self._ready),
        }
        return state

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore execution state from saved data."""
        self._message_thread = [self._message_factory.create(msg) for msg in state["message_thread"]]
        self._current_turn = state["current_turn"]
        self._remaining = {target: Counter(groups) for target, groups in state["remaining"].items()}
        self._enqueued_any = state["enqueued_any"]
        self._ready = deque(state["ready"])

    async def reset(self) -> None:
        """Reset execution state to the start of the graph."""
        self._current_turn = 0
        self._message_thread.clear()
        if self._termination_condition:
            await self._termination_condition.reset()
        self._reset_execution_state()

# From _graph/_digraph_group_chat.py
class GraphFlowConfig(BaseModel):
    """The declarative configuration for GraphFlow."""

    name: str | None = None
    description: str | None = None
    participants: List[ComponentModel]
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    graph: DiGraph

# From _graph/_digraph_group_chat.py
class GraphFlow(BaseGroupChat, Component[GraphFlowConfig]):
    """A team that runs a group chat following a Directed Graph execution pattern.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    This group chat executes agents based on a directed graph (:class:`DiGraph`) structure,
    allowing complex workflows such as sequential execution, parallel fan-out,
    conditional branching, join patterns, and loops with explicit exit conditions.

    The execution order is determined by the edges defined in the `DiGraph`. Each node
    in the graph corresponds to an agent, and edges define the flow of messages between agents.
    Nodes can be configured to activate when:

        - **All** parent nodes have completed (activation="all") → default
        - **Any** parent node completes (activation="any")

    Conditional branching is supported using edge conditions, where the next agent(s) are selected
    based on content in the chat history. Loops are permitted as long as there is a condition
    that eventually exits the loop.

    .. note::

        Use the :class:`DiGraphBuilder` class to create a :class:`DiGraph` easily. It provides a fluent API
        for adding nodes and edges, setting entry points, and validating the graph structure.
        See the :class:`DiGraphBuilder` documentation for more details.
        The :class:`GraphFlow` class is designed to be used with the :class:`DiGraphBuilder` for creating complex workflows.

    .. warning::

        When using callable conditions in edges, they will not be serialized
        when calling :meth:`dump_component`. This will be addressed in future releases.


    Args:
        participants (List[ChatAgent]): The participants in the group chat.
        termination_condition (TerminationCondition, optional): Termination condition for the chat.
        max_turns (int, optional): Maximum number of turns before forcing termination.
        graph (DiGraph): Directed execution graph defining node flow and conditions.

    Raises:
        ValueError: If participant names are not unique, or if graph validation fails (e.g., cycles without exit).

    Examples:

        **Sequential Flow: A → B → C**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                agent_a = AssistantAgent("A", model_client=model_client, system_message="You are a helpful assistant.")
                agent_b = AssistantAgent("B", model_client=model_client, system_message="Translate input to Chinese.")
                agent_c = AssistantAgent("C", model_client=model_client, system_message="Translate input to English.")

                # Create a directed graph with sequential flow A -> B -> C.
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                builder.add_edge(agent_a, agent_b).add_edge(agent_b, agent_c)
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(5),
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="Write a short story about a cat."):
                    print(event)


            asyncio.run(main())

        **Parallel Fan-out: A → (B, C)**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                agent_a = AssistantAgent("A", model_client=model_client, system_message="You are a helpful assistant.")
                agent_b = AssistantAgent("B", model_client=model_client, system_message="Translate input to Chinese.")
                agent_c = AssistantAgent("C", model_client=model_client, system_message="Translate input to Japanese.")

                # Create a directed graph with fan-out flow A -> (B, C).
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                builder.add_edge(agent_a, agent_b).add_edge(agent_a, agent_c)
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(5),
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="Write a short story about a cat."):
                    print(event)


            asyncio.run(main())

        **Conditional Branching: A → B (if 'yes') or C (otherwise)**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                agent_a = AssistantAgent(
                    "A",
                    model_client=model_client,
                    system_message="Detect if the input is in Chinese. If it is, say 'yes', else say 'no', and nothing else.",
                )
                agent_b = AssistantAgent("B", model_client=model_client, system_message="Translate input to English.")
                agent_c = AssistantAgent("C", model_client=model_client, system_message="Translate input to Chinese.")

                # Create a directed graph with conditional branching flow A -> B ("yes"), A -> C (otherwise).
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                # Create conditions as callables that check the message content.
                builder.add_edge(agent_a, agent_b, condition=lambda msg: "yes" in msg.to_model_text())
                builder.add_edge(agent_a, agent_c, condition=lambda msg: "yes" not in msg.to_model_text())
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(5),
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="AutoGen is a framework for building AI agents."):
                    print(event)


            asyncio.run(main())

        **Loop with exit condition: A → B → C (if 'APPROVE') or A (otherwise)**

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main():
                # Initialize agents with OpenAI model clients.
                model_client = OpenAIChatCompletionClient(model="gpt-4.1")
                agent_a = AssistantAgent(
                    "A",
                    model_client=model_client,
                    system_message="You are a helpful assistant.",
                )
                agent_b = AssistantAgent(
                    "B",
                    model_client=model_client,
                    system_message="Provide feedback on the input, if your feedback has been addressed, "
                    "say 'APPROVE', otherwise provide a reason for rejection.",
                )
                agent_c = AssistantAgent(
                    "C", model_client=model_client, system_message="Translate the final product to Korean."
                )

                # Create a loop graph with conditional exit: A -> B -> C ("APPROVE"), B -> A (otherwise).
                builder = DiGraphBuilder()
                builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
                builder.add_edge(agent_a, agent_b)

                # Create conditional edges using strings
                builder.add_edge(agent_b, agent_c, condition=lambda msg: "APPROVE" in msg.to_model_text())
                builder.add_edge(agent_b, agent_a, condition=lambda msg: "APPROVE" not in msg.to_model_text())

                builder.set_entry_point(agent_a)
                graph = builder.build()

                # Create a GraphFlow team with the directed graph.
                team = GraphFlow(
                    participants=[agent_a, agent_b, agent_c],
                    graph=graph,
                    termination_condition=MaxMessageTermination(20),  # Max 20 messages to avoid infinite loop.
                )

                # Run the team and print the events.
                async for event in team.run_stream(task="Write a short poem about AI Agents."):
                    print(event)


            asyncio.run(main())
    """

    component_config_schema = GraphFlowConfig
    component_provider_override = "autogen_agentchat.teams.GraphFlow"

    DEFAULT_NAME = "GraphFlow"
    DEFAULT_DESCRIPTION = "A team of agents"

    def __init__(
        self,
        participants: List[ChatAgent],
        graph: DiGraph,
        *,
        name: str | None = None,
        description: str | None = None,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        custom_message_types: List[type[BaseAgentEvent | BaseChatMessage]] | None = None,
    ) -> None:
        self._input_participants = participants
        self._input_termination_condition = termination_condition

        for participant in participants:
            if not isinstance(participant, ChatAgent):
                raise TypeError(f"Participant {participant} must be a ChatAgent.")

        # No longer add _StopAgent or StopMessageTermination
        # Termination is now handled directly in GraphFlowManager._apply_termination_condition
        super().__init__(
            name=name or self.DEFAULT_NAME,
            description=description or self.DEFAULT_DESCRIPTION,
            participants=list(participants),
            group_chat_manager_name="GraphManager",
            group_chat_manager_class=GraphFlowManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types,
        )
        self._graph = graph

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
    ) -> Callable[[], GraphFlowManager]:
        """Creates the factory method for initializing the DiGraph-based chat manager."""

        def _factory() -> GraphFlowManager:
            return GraphFlowManager(
                name=name,
                group_topic_type=group_topic_type,
                output_topic_type=output_topic_type,
                participant_topic_types=participant_topic_types,
                participant_names=participant_names,
                participant_descriptions=participant_descriptions,
                output_message_queue=output_message_queue,
                termination_condition=termination_condition,
                max_turns=max_turns,
                message_factory=message_factory,
                graph=self._graph,
            )

        return _factory

    def _to_config(self) -> GraphFlowConfig:
        """Converts the instance into a configuration object."""
        participants = [participant.dump_component() for participant in self._input_participants]
        termination_condition = (
            self._input_termination_condition.dump_component() if self._input_termination_condition else None
        )
        return GraphFlowConfig(
            name=self._name,
            description=self._description,
            participants=participants,
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            graph=self._graph,
        )

    @classmethod
    def _from_config(cls, config: GraphFlowConfig) -> Self:
        """Reconstructs an instance from a configuration object."""
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )
        return cls(
            name=config.name,
            description=config.description,
            participants=participants,
            graph=config.graph,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
        )

# From _graph/_digraph_group_chat.py
def check_condition(self, message: BaseChatMessage) -> bool:
        """Check if the edge condition is satisfied for the given message.

        Args:
            message: The message to check the condition against.

        Returns:
            True if condition is satisfied (None condition always returns True),
            False otherwise.
        """
        if self.condition_function is not None:
            return self.condition_function(message)
        elif isinstance(self.condition, str):
            # If it's a string, check if the string is in the message content
            return self.condition in message.to_model_text()
        return True

# From _graph/_digraph_group_chat.py
def get_parents(self) -> Dict[str, List[str]]:
        """Compute a mapping of each node to its parent nodes."""
        parents: Dict[str, List[str]] = {node: [] for node in self.nodes}
        for node in self.nodes.values():
            for edge in node.edges:
                parents[edge.target].append(node.name)
        return parents

# From _graph/_digraph_group_chat.py
def get_start_nodes(self) -> Set[str]:
        """Return the nodes that have no incoming edges (entry points)."""
        if self.default_start_node:
            return {self.default_start_node}

        parents = self.get_parents()
        return set([node_name for node_name, parent_list in parents.items() if not parent_list])

# From _graph/_digraph_group_chat.py
def get_leaf_nodes(self) -> Set[str]:
        """Return nodes that have no outgoing edges (final output nodes)."""
        return set([name for name, node in self.nodes.items() if not node.edges])

# From _graph/_digraph_group_chat.py
def has_cycles_with_exit(self) -> bool:
        """
        Check if the graph has any cycles and validate that each cycle has at least one conditional edge.

        Returns:
            bool: True if there is at least one cycle and all cycles have an exit condition.
                False if there are no cycles.

        Raises:
            ValueError: If there is a cycle without any conditional edge.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            for edge in self.nodes[node_name].edges:
                target = edge.target
                if target not in visited:
                    if dfs(target):
                        return True
                elif target in rec_stack:
                    # Found a cycle → extract the cycle
                    cycle_start_index = path.index(target)
                    cycle_nodes = path[cycle_start_index:]
                    cycle_edges: List[DiGraphEdge] = []
                    for n in cycle_nodes:
                        cycle_edges.extend(self.nodes[n].edges)
                    if all(edge.condition is None and edge.condition_function is None for edge in cycle_edges):
                        raise ValueError(
                            f"Cycle detected without exit condition: {' -> '.join(cycle_nodes + cycle_nodes[:1])}"
                        )
                    return True  # Found cycle, but it has an exit condition

            rec_stack.remove(node_name)
            path.pop()
            return False

        has_cycle = False
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    has_cycle = True

        return has_cycle

# From _graph/_digraph_group_chat.py
def get_has_cycles(self) -> bool:
        """Indicates if the graph has at least one cycle (with valid exit conditions)."""
        if self._has_cycles is None:
            self._has_cycles = self.has_cycles_with_exit()

        return self._has_cycles

# From _graph/_digraph_group_chat.py
def graph_validate(self) -> None:
        """Validate graph structure and execution rules."""
        if not self.nodes:
            raise ValueError("Graph has no nodes.")

        if not self.get_start_nodes():
            raise ValueError("Graph must have at least one start node")

        if not self.get_leaf_nodes():
            raise ValueError("Graph must have at least one leaf node")

        # Outgoing edge condition validation (per node)
        for node in self.nodes.values():
            # Check that if a node has an outgoing conditional edge, then all outgoing edges are conditional
            has_condition = any(
                edge.condition is not None or edge.condition_function is not None for edge in node.edges
            )
            has_unconditioned = any(edge.condition is None and edge.condition_function is None for edge in node.edges)
            if has_condition and has_unconditioned:
                raise ValueError(f"Node '{node.name}' has a mix of conditional and unconditional edges.")

        # Validate activation conditions across all edges in the graph
        self._validate_activation_conditions()

        self._has_cycles = self.has_cycles_with_exit()

# From _graph/_digraph_group_chat.py
def get_remaining_map(self) -> Dict[str, Dict[str, int]]:
        """Get the remaining map that tracks how many edges point to each target node with each activation group.

        Returns:
            Dictionary mapping target nodes to their activation groups and remaining counts
        """

        remaining_map: Dict[str, Dict[str, int]] = {}

        for node in self.nodes.values():
            for edge in node.edges:
                target = edge.target
                activation_group = edge.activation_group

                if target not in remaining_map:
                    remaining_map[target] = {}

                if activation_group not in remaining_map[target]:
                    remaining_map[target][activation_group] = 0

                remaining_map[target][activation_group] += 1

        return remaining_map

# From _graph/_digraph_group_chat.py
def dfs(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            for edge in self.nodes[node_name].edges:
                target = edge.target
                if target not in visited:
                    if dfs(target):
                        return True
                elif target in rec_stack:
                    # Found a cycle → extract the cycle
                    cycle_start_index = path.index(target)
                    cycle_nodes = path[cycle_start_index:]
                    cycle_edges: List[DiGraphEdge] = []
                    for n in cycle_nodes:
                        cycle_edges.extend(self.nodes[n].edges)
                    if all(edge.condition is None and edge.condition_function is None for edge in cycle_edges):
                        raise ValueError(
                            f"Cycle detected without exit condition: {' -> '.join(cycle_nodes + cycle_nodes[:1])}"
                        )
                    return True  # Found cycle, but it has an exit condition

            rec_stack.remove(node_name)
            path.pop()
            return False

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_core.code_executor import CodeExecutor
from autogen_core.tools import BaseTool
from pydantic import model_serializer

# From code_execution/_code_execution.py
class CodeExecutionInput(BaseModel):
    code: str = Field(description="The contents of the Python code block that should be executed")

# From code_execution/_code_execution.py
class CodeExecutionResult(BaseModel):
    success: bool
    output: str

    @model_serializer
    def ser_model(self) -> str:
        return self.output

# From code_execution/_code_execution.py
class PythonCodeExecutionToolConfig(BaseModel):
    """Configuration for PythonCodeExecutionTool"""

    executor: ComponentModel
    description: str = "Execute Python code blocks."

# From code_execution/_code_execution.py
class PythonCodeExecutionTool(
    BaseTool[CodeExecutionInput, CodeExecutionResult], Component[PythonCodeExecutionToolConfig]
):
    """A tool that executes Python code in a code executor and returns output.

    Example executors:

    * :class:`autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`
    * :class:`autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`
    * :class:`autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor`

    Example usage:

    .. code-block:: bash

        pip install -U "autogen-agentchat" "autogen-ext[openai]" "yfinance" "matplotlib"

    .. code-block:: python

        import asyncio
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.ui import Console
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
        from autogen_ext.tools.code_execution import PythonCodeExecutionTool


        async def main() -> None:
            tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
            agent = AssistantAgent(
                "assistant", OpenAIChatCompletionClient(model="gpt-4o"), tools=[tool], reflect_on_tool_use=True
            )
            await Console(
                agent.run_stream(
                    task="Create a plot of MSFT stock prices in 2024 and save it to a file. Use yfinance and matplotlib."
                )
            )


        asyncio.run(main())


    Args:
        executor (CodeExecutor): The code executor that will be used to execute the code blocks.
    """

    component_config_schema = PythonCodeExecutionToolConfig
    component_provider_override = "autogen_ext.tools.code_execution.PythonCodeExecutionTool"

    def __init__(self, executor: CodeExecutor):
        super().__init__(CodeExecutionInput, CodeExecutionResult, "CodeExecutor", "Execute Python code blocks.")
        self._executor = executor

    async def run(self, args: CodeExecutionInput, cancellation_token: CancellationToken) -> CodeExecutionResult:
        code_blocks = [CodeBlock(code=args.code, language="python")]
        result = await self._executor.execute_code_blocks(
            code_blocks=code_blocks, cancellation_token=cancellation_token
        )
        return CodeExecutionResult(success=result.exit_code == 0, output=result.output)

    def _to_config(self) -> PythonCodeExecutionToolConfig:
        """Convert current instance to config object"""
        return PythonCodeExecutionToolConfig(executor=self._executor.dump_component())

    @classmethod
    def _from_config(cls, config: PythonCodeExecutionToolConfig) -> Self:
        """Create instance from config object"""
        executor = CodeExecutor.load_component(config.executor)
        return cls(executor=executor)

# From code_execution/_code_execution.py
def ser_model(self) -> str:
        return self.output

import atexit
import datetime
import io
import json
import logging
import os
import secrets
import uuid
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Protocol
from typing import Type
from typing import cast
from typing import runtime_checkable
import aiohttp
import docker
import docker.errors
import requests
import websockets
from requests.adapters import HTTPAdapter
from requests.adapters import Retry

# From docker_jupyter/_jupyter_server.py
class JupyterConnectionInfo:
    """(Experimental)"""

    host: str
    """`str` - Host of the Jupyter gateway server"""
    use_https: bool
    """`bool` - Whether to use HTTPS"""
    port: Optional[int] = None
    """`Optional[int]` - Port of the Jupyter gateway server. If None, the default port is used"""
    token: Optional[str] = None
    """`Optional[str]` - Token for authentication. If None, no token is used"""

# From docker_jupyter/_jupyter_server.py
class JupyterConnectable(Protocol):
    """(Experimental)"""

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        """Return the connection information for this connectable."""
        ...

# From docker_jupyter/_jupyter_server.py
class JupyterClient:
    def __init__(self, connection_info: JupyterConnectionInfo):
        """(Experimental) A client for communicating with a Jupyter gateway server.

        Args:
            connection_info (JupyterConnectionInfo): Connection information
        """
        self._connection_info = connection_info
        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        self._session.mount("http://", HTTPAdapter(max_retries=retries))
        # Create aiohttp session for async requests
        self._async_session: aiohttp.ClientSession | None = None

    async def _ensure_async_session(self) -> aiohttp.ClientSession:
        if self._async_session is None:
            self._async_session = aiohttp.ClientSession()
        return self._async_session

    def _get_headers(self) -> Dict[str, str]:
        if self._connection_info.token is None:
            return {}
        return {"Authorization": f"token {self._connection_info.token}"}

    def _get_api_base_url(self) -> str:
        protocol = "https" if self._connection_info.use_https else "http"
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"{protocol}://{self._connection_info.host}{port}"

    def _get_ws_base_url(self) -> str:
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"ws://{self._connection_info.host}{port}"

    async def list_kernel_specs(self) -> Dict[str, Dict[str, str]]:
        response = self._session.get(f"{self._get_api_base_url()}/api/kernelspecs", headers=self._get_headers())
        return cast(Dict[str, Dict[str, str]], response.json())

    async def list_kernels(self) -> List[Dict[str, str]]:
        response = self._session.get(f"{self._get_api_base_url()}/api/kernels", headers=self._get_headers())
        return cast(List[Dict[str, str]], response.json())

    async def start_kernel(self, kernel_spec_name: str) -> str:
        """Start a new kernel asynchronously.

        Args:
            kernel_spec_name (str): Name of the kernel spec to start

        Returns:
            str: ID of the started kernel
        """
        session = await self._ensure_async_session()
        async with session.post(
            f"{self._get_api_base_url()}/api/kernels",
            headers=self._get_headers(),
            json={"name": kernel_spec_name},
        ) as response:
            data = await response.json()
            return cast(str, data["id"])

    async def delete_kernel(self, kernel_id: str) -> None:
        session = await self._ensure_async_session()
        async with session.delete(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}", headers=self._get_headers()
        ) as response:
            response.raise_for_status()

    async def restart_kernel(self, kernel_id: str) -> None:
        session = await self._ensure_async_session()
        async with session.post(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}/restart", headers=self._get_headers()
        ) as response:
            response.raise_for_status()

    async def get_kernel_client(self, kernel_id: str) -> "JupyterKernelClient":
        ws_url = f"{self._get_ws_base_url()}/api/kernels/{kernel_id}/channels"
        # Using websockets library for async websocket connections
        ws = await websockets.connect(ws_url, additional_headers=self._get_headers())
        return JupyterKernelClient(ws)

    async def close(self) -> None:
        """Close the async session"""
        if self._async_session is not None:
            await self._async_session.close()
            self._async_session = None
        self._session.close()

# From docker_jupyter/_jupyter_server.py
class DataItem:
    mime_type: str
    data: str

# From docker_jupyter/_jupyter_server.py
class ExecutionResult:
    is_ok: bool
    output: str
    data_items: List[DataItem]

# From docker_jupyter/_jupyter_server.py
class JupyterKernelClient:
    """An asynchronous client for communicating with a Jupyter kernel."""

    def __init__(self, websocket: websockets.ClientConnection) -> None:
        self._session_id = uuid.uuid4().hex
        self._websocket = websocket

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        await self.stop()

    async def stop(self) -> None:
        await self._websocket.close()

    async def _send_message(self, *, content: Dict[str, Any], channel: str, message_type: str) -> str:
        timestamp = datetime.datetime.now().isoformat()
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "username": "autogen",
                "version": "5.0",
                "session": self._session_id,
                "msg_id": message_id,
                "msg_type": message_type,
                "date": timestamp,
            },
            "parent_header": {},
            "channel": channel,
            "content": content,
            "metadata": {},
            "buffers": {},
        }
        await self._websocket.send(json.dumps(message))
        return message_id

    async def _receive_message(self, timeout_seconds: Optional[float]) -> Optional[Dict[str, Any]]:
        try:
            if timeout_seconds is not None:
                data = await asyncio.wait_for(self._websocket.recv(), timeout=timeout_seconds)
            else:
                data = await self._websocket.recv()
            if isinstance(data, bytes):
                return cast(Dict[str, Any], json.loads(data.decode("utf-8")))
            return cast(Dict[str, Any], json.loads(data))
        except asyncio.TimeoutError:
            return None

    async def wait_for_ready(self, timeout_seconds: Optional[float] = None) -> bool:
        message_id = await self._send_message(content={}, channel="shell", message_type="kernel_info_request")
        while True:
            message = await self._receive_message(timeout_seconds)
            # This means we timed out with no new messages.
            if message is None:
                return False
            if (
                message.get("parent_header", {}).get("msg_id") == message_id
                and message["msg_type"] == "kernel_info_reply"
            ):
                return True

    async def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        message_id = await self._send_message(
            content={
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            channel="shell",
            message_type="execute_request",
        )

        text_output: List[str] = []
        data_output: List[DataItem] = []
        while True:
            message = await self._receive_message(timeout_seconds)
            if message is None:
                return ExecutionResult(
                    is_ok=False, output="ERROR: Timeout waiting for output from code block.", data_items=[]
                )

            # Ignore messages that are not for this execution.
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue

            msg_type = message["msg_type"]
            content = message["content"]
            if msg_type in ["execute_result", "display_data"]:
                for data_type, data in content["data"].items():
                    if data_type == "text/plain":
                        text_output.append(data)
                    elif data_type.startswith("image/") or data_type == "text/html":
                        data_output.append(DataItem(mime_type=data_type, data=data))
                    else:
                        text_output.append(json.dumps(data))
            elif msg_type == "stream":
                text_output.append(content["text"])
            elif msg_type == "error":
                # Output is an error.
                return ExecutionResult(
                    is_ok=False,
                    output=f"ERROR: {content['ename']}: {content['evalue']}\n{content['traceback']}",
                    data_items=[],
                )
            if msg_type == "status" and content["execution_state"] == "idle":
                break
        return ExecutionResult(
            is_ok=True, output="\n".join([str(output) for output in text_output]), data_items=data_output
        )

# From docker_jupyter/_jupyter_server.py
class DockerJupyterServer(JupyterConnectable):
    DEFAULT_DOCKERFILE = """FROM quay.io/jupyter/docker-stacks-foundation

        SHELL ["/bin/bash", "-o", "pipefail", "-c"]

        USER ${NB_UID}
        RUN mamba install --yes jupyter_kernel_gateway ipykernel && \
            mamba clean --all -f -y && \
            fix-permissions "${CONDA_DIR}" && \
            fix-permissions "/home/${NB_USER}"

        ENV TOKEN="UNSET"
        CMD python -m jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 \
            --KernelGatewayApp.port=8888 \
            --KernelGatewayApp.auth_token="${TOKEN}" \
            --JupyterApp.answer_yes=true \
            --JupyterWebsocketPersonality.list_kernels=true

        EXPOSE 8888

        WORKDIR "${HOME}"
        """

    class GenerateToken:
        pass

    def __init__(
        self,
        *,
        custom_image_name: Optional[str] = None,
        container_name: Optional[str] = None,
        auto_remove: bool = True,
        stop_container: bool = True,
        docker_env: Optional[Dict[str, str]] = None,
        expose_port: int = 8888,
        token: Optional[Union[str, GenerateToken]] = None,
        work_dir: Union[Path, str] = "/workspace",
        bind_dir: Optional[Union[Path, str]] = None,
    ):
        """Start a Jupyter kernel gateway server in a Docker container.

        Args:
            custom_image_name: Custom Docker image to use. If None, builds and uses bundled image.
            container_name: Name for the Docker container. Auto-generated if None.
            auto_remove: If True, container will be deleted when stopped.
            stop_container: If True, container stops on program exit or when context manager exits.
            docker_env: Additional environment variables for the container.
            expose_port: Port to expose for Jupyter connection.
            token: Authentication token. If GenerateToken, creates random token. Empty for no auth.
            work_dir: Working directory inside the container.
            bind_dir: Local directory to bind to container's work_dir.
        """
        # Generate container name if not provided
        container_name = container_name or f"autogen-jupyterkernelgateway-{uuid.uuid4()}"

        # Initialize Docker client
        client = docker.from_env()
        # Set up bind directory if specified
        self._bind_dir: Optional[Path] = None
        if bind_dir:
            self._bind_dir = Path(bind_dir) if isinstance(bind_dir, str) else bind_dir
            self._bind_dir.mkdir(exist_ok=True)
            os.chmod(bind_dir, 0o777)

        # Determine and prepare Docker image
        image_name = custom_image_name or "autogen-jupyterkernelgateway"
        if not custom_image_name:
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                # Build default image if not found
                here = Path(__file__).parent
                dockerfile = io.BytesIO(self.DEFAULT_DOCKERFILE.encode("utf-8"))
                logging.info(f"Building image {image_name}...")
                client.images.build(path=str(here), fileobj=dockerfile, tag=image_name)
                logging.info(f"Image {image_name} built successfully")
        else:
            # Verify custom image exists
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound as err:
                raise ValueError(f"Custom image {image_name} does not exist") from err
        if docker_env is None:
            docker_env = {}
        if token is None:
            token = DockerJupyterServer.GenerateToken()
        # Set up authentication token
        self._token = secrets.token_hex(32) if isinstance(token, DockerJupyterServer.GenerateToken) else token

        # Prepare environment variables
        env = {"TOKEN": self._token}
        env.update(docker_env)

        # Define volume configuration if bind directory is specified
        volumes = {str(self._bind_dir): {"bind": str(work_dir), "mode": "rw"}} if self._bind_dir else None

        # Start the container
        container = client.containers.run(
            image_name,
            detach=True,
            auto_remove=auto_remove,
            environment=env,
            publish_all_ports=True,
            name=container_name,
            volumes=volumes,
            working_dir=str(work_dir),
        )

        # Wait for container to be ready
        self._wait_for_ready(container)

        # Store container information
        self._container = container
        self._port = int(container.ports[f"{expose_port}/tcp"][0]["HostPort"])
        self._container_id = container.id
        self._expose_port = expose_port

        if self._container_id is None:
            raise ValueError("Failed to obtain container id.")

        # Define cleanup function
        def cleanup() -> None:
            try:
                assert self._container_id is not None
                inner_container = client.containers.get(self._container_id)
                inner_container.stop()
            except docker.errors.NotFound:
                pass
            atexit.unregister(cleanup)

        # Register cleanup if container should be stopped automatically
        if stop_container:
            atexit.register(cleanup)

        self._cleanup_func = cleanup
        self._stop_container = stop_container

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        return JupyterConnectionInfo(host="127.0.0.1", use_https=False, port=self._port, token=self._token)

    def _wait_for_ready(self, container: Any, timeout: int = 60, stop_time: float = 0.1) -> None:
        elapsed_time = 0.0
        while container.status != "running" and elapsed_time < timeout:
            sleep(stop_time)
            elapsed_time += stop_time
            container.reload()
            continue
        if container.status != "running":
            raise ValueError("Container failed to start")

    async def stop(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._cleanup_func)

    async def get_client(self) -> JupyterClient:
        return JupyterClient(self.connection_info)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        await self.stop()

# From docker_jupyter/_jupyter_server.py
class GenerateToken:
        pass

# From docker_jupyter/_jupyter_server.py
def connection_info(self) -> JupyterConnectionInfo:
        """Return the connection information for this connectable."""
        ...

# From docker_jupyter/_jupyter_server.py
def cleanup() -> None:
            try:
                assert self._container_id is not None
                inner_container = client.containers.get(self._container_id)
                inner_container.stop()
            except docker.errors.NotFound:
                pass
            atexit.unregister(cleanup)

from typing import get_args
from autogen_core import FunctionCall
from autogen_core import Image
from autogen_core.models import AssistantMessage
from autogen_core.models import FunctionExecutionResultMessage
from autogen_core.models import LLMMessage
from autogen_core.models import ModelFamily
from autogen_core.models import SystemMessage
from autogen_core.models import UserMessage
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import ChatCompletionMessageToolCallParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from _transformation import LLMMessageContent
from _transformation import TransformerMap
from _transformation import TrasformerReturnType
from _transformation import build_conditional_transformer_func
from _transformation import build_transformer_func
from _transformation import register_transformer
from _utils import assert_valid_name

# From openai/_message_transform.py
def func_call_to_oai(message: FunctionCall) -> ChatCompletionMessageToolCallParam:
    return ChatCompletionMessageToolCallParam(
        id=message.id,
        function={
            "arguments": message.arguments,
            "name": message.name,
        },
        type="function",
    )

# From openai/_message_transform.py
def user_condition(message: LLMMessage, context: Dict[str, Any]) -> str:
    if isinstance(message.content, str):
        return "text"
    else:
        return "multimodal"

# From openai/_message_transform.py
def assistant_condition(message: LLMMessage, context: Dict[str, Any]) -> str:
    assert isinstance(message, AssistantMessage)
    if isinstance(message.content, list):
        if message.thought is not None:
            return "thought"
        else:
            return "tools"
    else:
        return "text"

# From openai/_message_transform.py
def function_execution_result_message(message: LLMMessage, context: Dict[str, Any]) -> TrasformerReturnType:
    assert isinstance(message, FunctionExecutionResultMessage)
    return [
        ChatCompletionToolMessageParam(content=x.content, role="tool", tool_call_id=x.call_id) for x in message.content
    ]

# From openai/_message_transform.py
def inner(message: LLMMessage, context: Dict[str, Any]) -> Dict[str, str]:
        return {"role": role}

import warnings
from autogen_core import EVENT_LOGGER_NAME
from autogen_core._cancellation_token import CancellationToken
from autogen_core.logging import LLMCallEvent
from autogen_core.logging import LLMStreamEndEvent
from autogen_core.logging import LLMStreamStartEvent
from autogen_core.models import ChatCompletionClient
from autogen_core.models import CreateResult
from autogen_core.models import ModelInfo
from autogen_core.models import RequestUsage
from autogen_core.models import validate_model_info
from autogen_core.tools import Tool
from autogen_core.tools import ToolSchema
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents import FinishReason
from semantic_kernel.contents import FunctionCallContent
from semantic_kernel.contents import FunctionResultContent
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.kernel import Kernel
from typing_extensions import AsyncGenerator
from autogen_ext.tools.semantic_kernel import KernelFunctionFromTool
from autogen_ext.tools.semantic_kernel import KernelFunctionFromToolSchema
from _utils.parse_r1_content import parse_r1_content

# From semantic_kernel/_sk_chat_completion_adapter.py
class SKChatCompletionAdapter(ChatCompletionClient):
    """
    SKChatCompletionAdapter is an adapter that allows using Semantic Kernel model clients
    as Autogen ChatCompletion clients. This makes it possible to seamlessly integrate
    Semantic Kernel connectors (e.g., Azure OpenAI, Google Gemini, Ollama, etc.) into
    Autogen agents that rely on a ChatCompletionClient interface.

    By leveraging this adapter, you can:

    - Pass in a `Kernel` and any supported Semantic Kernel `ChatCompletionClientBase` connector.
    - Provide tools (via Autogen `Tool` or `ToolSchema`) for function calls during chat completion.
    - Stream responses or retrieve them in a single request.
    - Provide prompt settings to control the chat completion behavior either globally through the constructor
        or on a per-request basis through the `extra_create_args` dictionary.

    The list of extras that can be installed:

    - `semantic-kernel-anthropic`: Install this extra to use Anthropic models.
    - `semantic-kernel-google`: Install this extra to use Google Gemini models.
    - `semantic-kernel-ollama`: Install this extra to use Ollama models.
    - `semantic-kernel-mistralai`: Install this extra to use MistralAI models.
    - `semantic-kernel-aws`: Install this extra to use AWS models.
    - `semantic-kernel-hugging-face`: Install this extra to use Hugging Face models.

    Args:
        sk_client (ChatCompletionClientBase):
            The Semantic Kernel client to wrap (e.g., AzureChatCompletion, GoogleAIChatCompletion, OllamaChatCompletion).
        kernel (Optional[Kernel]):
            The Semantic Kernel instance to use for executing requests. If not provided, one must be passed
            in the extra_create_args for each request.
        prompt_settings (Optional[PromptExecutionSettings]):
            Default prompt execution settings to use. Can be overridden per request.
        model_info (Optional[ModelInfo]):
            Information about the model's capabilities.
        service_id (Optional[str]):
            Optional service identifier.

    Examples:

        Anthropic models with function calling:

        .. code-block:: bash

            pip install "autogen-ext[semantic-kernel-anthropic]"

        .. code-block:: python

            import asyncio
            import os

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
            from autogen_core.models import ModelFamily, UserMessage
            from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
            from semantic_kernel import Kernel
            from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion, AnthropicChatPromptExecutionSettings
            from semantic_kernel.memory.null_memory import NullMemory


            async def get_weather(city: str) -> str:
                \"\"\"Get the weather for a city.\"\"\"
                return f"The weather in {city} is 75 degrees."


            async def main() -> None:
                sk_client = AnthropicChatCompletion(
                    ai_model_id="claude-3-5-sonnet-20241022",
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    service_id="my-service-id",  # Optional; for targeting specific services within Semantic Kernel
                )
                settings = AnthropicChatPromptExecutionSettings(
                    temperature=0.2,
                )

                model_client = SKChatCompletionAdapter(
                    sk_client,
                    kernel=Kernel(memory=NullMemory()),
                    prompt_settings=settings,
                    model_info={
                        "function_calling": True,
                        "json_output": True,
                        "vision": True,
                        "family": ModelFamily.CLAUDE_3_5_SONNET,
                        "structured_output": True,
                    },
                )

                # Call the model directly.
                response = await model_client.create([UserMessage(content="What is the capital of France?", source="test")])
                print(response)

                # Create an assistant agent with the model client.
                assistant = AssistantAgent(
                    "assistant", model_client=model_client, system_message="You are a helpful assistant.", tools=[get_weather]
                )
                # Call the assistant with a task.
                await Console(assistant.run_stream(task="What is the weather in Paris and London?"))


            asyncio.run(main())


        Google Gemini models with function calling:

        .. code-block:: bash

            pip install "autogen-ext[semantic-kernel-google]"

        .. code-block:: python

            import asyncio
            import os

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
            from autogen_core.models import UserMessage, ModelFamily
            from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
            from semantic_kernel import Kernel
            from semantic_kernel.connectors.ai.google.google_ai import (
                GoogleAIChatCompletion,
                GoogleAIChatPromptExecutionSettings,
            )
            from semantic_kernel.memory.null_memory import NullMemory


            def get_weather(city: str) -> str:
                \"\"\"Get the weather for a city.\"\"\"
                return f"The weather in {city} is 75 degrees."


            async def main() -> None:
                sk_client = GoogleAIChatCompletion(
                    gemini_model_id="gemini-2.0-flash",
                    api_key=os.environ["GEMINI_API_KEY"],
                )
                settings = GoogleAIChatPromptExecutionSettings(
                    temperature=0.2,
                )

                kernel = Kernel(memory=NullMemory())

                model_client = SKChatCompletionAdapter(
                    sk_client,
                    kernel=kernel,
                    prompt_settings=settings,
                    model_info={
                        "family": ModelFamily.GEMINI_2_0_FLASH,
                        "function_calling": True,
                        "json_output": True,
                        "vision": True,
                        "structured_output": True,
                    },
                )

                # Call the model directly.
                model_result = await model_client.create(
                    messages=[UserMessage(content="What is the capital of France?", source="User")]
                )
                print(model_result)

                # Create an assistant agent with the model client.
                assistant = AssistantAgent(
                    "assistant", model_client=model_client, tools=[get_weather], system_message="You are a helpful assistant."
                )
                # Call the assistant with a task.
                stream = assistant.run_stream(task="What is the weather in Paris and London?")
                await Console(stream)


            asyncio.run(main())


        Ollama models:

        .. code-block:: bash

            pip install "autogen-ext[semantic-kernel-ollama]"

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.models import UserMessage
            from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
            from semantic_kernel import Kernel
            from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
            from semantic_kernel.memory.null_memory import NullMemory


            async def main() -> None:
                sk_client = OllamaChatCompletion(
                    host="http://localhost:11434",
                    ai_model_id="llama3.2:latest",
                )
                ollama_settings = OllamaChatPromptExecutionSettings(
                    options={"temperature": 0.5},
                )

                model_client = SKChatCompletionAdapter(
                    sk_client, kernel=Kernel(memory=NullMemory()), prompt_settings=ollama_settings
                )

                # Call the model directly.
                model_result = await model_client.create(
                    messages=[UserMessage(content="What is the capital of France?", source="User")]
                )
                print(model_result)

                # Create an assistant agent with the model client.
                assistant = AssistantAgent("assistant", model_client=model_client)
                # Call the assistant with a task.
                result = await assistant.run(task="What is the capital of France?")
                print(result)


            asyncio.run(main())

    """

    def __init__(
        self,
        sk_client: ChatCompletionClientBase,
        kernel: Optional[Kernel] = None,
        prompt_settings: Optional[PromptExecutionSettings] = None,
        model_info: Optional[ModelInfo] = None,
        service_id: Optional[str] = None,
    ):
        self._service_id = service_id
        self._kernel = kernel
        self._prompt_settings = prompt_settings
        self._sk_client = sk_client
        self._model_info = model_info or ModelInfo(
            vision=False, function_calling=False, json_output=False, family=ModelFamily.UNKNOWN, structured_output=False
        )
        validate_model_info(self._model_info)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._tools_plugin: KernelPlugin = KernelPlugin(name="autogen_tools")

    def _convert_to_chat_history(self, messages: Sequence[LLMMessage]) -> ChatHistory:
        """Convert Autogen LLMMessages to SK ChatHistory"""
        chat_history = ChatHistory()

        for msg in messages:
            if msg.type == "SystemMessage":
                chat_history.add_system_message(msg.content)

            elif msg.type == "UserMessage":
                if isinstance(msg.content, str):
                    chat_history.add_user_message(msg.content)
                else:
                    # Handle list of str/Image - convert to string for now
                    chat_history.add_user_message(str(msg.content))

            elif msg.type == "AssistantMessage":
                # Check if it's a function-call style message
                if isinstance(msg.content, list) and all(isinstance(fc, FunctionCall) for fc in msg.content):
                    # If there's a 'thought' field, you can add that as plain assistant text
                    if msg.thought:
                        chat_history.add_assistant_message(msg.thought)

                    function_call_contents: list[FunctionCallContent] = []
                    for fc in msg.content:
                        function_call_contents.append(
                            FunctionCallContent(
                                id=fc.id,
                                name=fc.name,
                                plugin_name=self._tools_plugin.name,
                                function_name=fc.name,
                                arguments=fc.arguments,
                            )
                        )

                    # Mark the assistant's message as tool-calling
                    chat_history.add_assistant_message(
                        function_call_contents,
                        finish_reason=FinishReason.TOOL_CALLS,
                    )
                else:
                    # Plain assistant text
                    chat_history.add_assistant_message(msg.content)

            elif msg.type == "FunctionExecutionResultMessage":
                # Add each function result as a separate tool message
                tool_results: list[FunctionResultContent] = []
                for result in msg.content:
                    tool_results.append(
                        FunctionResultContent(
                            id=result.call_id,
                            plugin_name=self._tools_plugin.name,
                            function_name=result.name,
                            result=result.content,
                        )
                    )
                # A single "tool" message with one or more results
                chat_history.add_tool_message(tool_results)

        return chat_history

    def _build_execution_settings(
        self, default_prompt_settings: Optional[PromptExecutionSettings], tools: Sequence[Tool | ToolSchema]
    ) -> PromptExecutionSettings:
        """Build PromptExecutionSettings from extra_create_args"""

        if default_prompt_settings is not None:
            prompt_args: dict[str, Any] = default_prompt_settings.prepare_settings_dict()  # type: ignore
        else:
            prompt_args = {}

        # If tools are available, configure function choice behavior with auto_invoke disabled
        function_choice_behavior = None
        if tools:
            function_choice_behavior = FunctionChoiceBehavior.Auto(  # type: ignore
                auto_invoke=False
            )

        # Create settings with remaining args as extension_data
        settings = PromptExecutionSettings(
            service_id=self._service_id,
            extension_data=prompt_args,
            function_choice_behavior=function_choice_behavior,
        )

        return settings

    def _sync_tools_with_kernel(self, kernel: Kernel, tools: Sequence[Tool | ToolSchema]) -> None:
        """Sync tools with kernel by updating the plugin"""
        # Get current tool names in plugin
        current_tool_names = set(self._tools_plugin.functions.keys())

        # Get new tool names
        new_tool_names = {tool.schema["name"] if isinstance(tool, Tool) else tool["name"] for tool in tools}

        # Remove tools that are no longer needed
        for tool_name in current_tool_names - new_tool_names:
            del self._tools_plugin.functions[tool_name]

        # Add or update tools
        for tool in tools:
            if isinstance(tool, BaseTool):
                # Convert Tool to KernelFunction using KernelFunctionFromTool
                kernel_function = KernelFunctionFromTool(tool)  # type: ignore
                self._tools_plugin.functions[tool.schema["name"]] = kernel_function
            else:
                kernel_function = KernelFunctionFromToolSchema(tool)  # type: ignore
                self._tools_plugin.functions[tool.get("name")] = kernel_function  # type: ignore

        kernel.add_plugin(self._tools_plugin)

    def _process_tool_calls(self, result: ChatMessageContent) -> list[FunctionCall]:
        """Process tool calls from SK ChatMessageContent"""
        function_calls: list[FunctionCall] = []
        for item in result.items:
            if isinstance(item, FunctionCallContent):
                # Extract plugin name and function name
                plugin_name = item.plugin_name or ""
                function_name = item.function_name
                if plugin_name:
                    full_name = f"{plugin_name}-{function_name}"
                else:
                    full_name = function_name

                if item.id is None:
                    raise ValueError("Function call ID is required")

                if isinstance(item.arguments, Mapping):
                    arguments = json.dumps(item.arguments)
                else:
                    arguments = item.arguments or "{}"

                function_calls.append(FunctionCall(id=item.id, name=full_name, arguments=arguments))
        return function_calls

    def _get_kernel(self, extra_create_args: Mapping[str, Any]) -> Kernel:
        kernel = extra_create_args.get("kernel", self._kernel)
        if not kernel:
            raise ValueError("kernel must be provided either in constructor or extra_create_args")
        if not isinstance(kernel, Kernel):
            raise ValueError("kernel must be an instance of semantic_kernel.kernel.Kernel")
        return kernel

    def _get_prompt_settings(self, extra_create_args: Mapping[str, Any]) -> Optional[PromptExecutionSettings]:
        return extra_create_args.get("prompt_execution_settings", None) or self._prompt_settings

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
        """Create a chat completion using the Semantic Kernel client.

        The `extra_create_args` dictionary can include two special keys:

        1) `"kernel"` (optional):
            An instance of :class:`semantic_kernel.Kernel` used to execute the request.
            If not provided either in constructor or extra_create_args, a ValueError is raised.

        2) `"prompt_execution_settings"` (optional):
            An instance of a :class:`PromptExecutionSettings` subclass corresponding to the
            underlying Semantic Kernel client (e.g., `AzureChatPromptExecutionSettings`,
            `GoogleAIChatPromptExecutionSettings`). If not provided, the adapter's default
            prompt settings will be used.

        Args:
            messages: The list of LLM messages to send.
            tools: The tools that may be invoked during the chat.
            json_output: Whether the model is expected to return JSON.
            extra_create_args: Additional arguments to control the chat completion behavior.
            cancellation_token: Token allowing cancellation of the request.

        Returns:
            CreateResult: The result of the chat completion.
        """
        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            raise ValueError("structured output is not currently supported in SKChatCompletionAdapter")

        # Handle tool_choice parameter
        if tool_choice != "auto":
            warnings.warn(
                "tool_choice parameter is specified but may not be fully supported by SKChatCompletionAdapter.",
                stacklevel=2,
            )

        kernel = self._get_kernel(extra_create_args)

        chat_history = self._convert_to_chat_history(messages)
        user_settings = self._get_prompt_settings(extra_create_args)
        settings = self._build_execution_settings(user_settings, tools)

        # Sync tools with kernel
        self._sync_tools_with_kernel(kernel, tools)

        result = await self._sk_client.get_chat_message_contents(chat_history, settings=settings, kernel=kernel)
        # Track token usage from result metadata
        prompt_tokens = 0
        completion_tokens = 0

        if result[0].metadata and "usage" in result[0].metadata:
            usage = result[0].metadata["usage"]
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

        logger.info(
            LLMCallEvent(
                messages=[msg.model_dump() for msg in chat_history],
                response=ensure_serializable(result[0]).model_dump(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Process content based on whether there are tool calls
        content: Union[str, list[FunctionCall]]
        if any(isinstance(item, FunctionCallContent) for item in result[0].items):
            content = self._process_tool_calls(result[0])
            finish_reason: Literal["function_calls", "stop"] = "function_calls"
        else:
            content = result[0].content
            finish_reason = "stop"

        if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1:
            thought, content = parse_r1_content(content)
        else:
            thought = None

        return CreateResult(
            content=content,
            finish_reason=finish_reason,
            usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            cached=False,
            thought=thought,
        )

    @staticmethod
    def _merge_function_call_content(existing_call: FunctionCallContent, new_chunk: FunctionCallContent) -> None:
        """Helper to merge partial argument chunks from new_chunk into existing_call."""
        if isinstance(existing_call.arguments, str) and isinstance(new_chunk.arguments, str):
            existing_call.arguments += new_chunk.arguments
        elif isinstance(existing_call.arguments, dict) and isinstance(new_chunk.arguments, dict):
            existing_call.arguments.update(new_chunk.arguments)
        elif not existing_call.arguments or existing_call.arguments in ("{}", ""):
            # If existing had no arguments yet, just take the new one
            existing_call.arguments = new_chunk.arguments
        else:
            # If there's a mismatch (str vs dict), handle as needed
            warnings.warn("Mismatch in argument types during merge. Existing arguments retained.", stacklevel=2)

        # Optionally update name/function_name if newly provided
        if new_chunk.name:
            existing_call.name = new_chunk.name
        if new_chunk.plugin_name:
            existing_call.plugin_name = new_chunk.plugin_name
        if new_chunk.function_name:
            existing_call.function_name = new_chunk.function_name

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
        """Create a streaming chat completion using the Semantic Kernel client.

        The `extra_create_args` dictionary can include two special keys:

        1) `"kernel"` (optional):
            An instance of :class:`semantic_kernel.Kernel` used to execute the request.
            If not provided either in constructor or extra_create_args, a ValueError is raised.

        2) `"prompt_execution_settings"` (optional):
            An instance of a :class:`PromptExecutionSettings` subclass corresponding to the
            underlying Semantic Kernel client (e.g., `AzureChatPromptExecutionSettings`,
            `GoogleAIChatPromptExecutionSettings`). If not provided, the adapter's default
            prompt settings will be used.

        Args:
            messages: The list of LLM messages to send.
            tools: The tools that may be invoked during the chat.
            json_output: Whether the model is expected to return JSON.
            extra_create_args: Additional arguments to control the chat completion behavior.
            cancellation_token: Token allowing cancellation of the request.

        Yields:
            Union[str, CreateResult]: Either a string chunk of the response or a CreateResult containing function calls.
        """

        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            raise ValueError("structured output is not currently supported in SKChatCompletionAdapter")

        # Handle tool_choice parameter
        if tool_choice != "auto":
            warnings.warn(
                "tool_choice parameter is specified but may not be fully supported by SKChatCompletionAdapter.",
                stacklevel=2,
            )

        kernel = self._get_kernel(extra_create_args)
        chat_history = self._convert_to_chat_history(messages)
        user_settings = self._get_prompt_settings(extra_create_args)
        settings = self._build_execution_settings(user_settings, tools)
        self._sync_tools_with_kernel(kernel, tools)

        prompt_tokens = 0
        completion_tokens = 0
        accumulated_text = ""

        # Keep track of in-progress function calls. Keyed by ID
        # because partial chunks for the same function call might arrive separately.
        function_calls_in_progress: dict[str, FunctionCallContent] = {}

        # Track the ID of the last function call we saw so we can continue
        # accumulating chunk arguments for that call if new items have id=None
        last_function_call_id: Optional[str] = None

        first_chunk = True

        async for streaming_messages in self._sk_client.get_streaming_chat_message_contents(
            chat_history, settings=settings, kernel=kernel
        ):
            if first_chunk:
                first_chunk = False
                # Emit the start event.
                logger.info(
                    LLMStreamStartEvent(
                        messages=[msg.model_dump() for msg in chat_history],
                    )
                )
            for msg in streaming_messages:
                # Track token usage
                if msg.metadata and "usage" in msg.metadata:
                    usage = msg.metadata["usage"]
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)

                # Process function call deltas
                for item in msg.items:
                    if isinstance(item, FunctionCallContent):
                        # If the chunk has a valid ID, we start or continue that ID explicitly
                        if item.id:
                            last_function_call_id = item.id
                            if last_function_call_id not in function_calls_in_progress:
                                function_calls_in_progress[last_function_call_id] = item
                            else:
                                # Merge partial arguments into existing call
                                existing_call = function_calls_in_progress[last_function_call_id]
                                self._merge_function_call_content(existing_call, item)
                        else:
                            # item.id is None, so we assume it belongs to the last known ID
                            if not last_function_call_id:
                                # No call in progress means we can't merge
                                # You could either skip or raise an error here
                                warnings.warn(
                                    "Received function call chunk with no ID and no call in progress.", stacklevel=2
                                )
                                continue

                            existing_call = function_calls_in_progress[last_function_call_id]
                            # Merge partial chunk
                            self._merge_function_call_content(existing_call, item)

                # Check if the model signaled tool_calls finished
                if msg.finish_reason == "tool_calls" and function_calls_in_progress:
                    calls_to_yield: list[FunctionCall] = []
                    for _, call_content in function_calls_in_progress.items():
                        plugin_name = call_content.plugin_name or ""
                        function_name = call_content.function_name
                        if plugin_name:
                            full_name = f"{plugin_name}-{function_name}"
                        else:
                            full_name = function_name

                        if isinstance(call_content.arguments, dict):
                            arguments = json.dumps(call_content.arguments)
                        else:
                            assert isinstance(call_content.arguments, str)
                            arguments = call_content.arguments or "{}"

                        calls_to_yield.append(
                            FunctionCall(
                                id=call_content.id or "unknown_id",
                                name=full_name,
                                arguments=arguments,
                            )
                        )
                    # Yield all function calls in progress
                    yield CreateResult(
                        content=calls_to_yield,
                        finish_reason="function_calls",
                        usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                        cached=False,
                    )
                    return

                # Handle any plain text in the message
                if msg.content:
                    accumulated_text += msg.content
                    yield msg.content

        # If we exit the loop without tool calls finishing, yield whatever text was accumulated
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        thought = None
        if isinstance(accumulated_text, str) and self._model_info["family"] == ModelFamily.R1:
            thought, accumulated_text = parse_r1_content(accumulated_text)

        result = CreateResult(
            content=accumulated_text,
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            cached=False,
            thought=thought,
        )

        # Emit the end event.
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

        yield result

    async def close(self) -> None:
        pass  # No explicit close method in SK client?

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=self._total_prompt_tokens, completion_tokens=self._total_completion_tokens)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=self._total_prompt_tokens, completion_tokens=self._total_completion_tokens)

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        chat_history = self._convert_to_chat_history(messages)
        total_tokens = 0
        for message in chat_history.messages:
            if message.metadata and "usage" in message.metadata:
                usage = message.metadata["usage"]
                total_tokens += getattr(usage, "total_tokens", 0)
        return total_tokens

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        # Get total token count
        used_tokens = self.count_tokens(messages)
        # Assume max tokens from SK client if available, otherwise use default
        max_tokens = getattr(self._sk_client, "max_tokens", 4096)
        return max_tokens - used_tokens

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

# From semantic_kernel/_sk_chat_completion_adapter.py
def ensure_serializable(data: BaseModel) -> BaseModel:
    """
    Workaround for https://github.com/pydantic/pydantic/issues/7713, see https://github.com/pydantic/pydantic/issues/7713#issuecomment-2604574418
    """
    try:
        json.dumps(data)
    except TypeError:
        # use `vars` to coerce nested data into dictionaries
        data_json_from_dicts = json.dumps(data, default=lambda x: vars(x))  # type: ignore
        data_obj = json.loads(data_json_from_dicts)
        data = type(data)(**data_obj)
    return data

# From semantic_kernel/_sk_chat_completion_adapter.py
def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=self._total_prompt_tokens, completion_tokens=self._total_completion_tokens)

# From semantic_kernel/_sk_chat_completion_adapter.py
def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=self._total_prompt_tokens, completion_tokens=self._total_completion_tokens)

# From semantic_kernel/_sk_chat_completion_adapter.py
def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        chat_history = self._convert_to_chat_history(messages)
        total_tokens = 0
        for message in chat_history.messages:
            if message.metadata and "usage" in message.metadata:
                usage = message.metadata["usage"]
                total_tokens += getattr(usage, "total_tokens", 0)
        return total_tokens

# From semantic_kernel/_sk_chat_completion_adapter.py
def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        # Get total token count
        used_tokens = self.count_tokens(messages)
        # Assume max tokens from SK client if available, otherwise use default
        max_tokens = getattr(self._sk_client, "max_tokens", 4096)
        return max_tokens - used_tokens

# From semantic_kernel/_sk_chat_completion_adapter.py
def model_info(self) -> ModelInfo:
        return self._model_info

# From semantic_kernel/_sk_chat_completion_adapter.py
def capabilities(self) -> ModelInfo:
        return self.model_info

from langchain import LLMChain
from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel

# From BabyAgi/task_execution.py
class TaskExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "Can you help me to performs one task based on the following objective: "
            "{objective}."
            "Take into account these previously completed tasks: {context}."
            "Can you perform this task? Your task: {task}. Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

# From BabyAgi/task_execution.py
def from_llm(cls, llm: BaseLanguageModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "Can you help me to performs one task based on the following objective: "
            "{objective}."
            "Take into account these previously completed tasks: {context}."
            "Can you perform this task? Your task: {task}. Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

import subprocess
import sys
import importlib
import inspect
from datetime import datetime

# From core/execution.py
class FunctionExecutor:
    def __init__(self, python_func):
        self.python_func = python_func

    def _install_external_dependency(self, package_name: str, imp_name: str) -> Any:
        try:
            return importlib.import_module(imp_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return importlib.import_module(package_name)

    def _resolve_dependencies(self, function_version: Dict[str, Any], local_scope: Dict[str, Any],
                              parent_log_id: Optional[int], executed_functions: List[str],
                              visited: Optional[set] = None) -> None:
        if visited is None:
            visited = set()

        function_name = function_version['name']
        function_imports = self.python_func.db.get_function_imports(function_name)

        for imp in function_imports:
            lib_name = imp['lib'] if imp['lib'] else imp['name']
            if lib_name not in local_scope:
                module = self._install_external_dependency(lib_name, imp['name'])
                local_scope[imp['name']] = module

        for dep_name in function_version.get('dependencies', []):
            if dep_name not in local_scope and dep_name not in visited:
                visited.add(dep_name)
                dep_data = self.python_func.db.get_function(dep_name)
                if not dep_data:
                    raise ValueError(f"Dependency '{dep_name}' not found in the database.")
                self._resolve_dependencies(dep_data, local_scope, parent_log_id, executed_functions, visited)
                exec(dep_data['code'], local_scope)
                if dep_name in local_scope:
                    dep_func = local_scope[dep_name]
                    # Wrap the dependent function
                    local_scope[dep_name] = self._create_function_wrapper(dep_func, dep_name, parent_log_id, executed_functions)

    def _create_function_wrapper(self, func: callable, func_name: str, parent_log_id: int, executed_functions: List[str]):
        def wrapper(*args, **kwargs):
            return self.execute(func_name, *args, executed_functions=executed_functions, parent_log_id=parent_log_id, **kwargs)
        return wrapper

    def execute(
        self,
        function_name: str,
        *args,
        executed_functions: Optional[List[str]] = None,
        parent_log_id: Optional[int] = None,
        wrapper_log_id: Optional[int] = None,
        triggered_by_log_id: Optional[int] = None,
        **kwargs
    ) -> Any:
        start_time = datetime.now()
        executed_functions = executed_functions or []
        log_id = None
        bound_args = None
        output = None

        # Ensure wrapper_log_id is initialized
        wrapper_log_id = wrapper_log_id if wrapper_log_id is not None else None

        logger.info(f"Executing function: {function_name}")

        try:
            executed_functions.append(function_name)
            function_version = self.python_func.db.get_function(function_name)
            if not function_version:
                raise ValueError(f"Function '{function_name}' not found in the database.")

            # If the function being executed is the wrapper, create a special log entry and set wrapper_log_id
            if function_name == 'execute_function_wrapper':
                log_id = self._add_execution_log(
                    function_name,
                    start_time,
                    {},
                    None,
                    0,
                    parent_log_id,
                    triggered_by_log_id,
                    'started'
                )
                wrapper_log_id = log_id
            else:
                log_id = self._add_execution_log(
                    function_name,
                    start_time,
                    {},
                    None,
                    0,
                    wrapper_log_id if wrapper_log_id else parent_log_id,
                    triggered_by_log_id,
                    'started'
                )

            # Include 'datetime' in local_scope
            local_scope = {
                'func': self.python_func,
                'parent_log_id': log_id,
                'datetime': datetime  # Added datetime here
            }

            self._resolve_dependencies(
                function_version,
                local_scope,
                parent_log_id=log_id,
                executed_functions=executed_functions
            )
            self._inject_secret_keys(local_scope)

            exec(function_version['code'], local_scope)
            if function_name not in local_scope:
                raise ValueError(f"Failed to load function '{function_name}'.")

            func = local_scope[function_name]
            bound_args = self._bind_function_arguments(func, args, kwargs)
            self._validate_input_parameters(function_version, bound_args)

            params = bound_args.arguments
            self._update_execution_log_params(log_id, params)

            output = func(*bound_args.args, **bound_args.kwargs)
            end_time = datetime.now()
            time_spent = (end_time - start_time).total_seconds()

            self._update_execution_log(log_id, output, time_spent, 'success')

            # Check and execute triggers for the function
            self._execute_triggered_functions(function_name, output, executed_functions, log_id)
            return output

        except Exception as e:
            end_time = datetime.now()
            time_spent = (end_time - start_time).total_seconds()
            if log_id is not None:
                self._update_execution_log(log_id, None, time_spent, 'error', str(e))
            raise

    

    def _check_key_dependencies(self, function_version: Dict[str, Any]) -> None:
        if 'key_dependencies' in function_version.get('metadata', {}):
            for key_name in function_version['metadata']['key_dependencies']:
                if key_name not in self.python_func.db.get_all_secret_keys():
                    raise ValueError(f"Required secret key '{key_name}' not found for function '{function_version['name']}'")

    def _inject_secret_keys(self, local_scope: Dict[str, Any]) -> None:
        secret_keys = self.python_func.db.get_all_secret_keys()
        if secret_keys:
            logger.debug(f"Injecting secret keys: {list(secret_keys.keys())}")
            local_scope.update(secret_keys)

    def _bind_function_arguments(self, func: callable, args: tuple, kwargs: dict) -> inspect.BoundArguments:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args

    def _validate_input_parameters(self, function_version: Dict[str, Any], bound_args: inspect.BoundArguments) -> None:
        input_params = function_version.get('input_parameters', [])
        for param in input_params:
            if param['name'] not in bound_args.arguments:
                raise ValueError(f"Missing required input parameter '{param['name']}' for function '{function_version['name']}'")

    def _add_execution_log(self, function_name: str, start_time: datetime, params: Dict[str, Any],
                           output: Any, time_spent: float, parent_log_id: Optional[int],
                           triggered_by_log_id: Optional[int], log_type: str, error_message: Optional[str] = None) -> int:
        if log_type == 'started':
            message = "Execution started."
        elif log_type == 'success':
            message = "Execution successful."
        else:
            message = f"Execution failed. Error: {error_message}"
        return self.python_func.db.add_log(
            function_name=function_name,
            message=message,
            timestamp=start_time,
            params=params,
            output=output,
            time_spent=time_spent,
            parent_log_id=parent_log_id,
            triggered_by_log_id=triggered_by_log_id,
            log_type=log_type
        )

    def _update_execution_log(self, log_id: int, output: Any, time_spent: float, log_type: str,
          error_message: Optional[str] = None):
        message = "Execution successful." if log_type == 'success' else f"Execution failed. Error: {error_message}"
        update_data = {
            'message': message,
            'log_type': log_type
        }
        if output is not None:
            update_data['output'] = output
        if time_spent is not None:
            update_data['time_spent'] = time_spent

            self.python_func.db.update_log(log_id=log_id, **update_data)


    def _update_execution_log_params(self, log_id: int, params: Dict[str, Any]) -> None:
        self.python_func.db.update_log(log_id=log_id, params=params)


    
    def _execute_triggered_functions(self, function_name: str, output: Any, executed_functions: List[str], log_id: int) -> None:
        triggered_function_names = self.python_func.db.get_triggers_for_function(function_name)
        logger.info(f"Functions triggered by {function_name}: {triggered_function_names}")

        for triggered_function_name in triggered_function_names:
            if triggered_function_name in executed_functions:
                logger.warning(f"Triggered function '{triggered_function_name}' already executed in this chain. Skipping to prevent recursion.")
                continue

            try:
                logger.info(f"Preparing to execute trigger: {triggered_function_name}")
                triggered_function = self.python_func.db.get_function(triggered_function_name)
                if triggered_function:
                    trigger_args, trigger_kwargs = self._prepare_trigger_arguments(triggered_function, output)
                    logger.info(f"Executing trigger {triggered_function_name} with args: {trigger_args} and kwargs: {trigger_kwargs}")

                    trigger_output = self.execute(
                        triggered_function_name,
                        *trigger_args,
                        executed_functions=executed_functions.copy(),
                        parent_log_id=log_id,
                        triggered_by_log_id=log_id,
                        **trigger_kwargs
                    )
                    logger.info(f"Trigger {triggered_function_name} execution completed. Output: {trigger_output}")
                else:
                    logger.error(f"Triggered function '{triggered_function_name}' not found in the database.")
            except Exception as e:
                logger.error(f"Error executing triggered function '{triggered_function_name}': {str(e)}")

    def _prepare_trigger_arguments(self, triggered_function: Dict[str, Any], output: Any) -> tuple:
        triggered_params = triggered_function.get('input_parameters', [])
        if triggered_params:
            # If the triggered function expects parameters, pass the output as the first parameter
            return (output,), {}
        else:
            # If the triggered function doesn't expect parameters, don't pass any
            return (), {}

# From core/execution.py
def execute(
        self,
        function_name: str,
        *args,
        executed_functions: Optional[List[str]] = None,
        parent_log_id: Optional[int] = None,
        wrapper_log_id: Optional[int] = None,
        triggered_by_log_id: Optional[int] = None,
        **kwargs
    ) -> Any:
        start_time = datetime.now()
        executed_functions = executed_functions or []
        log_id = None
        bound_args = None
        output = None

        # Ensure wrapper_log_id is initialized
        wrapper_log_id = wrapper_log_id if wrapper_log_id is not None else None

        logger.info(f"Executing function: {function_name}")

        try:
            executed_functions.append(function_name)
            function_version = self.python_func.db.get_function(function_name)
            if not function_version:
                raise ValueError(f"Function '{function_name}' not found in the database.")

            # If the function being executed is the wrapper, create a special log entry and set wrapper_log_id
            if function_name == 'execute_function_wrapper':
                log_id = self._add_execution_log(
                    function_name,
                    start_time,
                    {},
                    None,
                    0,
                    parent_log_id,
                    triggered_by_log_id,
                    'started'
                )
                wrapper_log_id = log_id
            else:
                log_id = self._add_execution_log(
                    function_name,
                    start_time,
                    {},
                    None,
                    0,
                    wrapper_log_id if wrapper_log_id else parent_log_id,
                    triggered_by_log_id,
                    'started'
                )

            # Include 'datetime' in local_scope
            local_scope = {
                'func': self.python_func,
                'parent_log_id': log_id,
                'datetime': datetime  # Added datetime here
            }

            self._resolve_dependencies(
                function_version,
                local_scope,
                parent_log_id=log_id,
                executed_functions=executed_functions
            )
            self._inject_secret_keys(local_scope)

            exec(function_version['code'], local_scope)
            if function_name not in local_scope:
                raise ValueError(f"Failed to load function '{function_name}'.")

            func = local_scope[function_name]
            bound_args = self._bind_function_arguments(func, args, kwargs)
            self._validate_input_parameters(function_version, bound_args)

            params = bound_args.arguments
            self._update_execution_log_params(log_id, params)

            output = func(*bound_args.args, **bound_args.kwargs)
            end_time = datetime.now()
            time_spent = (end_time - start_time).total_seconds()

            self._update_execution_log(log_id, output, time_spent, 'success')

            # Check and execute triggers for the function
            self._execute_triggered_functions(function_name, output, executed_functions, log_id)
            return output

        except Exception as e:
            end_time = datetime.now()
            time_spent = (end_time - start_time).total_seconds()
            if log_id is not None:
                self._update_execution_log(log_id, None, time_spent, 'error', str(e))
            raise

# From core/execution.py
def wrapper(*args, **kwargs):
            return self.execute(func_name, *args, executed_functions=executed_functions, parent_log_id=parent_log_id, **kwargs)

from agent import Agent
from python.helpers import files
from python.helpers.print_style import PrintStyle
from python.helpers.tool import Tool
from python.helpers.tool import Response

# From tools/task_done.py
class TaskDone(Tool):

    def execute(self,**kwargs):
        self.agent.set_data("timeout", 0)
        return Response(message=self.args["text"], break_loop=True)

    def before_execution(self, **kwargs):
        pass # do not add anything to the history or output
    
    def after_execution(self, response, **kwargs):
        pass

# From tools/task_done.py
def before_execution(self, **kwargs):
        pass

# From tools/task_done.py
def after_execution(self, response, **kwargs):
        pass

import contextlib
import ast
import shlex
from io import StringIO
import time
from python.helpers import messages
from python.helpers.shell_local import LocalInteractiveSession
from python.helpers.shell_ssh import SSHInteractiveSession
from python.helpers.docker import DockerContainerManager

# From tools/code_execution_tool.py
class State:
    shell: LocalInteractiveSession | SSHInteractiveSession
    docker: DockerContainerManager | None

# From tools/code_execution_tool.py
class CodeExecution(Tool):

    def execute(self,**kwargs):

        if self.agent.handle_intervention(): return Response(message="", break_loop=False)  # wait for intervention and handle it, if paused
        
        self.prepare_state()

        # os.chdir(files.get_abs_path("./work_dir")) #change CWD to work_dir
        
        runtime = self.args["runtime"].lower().strip()
        if runtime == "python":
            response = self.execute_python_code(self.args["code"])
        elif runtime == "nodejs":
            response = self.execute_nodejs_code(self.args["code"])
        elif runtime == "terminal":
            response = self.execute_terminal_command(self.args["code"])
        elif runtime == "output":
            response = self.get_terminal_output()
        else:
            response = files.read_file("./prompts/fw.code_runtime_wrong.md", runtime=runtime)

        if not response: response = files.read_file("./prompts/fw.code_no_output.md")
        return Response(message=response, break_loop=False)

    def after_execution(self, response, **kwargs):
        msg_response = files.read_file("./prompts/fw.tool_response.md", tool_name=self.name, tool_response=response.message)
        self.agent.append_message(msg_response, human=True)

    def prepare_state(self):
        self.state = self.agent.get_data("cot_state")
        if not self.state:

            #initialize docker container if execution in docker is configured
            if self.agent.config.code_exec_docker_enabled:
                docker = DockerContainerManager(name=self.agent.config.code_exec_docker_name, image=self.agent.config.code_exec_docker_image, ports=self.agent.config.code_exec_docker_ports, volumes=self.agent.config.code_exec_docker_volumes)
                docker.start_container()
            else: docker = None

            #initialize local or remote interactive shell insterface
            if self.agent.config.code_exec_ssh_enabled:
                shell = SSHInteractiveSession(self.agent.config.code_exec_ssh_addr,self.agent.config.code_exec_ssh_port,self.agent.config.code_exec_ssh_user,self.agent.config.code_exec_ssh_pass)
            else: shell = LocalInteractiveSession()
                
            self.state = State(shell=shell,docker=docker)
            shell.connect()
        self.agent.set_data("cot_state", self.state)
    
    def execute_python_code(self, code):
        escaped_code = shlex.quote(code)
        command = f'python3 -c {escaped_code}'
        return self.terminal_session(command)

    def execute_nodejs_code(self, code):
        escaped_code = shlex.quote(code)
        command = f'node -e {escaped_code}'
        return self.terminal_session(command)

    def execute_terminal_command(self, command):
        return self.terminal_session(command)

    def terminal_session(self, command):

        if self.agent.handle_intervention(): return ""  # wait for intervention and handle it, if paused
       
        self.state.shell.send_command(command)

        PrintStyle(background_color="white",font_color="#1B4F72",bold=True).print(f"{self.agent.agent_name} code execution output:")
        return self.get_terminal_output()

    def get_terminal_output(self):
        idle=0
        while True:       
            time.sleep(0.1)  # Wait for some output to be generated
            full_output, partial_output = self.state.shell.read_output()

            if self.agent.handle_intervention(): return full_output  # wait for intervention and handle it, if paused
        
            if partial_output:
                PrintStyle(font_color="#85C1E9").stream(partial_output)
                idle=0    
            else:
                idle+=1
                if ( full_output and idle > 30 ) or ( not full_output and idle > 100 ): return full_output

# From tools/code_execution_tool.py
def prepare_state(self):
        self.state = self.agent.get_data("cot_state")
        if not self.state:

            #initialize docker container if execution in docker is configured
            if self.agent.config.code_exec_docker_enabled:
                docker = DockerContainerManager(name=self.agent.config.code_exec_docker_name, image=self.agent.config.code_exec_docker_image, ports=self.agent.config.code_exec_docker_ports, volumes=self.agent.config.code_exec_docker_volumes)
                docker.start_container()
            else: docker = None

            #initialize local or remote interactive shell insterface
            if self.agent.config.code_exec_ssh_enabled:
                shell = SSHInteractiveSession(self.agent.config.code_exec_ssh_addr,self.agent.config.code_exec_ssh_port,self.agent.config.code_exec_ssh_user,self.agent.config.code_exec_ssh_pass)
            else: shell = LocalInteractiveSession()
                
            self.state = State(shell=shell,docker=docker)
            shell.connect()
        self.agent.set_data("cot_state", self.state)

# From tools/code_execution_tool.py
def execute_python_code(self, code):
        escaped_code = shlex.quote(code)
        command = f'python3 -c {escaped_code}'
        return self.terminal_session(command)

# From tools/code_execution_tool.py
def execute_nodejs_code(self, code):
        escaped_code = shlex.quote(code)
        command = f'node -e {escaped_code}'
        return self.terminal_session(command)

# From tools/code_execution_tool.py
def execute_terminal_command(self, command):
        return self.terminal_session(command)

# From tools/code_execution_tool.py
def terminal_session(self, command):

        if self.agent.handle_intervention(): return ""  # wait for intervention and handle it, if paused
       
        self.state.shell.send_command(command)

        PrintStyle(background_color="white",font_color="#1B4F72",bold=True).print(f"{self.agent.agent_name} code execution output:")
        return self.get_terminal_output()

# From tools/code_execution_tool.py
def get_terminal_output(self):
        idle=0
        while True:       
            time.sleep(0.1)  # Wait for some output to be generated
            full_output, partial_output = self.state.shell.read_output()

            if self.agent.handle_intervention(): return full_output  # wait for intervention and handle it, if paused
        
            if partial_output:
                PrintStyle(font_color="#85C1E9").stream(partial_output)
                idle=0    
            else:
                idle+=1
                if ( full_output and idle > 30 ) or ( not full_output and idle > 100 ): return full_output


# From tools/response.py
class ResponseTool(Tool):

    def execute(self,**kwargs):
        self.agent.set_data("timeout", self.agent.config.response_timeout_seconds)
        return Response(message=self.args["text"], break_loop=True)

    def before_execution(self, **kwargs):
        pass # do not add anything to the history or output
    
    def after_execution(self, response, **kwargs):
        pass

from abc import abstractmethod
from typing import TypedDict

# From helpers/tool.py
class Response:
    def __init__(self, message: str, break_loop: bool) -> None:
        self.message = message
        self.break_loop = break_loop

# From helpers/tool.py
class Tool:

    def __init__(self, agent: Agent, name: str, args: dict[str,str], message: str, **kwargs) -> None:
        self.agent = agent
        self.name = name
        self.args = args
        self.message = message

    @abstractmethod
    def execute(self,**kwargs) -> Response:
        pass

    def before_execution(self, **kwargs):
        if self.agent.handle_intervention(): return # wait for intervention and handle it, if paused
        PrintStyle(font_color="#1B4F72", padding=True, background_color="white", bold=True).print(f"{self.agent.agent_name}: Using tool '{self.name}':")
        if self.args and isinstance(self.args, dict):
            for key, value in self.args.items():
                PrintStyle(font_color="#85C1E9", bold=True).stream(self.nice_key(key)+": ")
                PrintStyle(font_color="#85C1E9", padding=isinstance(value,str) and "\n" in value).stream(value)
                PrintStyle().print()
                    
    def after_execution(self, response: Response, **kwargs):
        text = messages.truncate_text(response.message.strip(), self.agent.config.max_tool_response_length)
        msg_response = files.read_file("./prompts/fw.tool_response.md", tool_name=self.name, tool_response=text)
        if self.agent.handle_intervention(): return # wait for intervention and handle it, if paused
        self.agent.append_message(msg_response, human=True)
        PrintStyle(font_color="#1B4F72", background_color="white", padding=True, bold=True).print(f"{self.agent.agent_name}: Response from tool '{self.name}':")
        PrintStyle(font_color="#85C1E9").print(response.message)

    def nice_key(self, key:str):
        words = key.split('_')
        words = [words[0].capitalize()] + [word.lower() for word in words[1:]]
        result = ' '.join(words)
        return result

# From helpers/tool.py
def nice_key(self, key:str):
        words = key.split('_')
        words = [words[0].capitalize()] + [word.lower() for word in words[1:]]
        result = ' '.join(words)
        return result

