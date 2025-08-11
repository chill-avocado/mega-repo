# Merged file for agent_frameworks/agent
# This file contains code merged from multiple repositories

import asyncio
import logging
import os
import sys
from autogen_core import PROTOBUF_DATA_CONTENT_TYPE
from autogen_core import AgentId
from autogen_core import DefaultSubscription
from autogen_core import DefaultTopicId
from autogen_core import TypeSubscription
from autogen_core import try_get_known_serializers_for_type
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from dotenv import load_dotenv
from protos.agent_events_pb2 import NewMessageReceived
from protos.agent_events_pb2 import Output
from user_input import UserProxy

import grpc
import warnings
from grpc._utilities import first_version_is_lower

from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import runtime_version
from google.protobuf import symbol_database
from google.protobuf.internal import builder

import tempfile
from typing import List
from typing import Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.conditions import SourceMatchTermination
from autogen_agentchat.conditions import StopMessageTermination
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.conditions import TimeoutTermination
from autogen_agentchat.conditions import TokenUsageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.teams import Swarm
from autogen_core import ComponentModel
from autogen_core.models import ModelInfo
from autogen_core.tools import StaticWorkbench
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._openai_client import AzureOpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.tools.mcp import McpWorkbench
from autogen_ext.tools.mcp import StdioServerParams
from autogen_ext.tools.mcp import StreamableHttpServerParams
from autogenstudio.datamodel import GalleryComponents
from autogenstudio.datamodel import GalleryConfig
from autogenstudio.datamodel import GalleryMetadata
from  import tools
import json

# From gallery/builder.py
class GalleryBuilder:
    """Enhanced builder class for creating AutoGen component galleries with custom labels."""

    def __init__(self, id: str, name: str, url: Optional[str] = None):
        self.id = id
        self.name = name
        self.url: Optional[str] = url
        self.teams: List[ComponentModel] = []
        self.agents: List[ComponentModel] = []
        self.models: List[ComponentModel] = []
        self.tools: List[ComponentModel] = []
        self.terminations: List[ComponentModel] = []
        self.workbenches: List[ComponentModel] = []

        # Default metadata
        self.metadata = GalleryMetadata(
            author="AutoGen Team",
            version="1.0.0",
            description="",
            tags=[],
            license="MIT",
            category="conversation",
        )

    def _update_component_metadata(
        self, component: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> ComponentModel:
        """Helper method to update component metadata."""
        if label is not None:
            component.label = label
        if description is not None:
            component.description = description
        return component

    def set_metadata(
        self,
        author: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        license: Optional[str] = None,
        category: Optional[str] = None,
    ) -> "GalleryBuilder":
        """Update gallery metadata."""
        if author:
            self.metadata.author = author
        if version:
            self.metadata.version = version
        if description:
            self.metadata.description = description
        if tags:
            self.metadata.tags = tags
        if license:
            self.metadata.license = license
        if category:
            self.metadata.category = category
        return self

    def add_team(
        self, team: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a team component to the gallery with optional custom label and description."""
        self.teams.append(self._update_component_metadata(team, label, description))
        return self

    def add_agent(
        self, agent: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add an agent component to the gallery with optional custom label and description."""
        self.agents.append(self._update_component_metadata(agent, label, description))
        return self

    def add_model(
        self, model: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a model component to the gallery with optional custom label and description."""
        self.models.append(self._update_component_metadata(model, label, description))
        return self

    def add_tool(
        self, tool: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a tool component to the gallery with optional custom label and description."""
        self.tools.append(self._update_component_metadata(tool, label, description))
        return self

    def add_termination(
        self, termination: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a termination condition component with optional custom label and description."""
        self.terminations.append(self._update_component_metadata(termination, label, description))
        return self

    def add_workbench(
        self, workbench: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a workbench component to the gallery with optional custom label and description."""
        self.workbenches.append(self._update_component_metadata(workbench, label, description))
        return self

    def build(self) -> GalleryConfig:
        """Build and return the complete gallery."""
        # Update timestamps
        # self.metadata.updated_at = datetime.now()

        return GalleryConfig(
            id=self.id,
            name=self.name,
            url=self.url,
            metadata=self.metadata,
            components=GalleryComponents(
                teams=self.teams,
                agents=self.agents,
                models=self.models,
                tools=self.tools,
                terminations=self.terminations,
                workbenches=self.workbenches,
            ),
        )

# From gallery/builder.py
def create_default_gallery() -> GalleryConfig:
    """Create a default gallery with all components including calculator and web surfer teams."""

    # model clients require API keys to be set in the environment or passed in
    # as arguments. For testing purposes, we set them to "test" if not already set.
    for key in ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(key):
            os.environ[key] = "test"

    # url = "https://raw.githubusercontent.com/microsoft/autogen/refs/heads/main/python/packages/autogen-studio/autogenstudio/gallery/default.json"
    builder = GalleryBuilder(id="gallery_default", name="Default Component Gallery")

    # Set metadata
    builder.set_metadata(
        description="A default gallery containing basic components for human-in-loop conversations",
        tags=["human-in-loop", "assistant", "web agents"],
        category="conversation",
    )

    # Create base model client
    base_model = OpenAIChatCompletionClient(model="gpt-4o-mini")
    builder.add_model(base_model.dump_component(), label="OpenAI GPT-4o Mini", description="OpenAI GPT-4o-mini")
    # Create Mistral vllm model
    mistral_vllm_model = OpenAIChatCompletionClient(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        base_url="http://localhost:1234/v1",
        model_info=ModelInfo(
            vision=False, function_calling=True, json_output=False, family="unknown", structured_output=False
        ),
    )
    builder.add_model(
        mistral_vllm_model.dump_component(),
        label="Mistral-7B Local",
        description="Local Mistral-7B model client for instruction-based generation (Ollama, LMStudio).",
    )

    anthropic_model = AnthropicChatCompletionClient(model="claude-3-7-sonnet-20250219")
    builder.add_model(
        anthropic_model.dump_component(),
        label="Anthropic Claude-3-7",
        description="Anthropic Claude-3 model client.",
    )

    # create an azure mode
    az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment="{your-azure-deployment}",
        model="gpt-4o-mini",
        api_version="2024-06-01",
        azure_endpoint="https://{your-custom-endpoint}.openai.azure.com/",
        api_key="test",
    )
    builder.add_model(
        az_model_client.dump_component(),
        label="AzureOpenAI GPT-4o-mini",
        description="GPT-4o Mini Azure OpenAI model client.",
    )

    builder.add_tool(
        tools.calculator_tool.dump_component(),
        label="Calculator Tool",
        description="A tool that performs basic arithmetic operations (addition, subtraction, multiplication, division).",
    )

    # Create calculator assistant agent
    calc_assistant = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant. Solve tasks carefully. When done, say TERMINATE.",
        model_client=base_model,
        tools=[tools.calculator_tool],
    )

    builder.add_agent(
        calc_assistant.dump_component(), description="An agent that provides assistance with ability to use tools."
    )

    # Create termination conditions
    calc_text_term = TextMentionTermination(text="TERMINATE")
    calc_max_term = MaxMessageTermination(max_messages=10)
    calc_or_term = calc_text_term | calc_max_term

    builder.add_termination(calc_text_term.dump_component())
    builder.add_termination(calc_max_term.dump_component())
    builder.add_termination(
        calc_or_term.dump_component(),
        label="OR Termination",
        description="Termination condition that ends the conversation when either a message contains 'TERMINATE' or the maximum number of messages is reached.",
    )

    # Add examples of new termination conditions

    # StopMessageTermination - terminates when a StopMessage is received
    stop_msg_term = StopMessageTermination()
    builder.add_termination(
        stop_msg_term.dump_component(),
        label="Stop Message Termination",
        description="Terminates the conversation when a StopMessage is received from any agent.",
    )

    # TokenUsageTermination - terminates based on token usage limits
    token_usage_term = TokenUsageTermination(max_total_token=1000, max_prompt_token=800, max_completion_token=200)
    builder.add_termination(
        token_usage_term.dump_component(),
        label="Token Usage Termination",
        description="Terminates the conversation when token usage limits are reached (1000 total, 800 prompt, 200 completion).",
    )

    # TimeoutTermination - terminates after a specified duration
    timeout_term = TimeoutTermination(timeout_seconds=300)  # 5 minutes
    builder.add_termination(
        timeout_term.dump_component(),
        label="Timeout Termination",
        description="Terminates the conversation after 5 minutes (300 seconds) have elapsed.",
    )

    # HandoffTermination - terminates when handoff to specific target occurs
    handoff_term = HandoffTermination(target="user_proxy")
    builder.add_termination(
        handoff_term.dump_component(),
        label="Handoff Termination",
        description="Terminates the conversation when a handoff to 'user_proxy' is detected.",
    )

    # SourceMatchTermination - terminates when specific sources respond
    source_match_term = SourceMatchTermination(sources=["assistant_agent", "critic_agent"])
    builder.add_termination(
        source_match_term.dump_component(),
        label="Source Match Termination",
        description="Terminates the conversation when either 'assistant_agent' or 'critic_agent' responds.",
    )

    # TextMessageTermination - terminates on TextMessage from specific source
    text_msg_term = TextMessageTermination(source="assistant_agent")
    builder.add_termination(
        text_msg_term.dump_component(),
        label="Text Message Termination",
        description="Terminates the conversation when a TextMessage is received from 'assistant_agent'.",
    )

    # Create a complex termination combining multiple conditions
    complex_term = (token_usage_term | timeout_term) & (calc_text_term | stop_msg_term)
    builder.add_termination(
        complex_term.dump_component(),
        label="Complex Termination",
        description="Complex termination: (token usage OR timeout) AND (text mention 'TERMINATE' OR stop message).",
    )

    # Create calculator team
    calc_team = RoundRobinGroupChat(participants=[calc_assistant], termination_condition=calc_or_term)
    builder.add_team(
        calc_team.dump_component(),
        label="RoundRobin Team",
        description="A single AssistantAgent (with a calculator tool) in a RoundRobinGroupChat team. ",
    )

    critic_agent = AssistantAgent(
        name="critic_agent",
        system_message="You are a helpful assistant. Critique the assistant's output and suggest improvements.",
        description="an agent that critiques and improves the assistant's output",
        model_client=base_model,
    )
    selector_default_team = SelectorGroupChat(
        participants=[calc_assistant, critic_agent], termination_condition=calc_or_term, model_client=base_model
    )
    builder.add_team(
        selector_default_team.dump_component(),
        label="Selector Team",
        description="A team with 2 agents - an AssistantAgent (with a calculator tool) and a CriticAgent in a SelectorGroupChat team.",
    )

    # Create Swarm team - agents with handoff capabilities
    # Alice agent with handoff to Bob
    alice_agent = AssistantAgent(
        name="Alice",
        system_message="You are Alice, a helpful assistant. You specialize in general questions. If someone asks about technical topics or needs detailed analysis, hand off to Bob by saying 'Let me hand this over to Bob for a detailed analysis.'",
        model_client=base_model,
        handoffs=["Bob"],
    )

    # Bob agent with handoff back to Alice
    bob_agent = AssistantAgent(
        name="Bob",
        system_message="You are Bob, a technical specialist. You handle detailed technical analysis. If the conversation becomes general or the user needs basic assistance, hand off to Alice by saying 'Let me hand this back to Alice for general assistance.'",
        model_client=base_model,
        handoffs=["Alice"],
    )

    # Create simple Swarm team with handoff-based conversation
    swarm_team = Swarm(participants=[alice_agent, bob_agent], termination_condition=calc_or_term)
    builder.add_team(
        swarm_team.dump_component(),
        label="Swarm Team",
        description="A team with 2 agents (Alice and Bob) that use handoff messages to transfer conversation control between agents based on expertise.",
    )

    # Create web surfer agent
    websurfer_agent = MultimodalWebSurfer(
        name="websurfer_agent",
        description="an agent that solves tasks by browsing the web",
        model_client=base_model,
        headless=True,
    )
    builder.add_agent(
        websurfer_agent.dump_component(),
        label="Web Surfer Agent",
        description="An agent that solves tasks by browsing the web using a headless browser.",
    )

    # Create verification assistant
    verification_assistant = AssistantAgent(
        name="assistant_agent",
        description="an agent that verifies and summarizes information",
        system_message="You are a task verification assistant who is working with a web surfer agent to solve tasks. At each point, check if the task has been completed as requested by the user. If the websurfer_agent responds and the task has not yet been completed, respond with what is left to do and then say 'keep going'. If and only when the task has been completed, summarize and present a final answer that directly addresses the user task in detail and then respond with TERMINATE.",
        model_client=base_model,
    )
    builder.add_agent(
        verification_assistant.dump_component(),
        label="Verification Assistant",
        description="an agent that verifies and summarizes information",
    )

    # Create user proxy
    web_user_proxy = UserProxyAgent(
        name="user_proxy",
        description="a human user that should be consulted only when the assistant_agent is unable to verify the information provided by the websurfer_agent",
    )
    builder.add_agent(web_user_proxy.dump_component())

    # Create web surfer team termination conditions
    web_max_term = MaxMessageTermination(max_messages=20)
    web_text_term = TextMentionTermination(text="TERMINATE")
    web_termination = web_max_term | web_text_term

    # Create web surfer team
    selector_prompt = """You are the cordinator of role play game. The following roles are available:
{roles}. Given a task, the websurfer_agent will be tasked to address it by browsing the web and providing information.  The assistant_agent will be tasked with verifying the information provided by the websurfer_agent and summarizing the information to present a final answer to the user. If the task  needs assistance from a human user (e.g., providing feedback, preferences, or the task is stalled), you should select the user_proxy role to provide the necessary information.

Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role."""

    websurfer_team = SelectorGroupChat(
        participants=[websurfer_agent, verification_assistant, web_user_proxy],
        selector_prompt=selector_prompt,
        model_client=base_model,
        termination_condition=web_termination,
    )

    builder.add_team(
        websurfer_team.dump_component(),
        label="Web Agent Team (Operator)",
        description="A team with 3 agents - a Web Surfer agent that can browse the web, a Verification Assistant that verifies and summarizes information, and a User Proxy that provides human feedback when needed.",
    )

    builder.add_tool(
        tools.generate_image_tool.dump_component(),
        label="Image Generation Tool",
        description="A tool that generates images based on a text description using OpenAI's DALL-E model. Note: Requires OpenAI API key to function.",
    )

    builder.add_tool(
        tools.fetch_webpage_tool.dump_component(),
        label="Fetch Webpage Tool",
        description="A tool that fetches the content of a webpage and converts it to markdown. Requires the requests and beautifulsoup4 library to function.",
    )

    builder.add_tool(
        tools.bing_search_tool.dump_component(),
        label="Bing Search Tool",
        description="A tool that performs Bing searches using the Bing Web Search API. Requires the requests library, BING_SEARCH_KEY env variable to function.",
    )

    builder.add_tool(
        tools.google_search_tool.dump_component(),
        label="Google Search Tool",
        description="A tool that performs Google searches using the Google Custom Search API. Requires the requests library, [GOOGLE_API_KEY, GOOGLE_CSE_ID] to be set,  env variable to function.",
    )

    code_executor = LocalCommandLineCodeExecutor(work_dir=".coding", timeout=360)
    code_execution_tool = PythonCodeExecutionTool(code_executor)
    builder.add_tool(
        code_execution_tool.dump_component(),
        label="Python Code Execution Tool",
        description="A tool that executes Python code in a local environment.",
    )

    # Create deep research agent
    model_client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0.7)

    research_assistant = AssistantAgent(
        name="research_assistant",
        description="A research assistant that performs web searches and analyzes information",
        model_client=model_client,
        tools=[tools.google_search_tool, tools.fetch_webpage_tool],
        system_message="""You are a research assistant focused on finding accurate information.
        Use the google_search tool to find relevant information.
        Break down complex queries into specific search terms.
        Always verify information across multiple sources when possible.
        When you find relevant information, explain why it's relevant and how it connects to the query. When you get feedback from the a verifier agent, use your tools to act on the feedback and make progress.""",
    )

    verifier = AssistantAgent(
        name="verifier",
        description="A verification specialist who ensures research quality and completeness",
        model_client=model_client,
        system_message="""You are a research verification specialist.
        Your role is to:
        1. Verify that search queries are effective and suggest improvements if needed
        2. Explore drill downs where needed e.g, if the answer is likely in a link in the returned search results, suggest clicking on the link
        3. Suggest additional angles or perspectives to explore. Be judicious in suggesting new paths to avoid scope creep or wasting resources, if the task appears to be addressed and we can provide a report, do this and respond with "TERMINATE".
        4. Track progress toward answering the original question
        5. When the research is complete, provide a detailed summary in markdown format. For incomplete research, end your message with "CONTINUE RESEARCH". For complete research, end your message with APPROVED.
        Your responses should be structured as:
        - Progress Assessment
        - Gaps/Issues (if any)
        - Suggestions (if needed)
        - Next Steps or Final Summary""",
    )

    summary_agent = AssistantAgent(
        name="summary_agent",
        description="A summary agent that provides a detailed markdown summary of the research as a report to the user.",
        model_client=model_client,
        system_message="""You are a summary agent. Your role is to provide a detailed markdown summary of the research as a report to the user. Your report should have a reasonable title that matches the research question and should summarize the key details in the results found in natural an actionable manner. The main results/answer should be in the first paragraph. Where reasonable, your report should have clear comparison tables that drive critical insights. Most importantly, you should have a reference section and cite the key sources (where available) for facts obtained INSIDE THE MAIN REPORT. Also, where appropriate, you may add images if available that illustrate concepts needed for the summary.
        Your report should end with the word "TERMINATE" to signal the end of the conversation.""",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=30)

    selector_prompt = """You are coordinating a research team by selecting the team member to speak/act next. The following team member roles are available:
    {roles}.
    The research_assistant performs searches and analyzes information.
    The verifier evaluates progress and ensures completeness.
    The summary_agent provides a detailed markdown summary of the research as a report to the user.

    Given the current context, select the most appropriate next speaker.
    The research_assistant should search and analyze.
    The verifier should evaluate progress and guide the research (select this role is there is a need to verify/evaluate progress). You should ONLY select the summary_agent role if the research is complete and it is time to generate a report.

    Base your selection on:
    1. Current stage of research
    2. Last speaker's findings or suggestions
    3. Need for verification vs need for new information
    Read the following conversation. Then select the next role from {participants} to play. Only return the role.

    {history}

    Read the above conversation. Then select the next role from {participants} to play. ONLY RETURN THE ROLE."""

    deep_research_team = SelectorGroupChat(
        participants=[research_assistant, verifier, summary_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,
    )

    builder.add_team(
        deep_research_team.dump_component(),
        label="Deep Research Team",
        description="A team with 3 agents - a Research Assistant that performs web searches and analyzes information, a Verifier that ensures research quality and completeness, and a Summary Agent that provides a detailed markdown summary of the research as a report to the user.",
    )

    # Add workbenches to the gallery

    # Create a static workbench with basic tools
    static_workbench = StaticWorkbench(tools=[tools.calculator_tool, tools.fetch_webpage_tool])
    builder.add_workbench(
        static_workbench.dump_component(),
        label="Basic Tools Workbench",
        description="A static workbench containing basic tools like calculator and webpage fetcher for common tasks.",
    )

    # Create an MCP workbench for fetching web content using mcp-server-fetch
    # Note: This requires uv to be installed (comes with uv package manager)
    fetch_server_params = StdioServerParams(
        command="uv",
        args=["tool", "run", "mcp-server-fetch"],
        read_timeout_seconds=60,
    )
    mcp_workbench = McpWorkbench(server_params=fetch_server_params)
    builder.add_workbench(
        mcp_workbench.dump_component(),
        label="MCP Fetch Workbench",
        description="An MCP workbench that provides web content fetching capabilities using the mcp-server-fetch MCP server. Allows agents to fetch and read content from web pages and APIs.",
    )

    # Create an MCP workbench with StreamableHttpServerParams for HTTP-based MCP servers
    # Note: This is an example - adjust URL and authentication as needed
    streamable_server_params = StreamableHttpServerParams(
        url="http://localhost:8005/mcp",
        headers={"Authorization": "Bearer your-api-key", "Content-Type": "application/json"},
        timeout=30,
        sse_read_timeout=60 * 5,
        terminate_on_close=True,
    )
    streamable_mcp_workbench = McpWorkbench(server_params=streamable_server_params)
    builder.add_workbench(
        streamable_mcp_workbench.dump_component(),
        label="MCP Streamable HTTP Workbench",
        description="An MCP workbench that connects to HTTP-based MCP servers using Server-Sent Events (SSE). Suitable for cloud-hosted MCP services and custom HTTP MCP implementations.",
    )

    # Create an MCP workbench for filesystem operations
    # Note: This requires npx to be installed and allows access to specified directories

    # Use cross-platform paths for filesystem access
    user_home = os.path.expanduser("~")
    temp_dir = tempfile.gettempdir()  # Cross-platform temp directory

    filesystem_server_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", user_home, temp_dir],
        read_timeout_seconds=60,
    )
    filesystem_mcp_workbench = McpWorkbench(server_params=filesystem_server_params)
    builder.add_workbench(
        filesystem_mcp_workbench.dump_component(),
        label="MCP Filesystem Workbench",
        description="An MCP workbench that provides filesystem access capabilities using the @modelcontextprotocol/server-filesystem MCP server. Allows agents to read, write, and manage files and directories within specified allowed paths.",
    )

    # Create an MCP workbench for testing with everything server
    # Note: This requires npx to be installed and provides comprehensive MCP testing tools
    everything_server_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
        read_timeout_seconds=60,
    )
    everything_mcp_workbench = McpWorkbench(server_params=everything_server_params)
    builder.add_workbench(
        everything_mcp_workbench.dump_component(),
        label="MCP Test Server",
        description="An MCP workbench that provides comprehensive testing tools using the @modelcontextprotocol/server-everything MCP server. Includes various tools for testing MCP functionality, protocol features, and capabilities.",
    )

    return builder.build()

# From gallery/builder.py
def create_default_lite_team():
    """Create a simple default team for lite mode - a basic assistant with calculator tool."""
    import json
    import os
    import tempfile

    # model clients require API keys to be set in the environment or passed in
    # as arguments. For testing purposes, we set them to "test" if not already set.
    for key in ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(key):
            os.environ[key] = "test"

    # Create base model client
    base_model = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Create assistant agent with calculator tool
    assistant = AssistantAgent(
        name="assistant",
        model_client=base_model,
        tools=[tools.calculator_tool],
    )

    # Create termination condition
    termination = TextMentionTermination(text="TERMINATE") | MaxMessageTermination(max_messages=5)

    # Create simple round robin team
    team = RoundRobinGroupChat(participants=[assistant], termination_condition=termination)

    # Create temporary file with team data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(team.dump_component().model_dump(), f, indent=2)
        return f.name

# From gallery/builder.py
def set_metadata(
        self,
        author: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        license: Optional[str] = None,
        category: Optional[str] = None,
    ) -> "GalleryBuilder":
        """Update gallery metadata."""
        if author:
            self.metadata.author = author
        if version:
            self.metadata.version = version
        if description:
            self.metadata.description = description
        if tags:
            self.metadata.tags = tags
        if license:
            self.metadata.license = license
        if category:
            self.metadata.category = category
        return self

# From gallery/builder.py
def add_team(
        self, team: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a team component to the gallery with optional custom label and description."""
        self.teams.append(self._update_component_metadata(team, label, description))
        return self

# From gallery/builder.py
def add_agent(
        self, agent: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add an agent component to the gallery with optional custom label and description."""
        self.agents.append(self._update_component_metadata(agent, label, description))
        return self

# From gallery/builder.py
def add_model(
        self, model: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a model component to the gallery with optional custom label and description."""
        self.models.append(self._update_component_metadata(model, label, description))
        return self

# From gallery/builder.py
def add_tool(
        self, tool: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a tool component to the gallery with optional custom label and description."""
        self.tools.append(self._update_component_metadata(tool, label, description))
        return self

# From gallery/builder.py
def add_termination(
        self, termination: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a termination condition component with optional custom label and description."""
        self.terminations.append(self._update_component_metadata(termination, label, description))
        return self

# From gallery/builder.py
def add_workbench(
        self, workbench: ComponentModel, label: Optional[str] = None, description: Optional[str] = None
    ) -> "GalleryBuilder":
        """Add a workbench component to the gallery with optional custom label and description."""
        self.workbenches.append(self._update_component_metadata(workbench, label, description))
        return self

# From gallery/builder.py
def build(self) -> GalleryConfig:
        """Build and return the complete gallery."""
        # Update timestamps
        # self.metadata.updated_at = datetime.now()

        return GalleryConfig(
            id=self.id,
            name=self.name,
            url=self.url,
            metadata=self.metadata,
            components=GalleryComponents(
                teams=self.teams,
                agents=self.agents,
                models=self.models,
                tools=self.tools,
                terminations=self.terminations,
                workbenches=self.workbenches,
            ),
        )

import uuid
from _agent_id import AgentId
from _agent_type import AgentType
from _subscription import Subscription
from _topic import TopicId
from exceptions import CantHandleException

# From autogen_core/_type_prefix_subscription.py
class TypePrefixSubscription(Subscription):
    """This subscription matches on topics based on a prefix of the type and maps to agents using the source of the topic as the agent key.

    This subscription causes each source to have its own agent instance.

    Example:

        .. code-block:: python

            from autogen_core import TypePrefixSubscription

            subscription = TypePrefixSubscription(topic_type_prefix="t1", agent_type="a1")

        In this case:

        - A topic_id with type `t1` and source `s1` will be handled by an agent of type `a1` with key `s1`
        - A topic_id with type `t1` and source `s2` will be handled by an agent of type `a1` with key `s2`.
        - A topic_id with type `t1SUFFIX` and source `s2` will be handled by an agent of type `a1` with key `s2`.

    Args:
        topic_type_prefix (str): Topic type prefix to match against
        agent_type (str): Agent type to handle this subscription
    """

    def __init__(self, topic_type_prefix: str, agent_type: str | AgentType, id: str | None = None):
        self._topic_type_prefix = topic_type_prefix
        if isinstance(agent_type, AgentType):
            self._agent_type = agent_type.type
        else:
            self._agent_type = agent_type
        self._id = id or str(uuid.uuid4())

    @property
    def id(self) -> str:
        return self._id

    @property
    def topic_type_prefix(self) -> str:
        return self._topic_type_prefix

    @property
    def agent_type(self) -> str:
        return self._agent_type

    def is_match(self, topic_id: TopicId) -> bool:
        return topic_id.type.startswith(self._topic_type_prefix)

    def map_to_agent(self, topic_id: TopicId) -> AgentId:
        if not self.is_match(topic_id):
            raise CantHandleException("TopicId does not match the subscription")

        return AgentId(type=self._agent_type, key=topic_id.source)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypePrefixSubscription):
            return False

        return self.id == other.id or (
            self.agent_type == other.agent_type and self.topic_type_prefix == other.topic_type_prefix
        )

# From autogen_core/_type_prefix_subscription.py
def id(self) -> str:
        return self._id

# From autogen_core/_type_prefix_subscription.py
def topic_type_prefix(self) -> str:
        return self._topic_type_prefix

# From autogen_core/_type_prefix_subscription.py
def agent_type(self) -> str:
        return self._agent_type

# From autogen_core/_type_prefix_subscription.py
def is_match(self, topic_id: TopicId) -> bool:
        return topic_id.type.startswith(self._topic_type_prefix)

# From autogen_core/_type_prefix_subscription.py
def map_to_agent(self, topic_id: TopicId) -> AgentId:
        if not self.is_match(topic_id):
            raise CantHandleException("TopicId does not match the subscription")

        return AgentId(type=self._agent_type, key=topic_id.source)

from dataclasses import dataclass

# From autogen_core/_agent_type.py
class AgentType:
    type: str
    """String representation of this agent type."""

from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping
from typing import Protocol
from typing import runtime_checkable
from _agent_metadata import AgentMetadata
from _message_context import MessageContext
from _agent_runtime import AgentRuntime

# From autogen_core/_agent.py
class Agent(Protocol):
    @property
    def metadata(self) -> AgentMetadata:
        """Metadata of the agent."""
        ...

    @property
    def id(self) -> AgentId:
        """ID of the agent."""
        ...

    async def bind_id_and_runtime(self, id: AgentId, runtime: "AgentRuntime") -> None:
        """Function used to bind an Agent instance to an `AgentRuntime`.

        Args:
            agent_id (AgentId): ID of the agent.
            runtime (AgentRuntime): AgentRuntime instance to bind the agent to.
        """
        ...

    async def on_message(self, message: Any, ctx: MessageContext) -> Any:
        """Message handler for the agent. This should only be called by the runtime, not by other agents.

        Args:
            message (Any): Received message. Type is one of the types in `subscriptions`.
            ctx (MessageContext): Context of the message.

        Returns:
            Any: Response to the message. Can be None.

        Raises:
            asyncio.CancelledError: If the message was cancelled.
            CantHandleException: If the agent cannot handle the message.
        """
        ...

    async def save_state(self) -> Mapping[str, Any]:
        """Save the state of the agent. The result must be JSON serializable."""
        ...

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load in the state of the agent obtained from `save_state`.

        Args:
            state (Mapping[str, Any]): State of the agent. Must be JSON serializable.
        """

        ...

    async def close(self) -> None:
        """Called when the runtime is closed"""
        ...

# From autogen_core/_agent.py
def metadata(self) -> AgentMetadata:
        """Metadata of the agent."""
        ...

from __future__ import annotations
import inspect
from asyncio import CancelledError
from asyncio import Future
from asyncio import Queue
from asyncio import Task
from collections.abc import Sequence
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import ParamSpec
from typing import Set
from typing import Type
from typing import TypeVar
from typing import cast
from opentelemetry.trace import TracerProvider
from logging import AgentConstructionExceptionEvent
from logging import DeliveryStage
from logging import MessageDroppedEvent
from logging import MessageEvent
from logging import MessageHandlerExceptionEvent
from logging import MessageKind
from _agent import Agent
from _agent_instantiation import AgentInstantiationContext
from _cancellation_token import CancellationToken
from _intervention import DropMessage
from _intervention import InterventionHandler
from _message_handler_context import MessageHandlerContext
from _runtime_impl_helpers import SubscriptionManager
from _runtime_impl_helpers import get_impl
from _serialization import JSON_DATA_CONTENT_TYPE
from _serialization import MessageSerializer
from _serialization import SerializationRegistry
from _telemetry import EnvelopeMetadata
from _telemetry import MessageRuntimeTracingConfig
from _telemetry import TraceHelper
from _telemetry import get_telemetry_envelope_metadata
from exceptions import MessageDroppedException
from asyncio import QueueShutDown
from _queue import Queue
from _queue import QueueShutDown

# From autogen_core/_single_threaded_agent_runtime.py
class PublishMessageEnvelope:
    """A message envelope for publishing messages to all agents that can handle
    the message of the type T."""

    message: Any
    cancellation_token: CancellationToken
    sender: AgentId | None
    topic_id: TopicId
    metadata: EnvelopeMetadata | None = None
    message_id: str

# From autogen_core/_single_threaded_agent_runtime.py
class SendMessageEnvelope:
    """A message envelope for sending a message to a specific agent that can handle
    the message of the type T."""

    message: Any
    sender: AgentId | None
    recipient: AgentId
    future: Future[Any]
    cancellation_token: CancellationToken
    metadata: EnvelopeMetadata | None = None
    message_id: str

# From autogen_core/_single_threaded_agent_runtime.py
class ResponseMessageEnvelope:
    """A message envelope for sending a response to a message."""

    message: Any
    future: Future[Any]
    sender: AgentId
    recipient: AgentId | None
    metadata: EnvelopeMetadata | None = None

# From autogen_core/_single_threaded_agent_runtime.py
class RunContext:
    def __init__(self, runtime: SingleThreadedAgentRuntime) -> None:
        self._runtime = runtime
        self._run_task = asyncio.create_task(self._run())
        self._stopped = asyncio.Event()

    async def _run(self) -> None:
        while True:
            if self._stopped.is_set():
                return

            await self._runtime._process_next()  # type: ignore

    async def stop(self) -> None:
        self._stopped.set()
        self._runtime._message_queue.shutdown(immediate=True)  # type: ignore
        await self._run_task

    async def stop_when_idle(self) -> None:
        await self._runtime._message_queue.join()  # type: ignore
        self._stopped.set()
        self._runtime._message_queue.shutdown(immediate=True)  # type: ignore
        await self._run_task

    async def stop_when(self, condition: Callable[[], bool], check_period: float = 1.0) -> None:
        async def check_condition() -> None:
            while not condition():
                await asyncio.sleep(check_period)
            await self.stop()

        await asyncio.create_task(check_condition())

# From autogen_core/_single_threaded_agent_runtime.py
class SingleThreadedAgentRuntime(AgentRuntime):
    """A single-threaded agent runtime that processes all messages using a single asyncio queue.
    Messages are delivered in the order they are received, and the runtime processes
    each message in a separate asyncio task concurrently.

    .. note::

        This runtime is suitable for development and standalone applications.
        It is not suitable for high-throughput or high-concurrency scenarios.

    Args:
        intervention_handlers (List[InterventionHandler], optional): A list of intervention
            handlers that can intercept messages before they are sent or published. Defaults to None.
        tracer_provider (TracerProvider, optional): The tracer provider to use for tracing. Defaults to None.
            Additionally, you can set environment variable `AUTOGEN_DISABLE_RUNTIME_TRACING` to `true` to disable the agent runtime telemetry if you don't have access to the runtime constructor. For example, if you are using `ComponentConfig`.
        ignore_unhandled_exceptions (bool, optional): Whether to ignore unhandled exceptions in that occur in agent event handlers. Any background exceptions will be raised on the next call to `process_next` or from an awaited `stop`, `stop_when_idle` or `stop_when`. Note, this does not apply to RPC handlers. Defaults to True.

    Examples:

        A simple example of creating a runtime, registering an agent, sending a message and stopping the runtime:

        .. code-block:: python

            import asyncio
            from dataclasses import dataclass

            from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler


            @dataclass
            class MyMessage:
                content: str


            class MyAgent(RoutedAgent):
                @message_handler
                async def handle_my_message(self, message: MyMessage, ctx: MessageContext) -> None:
                    print(f"Received message: {message.content}")


            async def main() -> None:
                # Create a runtime and register the agent
                runtime = SingleThreadedAgentRuntime()
                await MyAgent.register(runtime, "my_agent", lambda: MyAgent("My agent"))

                # Start the runtime, send a message and stop the runtime
                runtime.start()
                await runtime.send_message(MyMessage("Hello, world!"), recipient=AgentId("my_agent", "default"))
                await runtime.stop()


            asyncio.run(main())

        An example of creating a runtime, registering an agent, publishing a message and stopping the runtime:

        .. code-block:: python

            import asyncio
            from dataclasses import dataclass

            from autogen_core import (
                DefaultTopicId,
                MessageContext,
                RoutedAgent,
                SingleThreadedAgentRuntime,
                default_subscription,
                message_handler,
            )


            @dataclass
            class MyMessage:
                content: str


            # The agent is subscribed to the default topic.
            @default_subscription
            class MyAgent(RoutedAgent):
                @message_handler
                async def handle_my_message(self, message: MyMessage, ctx: MessageContext) -> None:
                    print(f"Received message: {message.content}")


            async def main() -> None:
                # Create a runtime and register the agent
                runtime = SingleThreadedAgentRuntime()
                await MyAgent.register(runtime, "my_agent", lambda: MyAgent("My agent"))

                # Start the runtime.
                runtime.start()
                # Publish a message to the default topic that the agent is subscribed to.
                await runtime.publish_message(MyMessage("Hello, world!"), DefaultTopicId())
                # Wait for the message to be processed and then stop the runtime.
                await runtime.stop_when_idle()


            asyncio.run(main())

    """

    def __init__(
        self,
        *,
        intervention_handlers: List[InterventionHandler] | None = None,
        tracer_provider: TracerProvider | None = None,
        ignore_unhandled_exceptions: bool = True,
    ) -> None:
        self._tracer_helper = TraceHelper(tracer_provider, MessageRuntimeTracingConfig("SingleThreadedAgentRuntime"))
        self._message_queue: Queue[PublishMessageEnvelope | SendMessageEnvelope | ResponseMessageEnvelope] = Queue()
        # (namespace, type) -> List[AgentId]
        self._agent_factories: Dict[
            str, Callable[[], Agent | Awaitable[Agent]] | Callable[[AgentRuntime, AgentId], Agent | Awaitable[Agent]]
        ] = {}
        self._instantiated_agents: Dict[AgentId, Agent] = {}
        self._intervention_handlers = intervention_handlers
        self._background_tasks: Set[Task[Any]] = set()
        self._subscription_manager = SubscriptionManager()
        self._run_context: RunContext | None = None
        self._serialization_registry = SerializationRegistry()
        self._ignore_unhandled_handler_exceptions = ignore_unhandled_exceptions
        self._background_exception: BaseException | None = None
        self._agent_instance_types: Dict[str, Type[Agent]] = {}

    @property
    def unprocessed_messages_count(
        self,
    ) -> int:
        return self._message_queue.qsize()

    @property
    def _known_agent_names(self) -> Set[str]:
        return set(self._agent_factories.keys())

    async def _create_otel_attributes(
        self,
        sender_agent_id: AgentId | None = None,
        recipient_agent_id: AgentId | None = None,
        message_context: MessageContext | None = None,
        message: Any = None,
    ) -> Mapping[str, str]:
        """Create OpenTelemetry attributes for the given agent and message.

        Args:
            sender_agent (Agent, optional): The sender agent instance.
            recipient_agent (Agent, optional): The recipient agent instance.
            message (Any): The message instance.

        Returns:
            Attributes: A dictionary of OpenTelemetry attributes.
        """
        if not sender_agent_id and not recipient_agent_id and not message:
            return {}
        attributes: Dict[str, str] = {}
        if sender_agent_id:
            sender_agent = await self._get_agent(sender_agent_id)
            attributes["sender_agent_type"] = sender_agent.id.type
            attributes["sender_agent_class"] = sender_agent.__class__.__name__
        if recipient_agent_id:
            recipient_agent = await self._get_agent(recipient_agent_id)
            attributes["recipient_agent_type"] = recipient_agent.id.type
            attributes["recipient_agent_class"] = recipient_agent.__class__.__name__

        if message_context:
            serialized_message_context = {
                "sender": str(message_context.sender),
                "topic_id": str(message_context.topic_id),
                "is_rpc": message_context.is_rpc,
                "message_id": message_context.message_id,
            }
            attributes["message_context"] = json.dumps(serialized_message_context)

        if message:
            try:
                serialized_message = self._try_serialize(message)
            except Exception as e:
                serialized_message = str(e)
        else:
            serialized_message = "No Message"
        attributes["message"] = serialized_message

        return attributes

    # Returns the response of the message
    async def send_message(
        self,
        message: Any,
        recipient: AgentId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any:
        if cancellation_token is None:
            cancellation_token = CancellationToken()

        if message_id is None:
            message_id = str(uuid.uuid4())

        event_logger.info(
            MessageEvent(
                payload=self._try_serialize(message),
                sender=sender,
                receiver=recipient,
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            )
        )

        with self._tracer_helper.trace_block(
            "create",
            recipient,
            parent=None,
            extraAttributes={"message_type": type(message).__name__},
        ):
            future = asyncio.get_event_loop().create_future()
            if recipient.type not in self._known_agent_names:
                future.set_exception(Exception("Recipient not found"))
                return await future

            content = message.__dict__ if hasattr(message, "__dict__") else message
            logger.info(f"Sending message of type {type(message).__name__} to {recipient.type}: {content}")

            await self._message_queue.put(
                SendMessageEnvelope(
                    message=message,
                    recipient=recipient,
                    future=future,
                    cancellation_token=cancellation_token,
                    sender=sender,
                    metadata=get_telemetry_envelope_metadata(),
                    message_id=message_id,
                )
            )

            cancellation_token.link_future(future)

            return await future

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> None:
        with self._tracer_helper.trace_block(
            "create",
            topic_id,
            parent=None,
            extraAttributes={"message_type": type(message).__name__},
        ):
            if cancellation_token is None:
                cancellation_token = CancellationToken()
            content = message.__dict__ if hasattr(message, "__dict__") else message
            logger.info(f"Publishing message of type {type(message).__name__} to all subscribers: {content}")

            if message_id is None:
                message_id = str(uuid.uuid4())

            event_logger.info(
                MessageEvent(
                    payload=self._try_serialize(message),
                    sender=sender,
                    receiver=topic_id,
                    kind=MessageKind.PUBLISH,
                    delivery_stage=DeliveryStage.SEND,
                )
            )

            await self._message_queue.put(
                PublishMessageEnvelope(
                    message=message,
                    cancellation_token=cancellation_token,
                    sender=sender,
                    topic_id=topic_id,
                    metadata=get_telemetry_envelope_metadata(),
                    message_id=message_id,
                )
            )

    async def save_state(self) -> Mapping[str, Any]:
        """Save the state of all instantiated agents.

        This method calls the :meth:`~autogen_core.BaseAgent.save_state` method on each agent and returns a dictionary
        mapping agent IDs to their state.

        .. note::
            This method does not currently save the subscription state. We will add this in the future.

        Returns:
            A dictionary mapping agent IDs to their state.

        """
        state: Dict[str, Dict[str, Any]] = {}
        for agent_id in self._instantiated_agents:
            state[str(agent_id)] = dict(await (await self._get_agent(agent_id)).save_state())
        return state

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the state of all instantiated agents.

        This method calls the :meth:`~autogen_core.BaseAgent.load_state` method on each agent with the state
        provided in the dictionary. The keys of the dictionary are the agent IDs, and the values are the state
        dictionaries returned by the :meth:`~autogen_core.BaseAgent.save_state` method.

        .. note::

            This method does not currently load the subscription state. We will add this in the future.

        """
        for agent_id_str in state:
            agent_id = AgentId.from_str(agent_id_str)
            if agent_id.type in self._known_agent_names:
                await (await self._get_agent(agent_id)).load_state(state[str(agent_id)])

    async def _process_send(self, message_envelope: SendMessageEnvelope) -> None:
        with self._tracer_helper.trace_block("send", message_envelope.recipient, parent=message_envelope.metadata):
            recipient = message_envelope.recipient

            if recipient.type not in self._known_agent_names:
                raise LookupError(f"Agent type '{recipient.type}' does not exist.")

            try:
                sender_id = str(message_envelope.sender) if message_envelope.sender is not None else "Unknown"
                logger.info(
                    f"Calling message handler for {recipient} with message type {type(message_envelope.message).__name__} sent by {sender_id}"
                )
                event_logger.info(
                    MessageEvent(
                        payload=self._try_serialize(message_envelope.message),
                        sender=message_envelope.sender,
                        receiver=recipient,
                        kind=MessageKind.DIRECT,
                        delivery_stage=DeliveryStage.DELIVER,
                    )
                )
                recipient_agent = await self._get_agent(recipient)

                message_context = MessageContext(
                    sender=message_envelope.sender,
                    topic_id=None,
                    is_rpc=True,
                    cancellation_token=message_envelope.cancellation_token,
                    message_id=message_envelope.message_id,
                )
                with self._tracer_helper.trace_block(
                    "process",
                    recipient_agent.id,
                    parent=message_envelope.metadata,
                    attributes=await self._create_otel_attributes(
                        sender_agent_id=message_envelope.sender,
                        recipient_agent_id=recipient,
                        message_context=message_context,
                        message=message_envelope.message,
                    ),
                ):
                    with MessageHandlerContext.populate_context(recipient_agent.id):
                        response = await recipient_agent.on_message(
                            message_envelope.message,
                            ctx=message_context,
                        )
            except CancelledError as e:
                if not message_envelope.future.cancelled():
                    message_envelope.future.set_exception(e)
                self._message_queue.task_done()
                event_logger.info(
                    MessageHandlerExceptionEvent(
                        payload=self._try_serialize(message_envelope.message),
                        handling_agent=recipient,
                        exception=e,
                    )
                )
                return
            except BaseException as e:
                message_envelope.future.set_exception(e)
                self._message_queue.task_done()
                event_logger.info(
                    MessageHandlerExceptionEvent(
                        payload=self._try_serialize(message_envelope.message),
                        handling_agent=recipient,
                        exception=e,
                    )
                )
                return

            event_logger.info(
                MessageEvent(
                    payload=self._try_serialize(response),
                    sender=message_envelope.recipient,
                    receiver=message_envelope.sender,
                    kind=MessageKind.RESPOND,
                    delivery_stage=DeliveryStage.SEND,
                )
            )

            await self._message_queue.put(
                ResponseMessageEnvelope(
                    message=response,
                    future=message_envelope.future,
                    sender=message_envelope.recipient,
                    recipient=message_envelope.sender,
                    metadata=get_telemetry_envelope_metadata(),
                )
            )
            self._message_queue.task_done()

    async def _process_publish(self, message_envelope: PublishMessageEnvelope) -> None:
        with self._tracer_helper.trace_block("publish", message_envelope.topic_id, parent=message_envelope.metadata):
            try:
                responses: List[Awaitable[Any]] = []
                recipients = await self._subscription_manager.get_subscribed_recipients(message_envelope.topic_id)
                for agent_id in recipients:
                    # Avoid sending the message back to the sender
                    if message_envelope.sender is not None and agent_id == message_envelope.sender:
                        continue

                    sender_agent = (
                        await self._get_agent(message_envelope.sender) if message_envelope.sender is not None else None
                    )
                    sender_name = str(sender_agent.id) if sender_agent is not None else "Unknown"
                    logger.info(
                        f"Calling message handler for {agent_id.type} with message type {type(message_envelope.message).__name__} published by {sender_name}"
                    )
                    event_logger.info(
                        MessageEvent(
                            payload=self._try_serialize(message_envelope.message),
                            sender=message_envelope.sender,
                            receiver=None,
                            kind=MessageKind.PUBLISH,
                            delivery_stage=DeliveryStage.DELIVER,
                        )
                    )
                    message_context = MessageContext(
                        sender=message_envelope.sender,
                        topic_id=message_envelope.topic_id,
                        is_rpc=False,
                        cancellation_token=message_envelope.cancellation_token,
                        message_id=message_envelope.message_id,
                    )
                    agent = await self._get_agent(agent_id)

                    async def _on_message(agent: Agent, message_context: MessageContext) -> Any:
                        with self._tracer_helper.trace_block(
                            "process",
                            agent.id,
                            parent=message_envelope.metadata,
                            attributes=await self._create_otel_attributes(
                                sender_agent_id=message_envelope.sender,
                                recipient_agent_id=agent.id,
                                message_context=message_context,
                                message=message_envelope.message,
                            ),
                        ):
                            with MessageHandlerContext.populate_context(agent.id):
                                try:
                                    return await agent.on_message(
                                        message_envelope.message,
                                        ctx=message_context,
                                    )
                                except BaseException as e:
                                    logger.error(f"Error processing publish message for {agent.id}", exc_info=True)
                                    event_logger.info(
                                        MessageHandlerExceptionEvent(
                                            payload=self._try_serialize(message_envelope.message),
                                            handling_agent=agent.id,
                                            exception=e,
                                        )
                                    )
                                    raise e

                    future = _on_message(agent, message_context)
                    responses.append(future)

                await asyncio.gather(*responses)
            except BaseException as e:
                if not self._ignore_unhandled_handler_exceptions:
                    self._background_exception = e
            finally:
                self._message_queue.task_done()
            # TODO if responses are given for a publish

    async def _process_response(self, message_envelope: ResponseMessageEnvelope) -> None:
        with self._tracer_helper.trace_block(
            "ack",
            message_envelope.recipient,
            parent=message_envelope.metadata,
            attributes=await self._create_otel_attributes(
                sender_agent_id=message_envelope.sender,
                recipient_agent_id=message_envelope.recipient,
                message=message_envelope.message,
            ),
        ):
            content = (
                message_envelope.message.__dict__
                if hasattr(message_envelope.message, "__dict__")
                else message_envelope.message
            )
            logger.info(
                f"Resolving response with message type {type(message_envelope.message).__name__} for recipient {message_envelope.recipient} from {message_envelope.sender.type}: {content}"
            )
            event_logger.info(
                MessageEvent(
                    payload=self._try_serialize(message_envelope.message),
                    sender=message_envelope.sender,
                    receiver=message_envelope.recipient,
                    kind=MessageKind.RESPOND,
                    delivery_stage=DeliveryStage.DELIVER,
                )
            )
            if not message_envelope.future.cancelled():
                message_envelope.future.set_result(message_envelope.message)
            self._message_queue.task_done()

    async def process_next(self) -> None:
        """Process the next message in the queue.

        If there is an unhandled exception in the background task, it will be raised here. `process_next` cannot be called again after an unhandled exception is raised.
        """
        await self._process_next()

    async def _process_next(self) -> None:
        """Process the next message in the queue."""

        if self._background_exception is not None:
            e = self._background_exception
            self._background_exception = None
            self._message_queue.shutdown(immediate=True)  # type: ignore
            raise e

        try:
            message_envelope = await self._message_queue.get()
        except QueueShutDown:
            if self._background_exception is not None:
                e = self._background_exception
                self._background_exception = None
                raise e from None
            return

        match message_envelope:
            case SendMessageEnvelope(message=message, sender=sender, recipient=recipient, future=future):
                if self._intervention_handlers is not None:
                    for handler in self._intervention_handlers:
                        with self._tracer_helper.trace_block(
                            "intercept", handler.__class__.__name__, parent=message_envelope.metadata
                        ):
                            try:
                                message_context = MessageContext(
                                    sender=sender,
                                    topic_id=None,
                                    is_rpc=True,
                                    cancellation_token=message_envelope.cancellation_token,
                                    message_id=message_envelope.message_id,
                                )
                                temp_message = await handler.on_send(
                                    message, message_context=message_context, recipient=recipient
                                )
                                _warn_if_none(temp_message, "on_send")
                            except BaseException as e:
                                future.set_exception(e)
                                return
                            if temp_message is DropMessage or isinstance(temp_message, DropMessage):
                                event_logger.info(
                                    MessageDroppedEvent(
                                        payload=self._try_serialize(message),
                                        sender=sender,
                                        receiver=recipient,
                                        kind=MessageKind.DIRECT,
                                    )
                                )
                                future.set_exception(MessageDroppedException())
                                return

                        message_envelope.message = temp_message
                task = asyncio.create_task(self._process_send(message_envelope))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            case PublishMessageEnvelope(
                message=message,
                sender=sender,
                topic_id=topic_id,
            ):
                if self._intervention_handlers is not None:
                    for handler in self._intervention_handlers:
                        with self._tracer_helper.trace_block(
                            "intercept", handler.__class__.__name__, parent=message_envelope.metadata
                        ):
                            try:
                                message_context = MessageContext(
                                    sender=sender,
                                    topic_id=topic_id,
                                    is_rpc=False,
                                    cancellation_token=message_envelope.cancellation_token,
                                    message_id=message_envelope.message_id,
                                )
                                temp_message = await handler.on_publish(message, message_context=message_context)
                                _warn_if_none(temp_message, "on_publish")
                            except BaseException as e:
                                # TODO: we should raise the intervention exception to the publisher.
                                logger.error(f"Exception raised in in intervention handler: {e}", exc_info=True)
                                return
                            if temp_message is DropMessage or isinstance(temp_message, DropMessage):
                                event_logger.info(
                                    MessageDroppedEvent(
                                        payload=self._try_serialize(message),
                                        sender=sender,
                                        receiver=topic_id,
                                        kind=MessageKind.PUBLISH,
                                    )
                                )
                                return

                        message_envelope.message = temp_message

                task = asyncio.create_task(self._process_publish(message_envelope))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            case ResponseMessageEnvelope(message=message, sender=sender, recipient=recipient, future=future):
                if self._intervention_handlers is not None:
                    for handler in self._intervention_handlers:
                        try:
                            temp_message = await handler.on_response(message, sender=sender, recipient=recipient)
                            _warn_if_none(temp_message, "on_response")
                        except BaseException as e:
                            # TODO: should we raise the exception to sender of the response instead?
                            future.set_exception(e)
                            return
                        if temp_message is DropMessage or isinstance(temp_message, DropMessage):
                            event_logger.info(
                                MessageDroppedEvent(
                                    payload=self._try_serialize(message),
                                    sender=sender,
                                    receiver=recipient,
                                    kind=MessageKind.RESPOND,
                                )
                            )
                            future.set_exception(MessageDroppedException())
                            return
                        message_envelope.message = temp_message
                task = asyncio.create_task(self._process_response(message_envelope))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        # Yield control to the message loop to allow other tasks to run
        await asyncio.sleep(0)

    def start(self) -> None:
        """Start the runtime message processing loop. This runs in a background task.

        Example:

        .. code-block:: python

            import asyncio
            from autogen_core import SingleThreadedAgentRuntime


            async def main() -> None:
                runtime = SingleThreadedAgentRuntime()
                runtime.start()

                # ... do other things ...

                await runtime.stop()


            asyncio.run(main())

        """
        if self._run_context is not None:
            raise RuntimeError("Runtime is already started")
        self._run_context = RunContext(self)

    async def close(self) -> None:
        """Calls :meth:`stop` if applicable and the :meth:`Agent.close` method on all instantiated agents"""
        # stop the runtime if it hasn't been stopped yet
        if self._run_context is not None:
            await self.stop()
        # close all the agents that have been instantiated
        for agent_id in self._instantiated_agents:
            agent = await self._get_agent(agent_id)
            await agent.close()

    async def stop(self) -> None:
        """Immediately stop the runtime message processing loop. The currently processing message will be completed, but all others following it will be discarded."""
        if self._run_context is None:
            raise RuntimeError("Runtime is not started")

        try:
            await self._run_context.stop()
        finally:
            self._run_context = None
            self._message_queue = Queue()

    async def stop_when_idle(self) -> None:
        """Stop the runtime message processing loop when there is
        no outstanding message being processed or queued. This is the most common way to stop the runtime."""
        if self._run_context is None:
            raise RuntimeError("Runtime is not started")

        try:
            await self._run_context.stop_when_idle()
        finally:
            self._run_context = None
            self._message_queue = Queue()

    async def stop_when(self, condition: Callable[[], bool]) -> None:
        """Stop the runtime message processing loop when the condition is met.

        .. caution::

            This method is not recommended to be used, and is here for legacy
            reasons. It will spawn a busy loop to continually check the
            condition. It is much more efficient to call `stop_when_idle` or
            `stop` instead. If you need to stop the runtime based on a
            condition, consider using a background task and asyncio.Event to
            signal when the condition is met and the background task should call
            stop.

        """
        if self._run_context is None:
            raise RuntimeError("Runtime is not started")
        await self._run_context.stop_when(condition)

        self._run_context = None
        self._message_queue = Queue()

    async def agent_metadata(self, agent: AgentId) -> AgentMetadata:
        return (await self._get_agent(agent)).metadata

    async def agent_save_state(self, agent: AgentId) -> Mapping[str, Any]:
        return await (await self._get_agent(agent)).save_state()

    async def agent_load_state(self, agent: AgentId, state: Mapping[str, Any]) -> None:
        await (await self._get_agent(agent)).load_state(state)

    async def register_factory(
        self,
        type: str | AgentType,
        agent_factory: Callable[[], T | Awaitable[T]],
        *,
        expected_class: type[T] | None = None,
    ) -> AgentType:
        if isinstance(type, str):
            type = AgentType(type)

        if type.type in self._agent_factories:
            raise ValueError(f"Agent with type {type} already exists.")

        async def factory_wrapper() -> T:
            maybe_agent_instance = agent_factory()
            if inspect.isawaitable(maybe_agent_instance):
                agent_instance = await maybe_agent_instance
            else:
                agent_instance = maybe_agent_instance

            if expected_class is not None and not issubclass(type_func_alias(agent_instance), expected_class):
                raise ValueError(
                    f"Factory registered using the wrong type: expected {expected_class.__name__}, got {type_func_alias(agent_instance).__name__}"
                )
            return agent_instance

        self._agent_factories[type.type] = factory_wrapper

        return type

    async def register_agent_instance(
        self,
        agent_instance: Agent,
        agent_id: AgentId,
    ) -> AgentId:
        def agent_factory() -> Agent:
            raise RuntimeError(
                "Agent factory was invoked for an agent instance that was not registered. This is likely due to the agent type being incorrectly subscribed to a topic. If this exception occurs when publishing a message to the DefaultTopicId, then it is likely that `skip_class_subscriptions` needs to be turned off when registering the agent."
            )

        if agent_id in self._instantiated_agents:
            raise ValueError(f"Agent with id {agent_id} already exists.")

        if agent_id.type not in self._agent_factories:
            self._agent_factories[agent_id.type] = agent_factory
            self._agent_instance_types[agent_id.type] = type_func_alias(agent_instance)
        else:
            if self._agent_factories[agent_id.type].__code__ != agent_factory.__code__:
                raise ValueError("Agent factories and agent instances cannot be registered to the same type.")
            if self._agent_instance_types[agent_id.type] != type_func_alias(agent_instance):
                raise ValueError("Agent instances must be the same object type.")

        await agent_instance.bind_id_and_runtime(id=agent_id, runtime=self)
        self._instantiated_agents[agent_id] = agent_instance
        return agent_id

    async def _invoke_agent_factory(
        self,
        agent_factory: Callable[[], T | Awaitable[T]] | Callable[[AgentRuntime, AgentId], T | Awaitable[T]],
        agent_id: AgentId,
    ) -> T:
        with AgentInstantiationContext.populate_context((self, agent_id)):
            try:
                if len(inspect.signature(agent_factory).parameters) == 0:
                    factory_one = cast(Callable[[], T], agent_factory)
                    agent = factory_one()
                elif len(inspect.signature(agent_factory).parameters) == 2:
                    warnings.warn(
                        "Agent factories that take two arguments are deprecated. Use AgentInstantiationContext instead. Two arg factories will be removed in a future version.",
                        stacklevel=2,
                    )
                    factory_two = cast(Callable[[AgentRuntime, AgentId], T], agent_factory)
                    agent = factory_two(self, agent_id)
                else:
                    raise ValueError("Agent factory must take 0 or 2 arguments.")

                if inspect.isawaitable(agent):
                    agent = cast(T, await agent)
                return agent

            except BaseException as e:
                event_logger.info(
                    AgentConstructionExceptionEvent(
                        agent_id=agent_id,
                        exception=e,
                    )
                )
                logger.error(f"Error constructing agent {agent_id}", exc_info=True)
                raise

    async def _get_agent(self, agent_id: AgentId) -> Agent:
        if agent_id in self._instantiated_agents:
            return self._instantiated_agents[agent_id]

        if agent_id.type not in self._agent_factories:
            raise LookupError(f"Agent with name {agent_id.type} not found.")

        agent_factory = self._agent_factories[agent_id.type]
        agent = await self._invoke_agent_factory(agent_factory, agent_id)
        self._instantiated_agents[agent_id] = agent
        return agent

    # TODO: uncomment out the following type ignore when this is fixed in mypy: https://github.com/python/mypy/issues/3737
    async def try_get_underlying_agent_instance(self, id: AgentId, type: Type[T] = Agent) -> T:  # type: ignore[assignment]
        if id.type not in self._agent_factories:
            raise LookupError(f"Agent with name {id.type} not found.")

        # TODO: check if remote
        agent_instance = await self._get_agent(id)

        if not isinstance(agent_instance, type):
            raise TypeError(
                f"Agent with name {id.type} is not of type {type.__name__}. It is of type {type_func_alias(agent_instance).__name__}"
            )

        return agent_instance

    async def add_subscription(self, subscription: Subscription) -> None:
        await self._subscription_manager.add_subscription(subscription)

    async def remove_subscription(self, id: str) -> None:
        await self._subscription_manager.remove_subscription(id)

    async def get(
        self, id_or_type: AgentId | AgentType | str, /, key: str = "default", *, lazy: bool = True
    ) -> AgentId:
        return await get_impl(
            id_or_type=id_or_type,
            key=key,
            lazy=lazy,
            instance_getter=self._get_agent,
        )

    def add_message_serializer(self, serializer: MessageSerializer[Any] | Sequence[MessageSerializer[Any]]) -> None:
        self._serialization_registry.add_serializer(serializer)

    def _try_serialize(self, message: Any) -> str:
        try:
            type_name = self._serialization_registry.type_name(message)
            return self._serialization_registry.serialize(
                message, type_name=type_name, data_content_type=JSON_DATA_CONTENT_TYPE
            ).decode("utf-8")
        except ValueError:
            return "Message could not be serialized"

# From autogen_core/_single_threaded_agent_runtime.py
def unprocessed_messages_count(
        self,
    ) -> int:
        return self._message_queue.qsize()

# From autogen_core/_single_threaded_agent_runtime.py
def start(self) -> None:
        """Start the runtime message processing loop. This runs in a background task.

        Example:

        .. code-block:: python

            import asyncio
            from autogen_core import SingleThreadedAgentRuntime


            async def main() -> None:
                runtime = SingleThreadedAgentRuntime()
                runtime.start()

                # ... do other things ...

                await runtime.stop()


            asyncio.run(main())

        """
        if self._run_context is not None:
            raise RuntimeError("Runtime is already started")
        self._run_context = RunContext(self)

# From autogen_core/_single_threaded_agent_runtime.py
def add_message_serializer(self, serializer: MessageSerializer[Any] | Sequence[MessageSerializer[Any]]) -> None:
        self._serialization_registry.add_serializer(serializer)

# From autogen_core/_single_threaded_agent_runtime.py
def agent_factory() -> Agent:
            raise RuntimeError(
                "Agent factory was invoked for an agent instance that was not registered. This is likely due to the agent type being incorrectly subscribed to a topic. If this exception occurs when publishing a message to the DefaultTopicId, then it is likely that `skip_class_subscriptions` needs to be turned off when registering the agent."
            )

from typing import overload

# From autogen_core/_agent_runtime.py
class AgentRuntime(Protocol):
    async def send_message(
        self,
        message: Any,
        recipient: AgentId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any:
        """Send a message to an agent and get a response.

        Args:
            message (Any): The message to send.
            recipient (AgentId): The agent to send the message to.
            sender (AgentId | None, optional): Agent which sent the message. Should **only** be None if this was sent from no agent, such as directly to the runtime externally. Defaults to None.
            cancellation_token (CancellationToken | None, optional): Token used to cancel an in progress . Defaults to None.

        Raises:
            CantHandleException: If the recipient cannot handle the message.
            UndeliverableException: If the message cannot be delivered.
            Other: Any other exception raised by the recipient.

        Returns:
            Any: The response from the agent.
        """

        ...

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> None:
        """Publish a message to all agents in the given namespace, or if no namespace is provided, the namespace of the sender.

        No responses are expected from publishing.

        Args:
            message (Any): The message to publish.
            topic_id (TopicId): The topic to publish the message to.
            sender (AgentId | None, optional): The agent which sent the message. Defaults to None.
            cancellation_token (CancellationToken | None, optional): Token used to cancel an in progress. Defaults to None.
            message_id (str | None, optional): The message id. If None, a new message id will be generated. Defaults to None. This message id must be unique. and is recommended to be a UUID.

        Raises:
            UndeliverableException: If the message cannot be delivered.
        """
        ...

    async def register_factory(
        self,
        type: str | AgentType,
        agent_factory: Callable[[], T | Awaitable[T]],
        *,
        expected_class: type[T] | None = None,
    ) -> AgentType:
        """Register an agent factory with the runtime associated with a specific type. The type must be unique. This API does not add any subscriptions.

        .. note::

            This is a low level API and usually the agent class's `register` method should be used instead, as this also handles subscriptions automatically.

        Example:

        .. code-block:: python

            from dataclasses import dataclass

            from autogen_core import AgentRuntime, MessageContext, RoutedAgent, event
            from autogen_core.models import UserMessage


            @dataclass
            class MyMessage:
                content: str


            class MyAgent(RoutedAgent):
                def __init__(self) -> None:
                    super().__init__("My core agent")

                @event
                async def handler(self, message: UserMessage, context: MessageContext) -> None:
                    print("Event received: ", message.content)


            async def my_agent_factory():
                return MyAgent()


            async def main() -> None:
                runtime: AgentRuntime = ...  # type: ignore
                await runtime.register_factory("my_agent", lambda: MyAgent())


            import asyncio

            asyncio.run(main())


        Args:
            type (str): The type of agent this factory creates. It is not the same as agent class name. The `type` parameter is used to differentiate between different factory functions rather than agent classes.
            agent_factory (Callable[[], T]): The factory that creates the agent, where T is a concrete Agent type. Inside the factory, use `autogen_core.AgentInstantiationContext` to access variables like the current runtime and agent ID.
            expected_class (type[T] | None, optional): The expected class of the agent, used for runtime validation of the factory. Defaults to None. If None, no validation is performed.
        """
        ...

    async def register_agent_instance(
        self,
        agent_instance: Agent,
        agent_id: AgentId,
    ) -> AgentId:
        """Register an agent instance with the runtime. The type may be reused, but each agent_id must be unique. All agent instances within a type must be of the same object type. This API does not add any subscriptions.

        .. note::

            This is a low level API and usually the agent class's `register_instance` method should be used instead, as this also handles subscriptions automatically.

        Example:

        .. code-block:: python

            from dataclasses import dataclass

            from autogen_core import AgentId, AgentRuntime, MessageContext, RoutedAgent, event
            from autogen_core.models import UserMessage


            @dataclass
            class MyMessage:
                content: str


            class MyAgent(RoutedAgent):
                def __init__(self) -> None:
                    super().__init__("My core agent")

                @event
                async def handler(self, message: UserMessage, context: MessageContext) -> None:
                    print("Event received: ", message.content)


            async def main() -> None:
                runtime: AgentRuntime = ...  # type: ignore
                agent = MyAgent()
                await runtime.register_agent_instance(
                    agent_instance=agent, agent_id=AgentId(type="my_agent", key="default")
                )


            import asyncio

            asyncio.run(main())


        Args:
            agent_instance (Agent): A concrete instance of the agent.
            agent_id (AgentId): The agent's identifier. The agent's type is `agent_id.type`.
        """
        ...

    # TODO: uncomment out the following type ignore when this is fixed in mypy: https://github.com/python/mypy/issues/3737
    async def try_get_underlying_agent_instance(self, id: AgentId, type: Type[T] = Agent) -> T:  # type: ignore[assignment]
        """Try to get the underlying agent instance by name and namespace. This is generally discouraged (hence the long name), but can be useful in some cases.

        If the underlying agent is not accessible, this will raise an exception.

        Args:
            id (AgentId): The agent id.
            type (Type[T], optional): The expected type of the agent. Defaults to Agent.

        Returns:
            T: The concrete agent instance.

        Raises:
            LookupError: If the agent is not found.
            NotAccessibleError: If the agent is not accessible, for example if it is located remotely.
            TypeError: If the agent is not of the expected type.
        """
        ...

    @overload
    async def get(self, id: AgentId, /, *, lazy: bool = ...) -> AgentId: ...

    @overload
    async def get(self, type: AgentType | str, /, key: str = ..., *, lazy: bool = ...) -> AgentId: ...

    async def get(
        self, id_or_type: AgentId | AgentType | str, /, key: str = "default", *, lazy: bool = True
    ) -> AgentId: ...

    async def save_state(self) -> Mapping[str, Any]:
        """Save the state of the entire runtime, including all hosted agents. The only way to restore the state is to pass it to :meth:`load_state`.

        The structure of the state is implementation defined and can be any JSON serializable object.

        Returns:
            Mapping[str, Any]: The saved state.
        """
        ...

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the state of the entire runtime, including all hosted agents. The state should be the same as the one returned by :meth:`save_state`.

        Args:
            state (Mapping[str, Any]): The saved state.
        """
        ...

    async def agent_metadata(self, agent: AgentId) -> AgentMetadata:
        """Get the metadata for an agent.

        Args:
            agent (AgentId): The agent id.

        Returns:
            AgentMetadata: The agent metadata.
        """
        ...

    async def agent_save_state(self, agent: AgentId) -> Mapping[str, Any]:
        """Save the state of a single agent.

        The structure of the state is implementation defined and can be any JSON serializable object.

        Args:
            agent (AgentId): The agent id.

        Returns:
            Mapping[str, Any]: The saved state.
        """
        ...

    async def agent_load_state(self, agent: AgentId, state: Mapping[str, Any]) -> None:
        """Load the state of a single agent.

        Args:
            agent (AgentId): The agent id.
            state (Mapping[str, Any]): The saved state.
        """
        ...

    async def add_subscription(self, subscription: Subscription) -> None:
        """Add a new subscription that the runtime should fulfill when processing published messages

        Args:
            subscription (Subscription): The subscription to add
        """
        ...

    async def remove_subscription(self, id: str) -> None:
        """Remove a subscription from the runtime

        Args:
            id (str): id of the subscription to remove

        Raises:
            LookupError: If the subscription does not exist
        """
        ...

    def add_message_serializer(self, serializer: MessageSerializer[Any] | Sequence[MessageSerializer[Any]]) -> None:
        """Add a new message serialization serializer to the runtime

        Note: This will deduplicate serializers based on the type_name and data_content_type properties

        Args:
            serializer (MessageSerializer[Any] | Sequence[MessageSerializer[Any]]): The serializer/s to add
        """
        ...

from abc import ABC
from abc import abstractmethod
from typing import ClassVar
from typing import Tuple
from typing import final
from typing_extensions import Self
from _serialization import try_get_known_serializers_for_type
from _subscription import UnboundSubscription
from _subscription_context import SubscriptionInstantiationContext
from _type_prefix_subscription import TypePrefixSubscription
from _type_subscription import TypeSubscription

# From autogen_core/_base_agent.py
class BaseAgent(ABC, Agent):
    internal_unbound_subscriptions_list: ClassVar[List[UnboundSubscription]] = []
    """:meta private:"""
    internal_extra_handles_types: ClassVar[List[Tuple[Type[Any], List[MessageSerializer[Any]]]]] = []
    """:meta private:"""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Automatically set class_variable in each subclass so that they are not shared between subclasses
        cls.internal_extra_handles_types = []
        cls.internal_unbound_subscriptions_list = []

    @classmethod
    def _handles_types(cls) -> List[Tuple[Type[Any], List[MessageSerializer[Any]]]]:
        return cls.internal_extra_handles_types

    @classmethod
    def _unbound_subscriptions(cls) -> List[UnboundSubscription]:
        return cls.internal_unbound_subscriptions_list

    @property
    def metadata(self) -> AgentMetadata:
        assert self._id is not None
        return AgentMetadata(key=self._id.key, type=self._id.type, description=self._description)

    def __init__(self, description: str) -> None:
        if AgentInstantiationContext.is_in_factory_call():
            self._runtime: AgentRuntime = AgentInstantiationContext.current_runtime()
            self._id = AgentInstantiationContext.current_agent_id()
        if not isinstance(description, str):
            raise ValueError("Agent description must be a string")
        self._description = description

    async def bind_id_and_runtime(self, id: AgentId, runtime: AgentRuntime) -> None:
        if hasattr(self, "_id"):
            if self._id != id:
                raise RuntimeError("Agent is already bound to a different ID")

        if hasattr(self, "_runtime"):
            if self._runtime != runtime:
                raise RuntimeError("Agent is already bound to a different runtime")

        self._id = id
        self._runtime = runtime

    @property
    def type(self) -> str:
        return self.id.type

    @property
    def id(self) -> AgentId:
        return self._id

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    @final
    async def on_message(self, message: Any, ctx: MessageContext) -> Any:
        return await self.on_message_impl(message, ctx)

    @abstractmethod
    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any: ...

    async def send_message(
        self,
        message: Any,
        recipient: AgentId,
        *,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any:
        """See :py:meth:`autogen_core.AgentRuntime.send_message` for more information."""
        if cancellation_token is None:
            cancellation_token = CancellationToken()

        return await self._runtime.send_message(
            message,
            sender=self.id,
            recipient=recipient,
            cancellation_token=cancellation_token,
            message_id=message_id,
        )

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        await self._runtime.publish_message(message, topic_id, sender=self.id, cancellation_token=cancellation_token)

    async def save_state(self) -> Mapping[str, Any]:
        warnings.warn("save_state not implemented", stacklevel=2)
        return {}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        warnings.warn("load_state not implemented", stacklevel=2)
        pass

    async def close(self) -> None:
        pass

    async def register_instance(
        self,
        runtime: AgentRuntime,
        agent_id: AgentId,
        *,
        skip_class_subscriptions: bool = True,
        skip_direct_message_subscription: bool = False,
    ) -> AgentId:
        """
        This function is similar to `register` but is used for registering an instance of an agent. A subscription based on the agent ID is created and added to the runtime.
        """
        agent_id = await runtime.register_agent_instance(agent_instance=self, agent_id=agent_id)

        id_subscription = TypeSubscription(topic_type=agent_id.key, agent_type=agent_id.type)
        await runtime.add_subscription(id_subscription)

        if not skip_class_subscriptions:
            with SubscriptionInstantiationContext.populate_context(AgentType(agent_id.type)):
                subscriptions: List[Subscription] = []
                for unbound_subscription in self._unbound_subscriptions():
                    subscriptions_list_result = unbound_subscription()
                    if inspect.isawaitable(subscriptions_list_result):
                        subscriptions_list = await subscriptions_list_result
                    else:
                        subscriptions_list = subscriptions_list_result

                    subscriptions.extend(subscriptions_list)
            for subscription in subscriptions:
                await runtime.add_subscription(subscription)

        if not skip_direct_message_subscription:
            # Additionally adds a special prefix subscription for this agent to receive direct messages
            try:
                await runtime.add_subscription(
                    TypePrefixSubscription(
                        # The prefix MUST include ":" to avoid collisions with other agents
                        topic_type_prefix=agent_id.type + ":",
                        agent_type=agent_id.type,
                    )
                )
            except ValueError:
                # We don't care if the subscription already exists
                pass

        # TODO: deduplication
        for _message_type, serializer in self._handles_types():
            runtime.add_message_serializer(serializer)

        return agent_id

    @classmethod
    async def register(
        cls,
        runtime: AgentRuntime,
        type: str,
        factory: Callable[[], Self | Awaitable[Self]],
        *,
        skip_class_subscriptions: bool = False,
        skip_direct_message_subscription: bool = False,
    ) -> AgentType:
        agent_type = AgentType(type)
        agent_type = await runtime.register_factory(type=agent_type, agent_factory=factory, expected_class=cls)
        if not skip_class_subscriptions:
            with SubscriptionInstantiationContext.populate_context(agent_type):
                subscriptions: List[Subscription] = []
                for unbound_subscription in cls._unbound_subscriptions():
                    subscriptions_list_result = unbound_subscription()
                    if inspect.isawaitable(subscriptions_list_result):
                        subscriptions_list = await subscriptions_list_result
                    else:
                        subscriptions_list = subscriptions_list_result

                    subscriptions.extend(subscriptions_list)
            for subscription in subscriptions:
                await runtime.add_subscription(subscription)

        if not skip_direct_message_subscription:
            # Additionally adds a special prefix subscription for this agent to receive direct messages
            await runtime.add_subscription(
                TypePrefixSubscription(
                    # The prefix MUST include ":" to avoid collisions with other agents
                    topic_type_prefix=agent_type.type + ":",
                    agent_type=agent_type.type,
                )
            )

        # TODO: deduplication
        for _message_type, serializer in cls._handles_types():
            runtime.add_message_serializer(serializer)

        return agent_type

# From autogen_core/_base_agent.py
def subscription_factory(subscription: UnboundSubscription) -> Callable[[Type[BaseAgentType]], Type[BaseAgentType]]:
    """:meta private:"""

    def decorator(cls: Type[BaseAgentType]) -> Type[BaseAgentType]:
        cls.internal_unbound_subscriptions_list.append(subscription)
        return cls

    return decorator

# From autogen_core/_base_agent.py
def handles(
    type: Type[Any], serializer: MessageSerializer[Any] | List[MessageSerializer[Any]] | None = None
) -> Callable[[Type[BaseAgentType]], Type[BaseAgentType]]:
    def decorator(cls: Type[BaseAgentType]) -> Type[BaseAgentType]:
        if serializer is None:
            serializer_list = try_get_known_serializers_for_type(type)
        else:
            serializer_list = [serializer] if not isinstance(serializer, Sequence) else serializer

        if len(serializer_list) == 0:
            raise ValueError(f"No serializers found for type {type}. Please provide an explicit serializer.")

        cls.internal_extra_handles_types.append((type, serializer_list))
        return cls

    return decorator

# From autogen_core/_base_agent.py
def decorator(cls: Type[BaseAgentType]) -> Type[BaseAgentType]:
        cls.internal_unbound_subscriptions_list.append(subscription)
        return cls

# From autogen_core/_base_agent.py
def type(self) -> str:
        return self.id.type

# From autogen_core/_base_agent.py
def runtime(self) -> AgentRuntime:
        return self._runtime

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

# From autogen_core/_message_handler_context.py
class MessageHandlerContext:
    def __init__(self) -> None:
        raise RuntimeError(
            "MessageHandlerContext cannot be instantiated. It is a static class that provides context management for message handling."
        )

    _MESSAGE_HANDLER_CONTEXT: ClassVar[ContextVar[AgentId]] = ContextVar("_MESSAGE_HANDLER_CONTEXT")

    @classmethod
    @contextmanager
    def populate_context(cls, ctx: AgentId) -> Generator[None, Any, None]:
        """:meta private:"""
        token = MessageHandlerContext._MESSAGE_HANDLER_CONTEXT.set(ctx)
        try:
            yield
        finally:
            MessageHandlerContext._MESSAGE_HANDLER_CONTEXT.reset(token)

    @classmethod
    def agent_id(cls) -> AgentId:
        try:
            return cls._MESSAGE_HANDLER_CONTEXT.get()
        except LookupError as e:
            raise RuntimeError("MessageHandlerContext.agent_id() must be called within a message handler.") from e

# From autogen_core/_message_handler_context.py
def populate_context(cls, ctx: AgentId) -> Generator[None, Any, None]:
        """:meta private:"""
        token = MessageHandlerContext._MESSAGE_HANDLER_CONTEXT.set(ctx)
        try:
            yield
        finally:
            MessageHandlerContext._MESSAGE_HANDLER_CONTEXT.reset(token)

# From autogen_core/_message_handler_context.py
def agent_id(cls) -> AgentId:
        try:
            return cls._MESSAGE_HANDLER_CONTEXT.get()
        except LookupError as e:
            raise RuntimeError("MessageHandlerContext.agent_id() must be called within a message handler.") from e


# From autogen_core/_subscription.py
class Subscription(Protocol):
    """Subscriptions define the topics that an agent is interested in."""

    @property
    def id(self) -> str:
        """Get the ID of the subscription.

        Implementations should return a unique ID for the subscription. Usually this is a UUID.

        Returns:
            str: ID of the subscription.
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Check if two subscriptions are equal.

        Args:
            other (object): Other subscription to compare against.

        Returns:
            bool: True if the subscriptions are equal, False otherwise.
        """
        if not isinstance(other, Subscription):
            return False

        return self.id == other.id

    def is_match(self, topic_id: TopicId) -> bool:
        """Check if a given topic_id matches the subscription.

        Args:
            topic_id (TopicId): TopicId to check.

        Returns:
            bool: True if the topic_id matches the subscription, False otherwise.
        """
        ...

    def map_to_agent(self, topic_id: TopicId) -> AgentId:
        """Map a topic_id to an agent. Should only be called if `is_match` returns True for the given topic_id.

        Args:
            topic_id (TopicId): TopicId to map.

        Returns:
            AgentId: ID of the agent that should handle the topic_id.

        Raises:
            CantHandleException: If the subscription cannot handle the topic_id.
        """
        ...

from typing import TypedDict

# From autogen_core/_agent_metadata.py
class AgentMetadata(TypedDict):
    type: str
    key: str
    description: str


# From autogen_core/_subscription_context.py
class SubscriptionInstantiationContext:
    def __init__(self) -> None:
        raise RuntimeError(
            "SubscriptionInstantiationContext cannot be instantiated. It is a static class that provides context management for subscription instantiation."
        )

    _SUBSCRIPTION_CONTEXT_VAR: ClassVar[ContextVar[AgentType]] = ContextVar("_SUBSCRIPTION_CONTEXT_VAR")

    @classmethod
    @contextmanager
    def populate_context(cls, ctx: AgentType) -> Generator[None, Any, None]:
        """:meta private:"""
        token = SubscriptionInstantiationContext._SUBSCRIPTION_CONTEXT_VAR.set(ctx)
        try:
            yield
        finally:
            SubscriptionInstantiationContext._SUBSCRIPTION_CONTEXT_VAR.reset(token)

    @classmethod
    def agent_type(cls) -> AgentType:
        try:
            return cls._SUBSCRIPTION_CONTEXT_VAR.get()
        except LookupError as e:
            raise RuntimeError(
                "SubscriptionInstantiationContext.runtime() must be called within an instantiation context such as when the AgentRuntime is instantiating an agent. Mostly likely this was caused by directly instantiating an agent instead of using the AgentRuntime to do so."
            ) from e


# From autogen_core/_type_subscription.py
class TypeSubscription(Subscription):
    """This subscription matches on topics based on the type and maps to agents using the source of the topic as the agent key.

    This subscription causes each source to have its own agent instance.

    Example:

        .. code-block:: python

            from autogen_core import TypeSubscription

            subscription = TypeSubscription(topic_type="t1", agent_type="a1")

        In this case:

        - A topic_id with type `t1` and source `s1` will be handled by an agent of type `a1` with key `s1`
        - A topic_id with type `t1` and source `s2` will be handled by an agent of type `a1` with key `s2`.

    Args:
        topic_type (str): Topic type to match against
        agent_type (str): Agent type to handle this subscription
    """

    def __init__(self, topic_type: str, agent_type: str | AgentType, id: str | None = None):
        self._topic_type = topic_type
        if isinstance(agent_type, AgentType):
            self._agent_type = agent_type.type
        else:
            self._agent_type = agent_type
        self._id = id or str(uuid.uuid4())

    @property
    def id(self) -> str:
        return self._id

    @property
    def topic_type(self) -> str:
        return self._topic_type

    @property
    def agent_type(self) -> str:
        return self._agent_type

    def is_match(self, topic_id: TopicId) -> bool:
        return topic_id.type == self._topic_type

    def map_to_agent(self, topic_id: TopicId) -> AgentId:
        if not self.is_match(topic_id):
            raise CantHandleException("TopicId does not match the subscription")

        return AgentId(type=self._agent_type, key=topic_id.source)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeSubscription):
            return False

        return self.id == other.id or (self.agent_type == other.agent_type and self.topic_type == other.topic_type)

# From autogen_core/_type_subscription.py
def topic_type(self) -> str:
        return self._topic_type

import re

# From autogen_core/_agent_id.py
class AgentId:
    """
    Agent ID uniquely identifies an agent instance within an agent runtime - including distributed runtime. It is the 'address' of the agent instance for receiving messages.

    See here for more information: :ref:`agentid_and_lifecycle`
    """

    def __init__(self, type: str | AgentType, key: str) -> None:
        if isinstance(type, AgentType):
            type = type.type

        if not is_valid_agent_type(type):
            raise ValueError(rf"Invalid agent type: {type}. Allowed values MUST match the regex: `^[\w\-\.]+\Z`")

        self._type = type
        self._key = key

    def __hash__(self) -> int:
        return hash((self._type, self._key))

    def __str__(self) -> str:
        return f"{self._type}/{self._key}"

    def __repr__(self) -> str:
        return f'AgentId(type="{self._type}", key="{self._key}")'

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AgentId):
            return False
        return self._type == value.type and self._key == value.key

    @classmethod
    def from_str(cls, agent_id: str) -> Self:
        """Convert a string of the format ``type/key`` into an AgentId"""
        items = agent_id.split("/", maxsplit=1)
        if len(items) != 2:
            raise ValueError(f"Invalid agent id: {agent_id}")
        type, key = items[0], items[1]
        return cls(type, key)

    @property
    def type(self) -> str:
        """
        An identifier that associates an agent with a specific factory function.

        Strings may only be composed of alphanumeric letters (a-z) and (0-9), or underscores (_).
        """
        return self._type

    @property
    def key(self) -> str:
        """
        Agent instance identifier.

        Strings may only be composed of alphanumeric letters (a-z) and (0-9), or underscores (_).
        """
        return self._key

# From autogen_core/_agent_id.py
def is_valid_agent_type(value: str) -> bool:
    return bool(re.match(r"^[\w\-\.]+\Z", value))

# From autogen_core/_agent_id.py
def from_str(cls, agent_id: str) -> Self:
        """Convert a string of the format ``type/key`` into an AgentId"""
        items = agent_id.split("/", maxsplit=1)
        if len(items) != 2:
            raise ValueError(f"Invalid agent id: {agent_id}")
        type, key = items[0], items[1]
        return cls(type, key)

# From autogen_core/_agent_id.py
def key(self) -> str:
        """
        Agent instance identifier.

        Strings may only be composed of alphanumeric letters (a-z) and (0-9), or underscores (_).
        """
        return self._key


# From autogen_core/_agent_instantiation.py
class AgentInstantiationContext:
    """A static class that provides context for agent instantiation.

    This static class can be used to access the current runtime and agent ID
    during agent instantiation -- inside the factory function or the agent's
    class constructor.

    Example:

        Get the current runtime and agent ID inside the factory function and
        the agent's constructor:

        .. code-block:: python

            import asyncio
            from dataclasses import dataclass

            from autogen_core import (
                AgentId,
                AgentInstantiationContext,
                MessageContext,
                RoutedAgent,
                SingleThreadedAgentRuntime,
                message_handler,
            )


            @dataclass
            class TestMessage:
                content: str


            class TestAgent(RoutedAgent):
                def __init__(self, description: str):
                    super().__init__(description)
                    # Get the current runtime -- we don't use it here, but it's available.
                    _ = AgentInstantiationContext.current_runtime()
                    # Get the current agent ID.
                    agent_id = AgentInstantiationContext.current_agent_id()
                    print(f"Current AgentID from constructor: {agent_id}")

                @message_handler
                async def handle_test_message(self, message: TestMessage, ctx: MessageContext) -> None:
                    print(f"Received message: {message.content}")


            def test_agent_factory() -> TestAgent:
                # Get the current runtime -- we don't use it here, but it's available.
                _ = AgentInstantiationContext.current_runtime()
                # Get the current agent ID.
                agent_id = AgentInstantiationContext.current_agent_id()
                print(f"Current AgentID from factory: {agent_id}")
                return TestAgent(description="Test agent")


            async def main() -> None:
                # Create a SingleThreadedAgentRuntime instance.
                runtime = SingleThreadedAgentRuntime()

                # Start the runtime.
                runtime.start()

                # Register the agent type with a factory function.
                await runtime.register_factory("test_agent", test_agent_factory)

                # Send a message to the agent. The runtime will instantiate the agent and call the message handler.
                await runtime.send_message(TestMessage(content="Hello, world!"), AgentId("test_agent", "default"))

                # Stop the runtime.
                await runtime.stop()


            asyncio.run(main())

    """

    def __init__(self) -> None:
        raise RuntimeError(
            "AgentInstantiationContext cannot be instantiated. It is a static class that provides context management for agent instantiation."
        )

    _AGENT_INSTANTIATION_CONTEXT_VAR: ClassVar[ContextVar[tuple[AgentRuntime, AgentId]]] = ContextVar(
        "_AGENT_INSTANTIATION_CONTEXT_VAR"
    )

    @classmethod
    @contextmanager
    def populate_context(cls, ctx: tuple[AgentRuntime, AgentId]) -> Generator[None, Any, None]:
        """:meta private:"""
        token = AgentInstantiationContext._AGENT_INSTANTIATION_CONTEXT_VAR.set(ctx)
        try:
            yield
        finally:
            AgentInstantiationContext._AGENT_INSTANTIATION_CONTEXT_VAR.reset(token)

    @classmethod
    def current_runtime(cls) -> AgentRuntime:
        try:
            return cls._AGENT_INSTANTIATION_CONTEXT_VAR.get()[0]
        except LookupError as e:
            raise RuntimeError(
                "AgentInstantiationContext.runtime() must be called within an instantiation context such as when the AgentRuntime is instantiating an agent. Mostly likely this was caused by directly instantiating an agent instead of using the AgentRuntime to do so."
            ) from e

    @classmethod
    def current_agent_id(cls) -> AgentId:
        try:
            return cls._AGENT_INSTANTIATION_CONTEXT_VAR.get()[1]
        except LookupError as e:
            raise RuntimeError(
                "AgentInstantiationContext.agent_id() must be called within an instantiation context such as when the AgentRuntime is instantiating an agent. Mostly likely this was caused by directly instantiating an agent instead of using the AgentRuntime to do so."
            ) from e

    @classmethod
    def is_in_factory_call(cls) -> bool:
        if cls._AGENT_INSTANTIATION_CONTEXT_VAR.get(None) is None:
            return False
        return True

# From autogen_core/_agent_instantiation.py
def current_runtime(cls) -> AgentRuntime:
        try:
            return cls._AGENT_INSTANTIATION_CONTEXT_VAR.get()[0]
        except LookupError as e:
            raise RuntimeError(
                "AgentInstantiationContext.runtime() must be called within an instantiation context such as when the AgentRuntime is instantiating an agent. Mostly likely this was caused by directly instantiating an agent instead of using the AgentRuntime to do so."
            ) from e

# From autogen_core/_agent_instantiation.py
def current_agent_id(cls) -> AgentId:
        try:
            return cls._AGENT_INSTANTIATION_CONTEXT_VAR.get()[1]
        except LookupError as e:
            raise RuntimeError(
                "AgentInstantiationContext.agent_id() must be called within an instantiation context such as when the AgentRuntime is instantiating an agent. Mostly likely this was caused by directly instantiating an agent instead of using the AgentRuntime to do so."
            ) from e

# From autogen_core/_agent_instantiation.py
def is_in_factory_call(cls) -> bool:
        if cls._AGENT_INSTANTIATION_CONTEXT_VAR.get(None) is None:
            return False
        return True

from enum import Enum

# From autogen_core/logging.py
class LLMCallEvent:
    def __init__(
        self,
        *,
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
        prompt_tokens: int,
        completion_tokens: int,
        **kwargs: Any,
    ) -> None:
        """To be used by model clients to log the call to the LLM.

        Args:
            messages (List[Dict[str, Any]]): The messages used in the call. Must be json serializable.
            response (Dict[str, Any]): The response of the call. Must be json serializable.
            prompt_tokens (int): Number of tokens used in the prompt.
            completion_tokens (int): Number of tokens used in the completion.

        Example:

            .. code-block:: python

                import logging
                from autogen_core import EVENT_LOGGER_NAME
                from autogen_core.logging import LLMCallEvent

                response = {"content": "Hello, world!"}
                messages = [{"role": "user", "content": "Hello, world!"}]
                logger = logging.getLogger(EVENT_LOGGER_NAME)
                logger.info(LLMCallEvent(prompt_tokens=10, completion_tokens=20, response=response, messages=messages))

        """
        self.kwargs = kwargs
        self.kwargs["type"] = "LLMCall"
        self.kwargs["messages"] = messages
        self.kwargs["response"] = response
        self.kwargs["prompt_tokens"] = prompt_tokens
        self.kwargs["completion_tokens"] = completion_tokens
        try:
            agent_id = MessageHandlerContext.agent_id()
        except RuntimeError:
            agent_id = None
        self.kwargs["agent_id"] = None if agent_id is None else str(agent_id)

    @property
    def prompt_tokens(self) -> int:
        return cast(int, self.kwargs["prompt_tokens"])

    @property
    def completion_tokens(self) -> int:
        return cast(int, self.kwargs["completion_tokens"])

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class LLMStreamStartEvent:
    """To be used by model clients to log the start of a stream.

    Args:
        messages (List[Dict[str, Any]]): The messages used in the call. Must be json serializable.

    Example:

        .. code-block:: python

            import logging
            from autogen_core import EVENT_LOGGER_NAME
            from autogen_core.logging import LLMStreamStartEvent

            messages = [{"role": "user", "content": "Hello, world!"}]
            logger = logging.getLogger(EVENT_LOGGER_NAME)
            logger.info(LLMStreamStartEvent(messages=messages))

    """

    def __init__(
        self,
        *,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs
        self.kwargs["type"] = "LLMStreamStart"
        self.kwargs["messages"] = messages
        try:
            agent_id = MessageHandlerContext.agent_id()
        except RuntimeError:
            agent_id = None
        self.kwargs["agent_id"] = None if agent_id is None else str(agent_id)

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class LLMStreamEndEvent:
    def __init__(
        self,
        *,
        response: Dict[str, Any],
        prompt_tokens: int,
        completion_tokens: int,
        **kwargs: Any,
    ) -> None:
        """To be used by model clients to log the end of a stream.

        Args:
            response (Dict[str, Any]): The response of the call. Must be json serializable.
            prompt_tokens (int): Number of tokens used in the prompt.
            completion_tokens (int): Number of tokens used in the completion.

        Example:

            .. code-block:: python

                import logging
                from autogen_core import EVENT_LOGGER_NAME
                from autogen_core.logging import LLMStreamEndEvent

                response = {"content": "Hello, world!"}
                logger = logging.getLogger(EVENT_LOGGER_NAME)
                logger.info(LLMStreamEndEvent(prompt_tokens=10, completion_tokens=20, response=response))

        """
        self.kwargs = kwargs
        self.kwargs["type"] = "LLMStreamEnd"
        self.kwargs["response"] = response
        self.kwargs["prompt_tokens"] = prompt_tokens
        self.kwargs["completion_tokens"] = completion_tokens
        try:
            agent_id = MessageHandlerContext.agent_id()
        except RuntimeError:
            agent_id = None
        self.kwargs["agent_id"] = None if agent_id is None else str(agent_id)

    @property
    def prompt_tokens(self) -> int:
        return cast(int, self.kwargs["prompt_tokens"])

    @property
    def completion_tokens(self) -> int:
        return cast(int, self.kwargs["completion_tokens"])

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class ToolCallEvent:
    def __init__(
        self,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
    ) -> None:
        """Used by subclasses of :class:`~autogen_core.tools.BaseTool` to log executions of tools.

        Args:
            tool_name (str): The name of the tool.
            arguments (Dict[str, Any]): The arguments of the tool. Must be json serializable.
            result (str): The result of the tool. Must be a string.

        Example:

            .. code-block:: python

                from autogen_core import EVENT_LOGGER_NAME
                from autogen_core.logging import ToolCallEvent

                logger = logging.getLogger(EVENT_LOGGER_NAME)
                logger.info(ToolCallEvent(tool_name="Tool1", call_id="123", arguments={"arg1": "value1"}))

        """
        self.kwargs: Dict[str, Any] = {}
        self.kwargs["type"] = "ToolCall"
        self.kwargs["tool_name"] = tool_name
        self.kwargs["arguments"] = arguments
        self.kwargs["result"] = result
        try:
            agent_id = MessageHandlerContext.agent_id()
        except RuntimeError:
            agent_id = None
        self.kwargs["agent_id"] = None if agent_id is None else str(agent_id)

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class MessageKind(Enum):
    DIRECT = 1
    PUBLISH = 2
    RESPOND = 3

# From autogen_core/logging.py
class DeliveryStage(Enum):
    SEND = 1
    DELIVER = 2

# From autogen_core/logging.py
class MessageEvent:
    def __init__(
        self,
        *,
        payload: str,
        sender: AgentId | None,
        receiver: AgentId | TopicId | None,
        kind: MessageKind,
        delivery_stage: DeliveryStage,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs
        self.kwargs["payload"] = payload
        self.kwargs["sender"] = None if sender is None else str(sender)
        self.kwargs["receiver"] = None if receiver is None else str(receiver)
        self.kwargs["kind"] = str(kind)
        self.kwargs["delivery_stage"] = str(delivery_stage)
        self.kwargs["type"] = "Message"

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class MessageDroppedEvent:
    def __init__(
        self,
        *,
        payload: str,
        sender: AgentId | None,
        receiver: AgentId | TopicId | None,
        kind: MessageKind,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs
        self.kwargs["payload"] = payload
        self.kwargs["sender"] = None if sender is None else str(sender)
        self.kwargs["receiver"] = None if receiver is None else str(receiver)
        self.kwargs["kind"] = str(kind)
        self.kwargs["type"] = "MessageDropped"

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class MessageHandlerExceptionEvent:
    def __init__(
        self,
        *,
        payload: str,
        handling_agent: AgentId,
        exception: BaseException,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs
        self.kwargs["payload"] = payload
        self.kwargs["handling_agent"] = str(handling_agent)
        self.kwargs["exception"] = str(exception)
        self.kwargs["type"] = "MessageHandlerException"

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
class AgentConstructionExceptionEvent:
    def __init__(
        self,
        *,
        agent_id: AgentId,
        exception: BaseException,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs
        self.kwargs["agent_id"] = str(agent_id)
        self.kwargs["exception"] = str(exception)
        self.kwargs["type"] = "AgentConstructionException"

    # This must output the event in a json serializable format
    def __str__(self) -> str:
        return json.dumps(self.kwargs)

# From autogen_core/logging.py
def prompt_tokens(self) -> int:
        return cast(int, self.kwargs["prompt_tokens"])

# From autogen_core/logging.py
def completion_tokens(self) -> int:
        return cast(int, self.kwargs["completion_tokens"])


# From autogen_core/_agent_proxy.py
class AgentProxy:
    """A helper class that allows you to use an :class:`~autogen_core.AgentId` in place of its associated :class:`~autogen_core.Agent`"""

    def __init__(self, agent: AgentId, runtime: AgentRuntime):
        self._agent = agent
        self._runtime = runtime

    @property
    def id(self) -> AgentId:
        """Target agent for this proxy"""
        return self._agent

    @property
    def metadata(self) -> Awaitable[AgentMetadata]:
        """Metadata of the agent."""
        return self._runtime.agent_metadata(self._agent)

    async def send_message(
        self,
        message: Any,
        *,
        sender: AgentId,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any:
        return await self._runtime.send_message(
            message,
            recipient=self._agent,
            sender=sender,
            cancellation_token=cancellation_token,
            message_id=message_id,
        )

    async def save_state(self) -> Mapping[str, Any]:
        """Save the state of the agent. The result must be JSON serializable."""
        return await self._runtime.agent_save_state(self._agent)

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load in the state of the agent obtained from `save_state`.

        Args:
            state (Mapping[str, Any]): State of the agent. Must be JSON serializable.
        """
        await self._runtime.agent_load_state(self._agent, state)

from functools import wraps
from typing import Coroutine
from typing import DefaultDict
from typing import Literal
from typing import Sequence
from typing import get_type_hints
from _base_agent import BaseAgent
from _type_helpers import AnyType
from _type_helpers import get_types

# From autogen_core/_routed_agent.py
class MessageHandler(Protocol[AgentT, ReceivesT, ProducesT]):  # type: ignore
    target_types: Sequence[type]
    produces_types: Sequence[type]
    is_message_handler: Literal[True]
    router: Callable[[ReceivesT, MessageContext], bool]

    # agent_instance binds to self in the method
    @staticmethod
    async def __call__(agent_instance: AgentT, message: ReceivesT, ctx: MessageContext) -> ProducesT: ...

# From autogen_core/_routed_agent.py
class RoutedAgent(BaseAgent):
    """A base class for agents that route messages to handlers based on the type of the message
    and optional matching functions.

    To create a routed agent, subclass this class and add message handlers as methods decorated with
    either :func:`event` or :func:`rpc` decorator.

    Example:

    .. code-block:: python

        from dataclasses import dataclass
        from autogen_core import MessageContext
        from autogen_core import RoutedAgent, event, rpc


        @dataclass
        class Message:
            pass


        @dataclass
        class MessageWithContent:
            content: str


        @dataclass
        class Response:
            pass


        class MyAgent(RoutedAgent):
            def __init__(self):
                super().__init__("MyAgent")

            @event
            async def handle_event_message(self, message: Message, ctx: MessageContext) -> None:
                assert ctx.topic_id is not None
                await self.publish_message(MessageWithContent("event handled"), ctx.topic_id)

            @rpc(match=lambda message, ctx: message.content == "special")  # type: ignore
            async def handle_special_rpc_message(self, message: MessageWithContent, ctx: MessageContext) -> Response:
                return Response()
    """

    def __init__(self, description: str) -> None:
        # Self is already bound to the handlers
        self._handlers: DefaultDict[
            Type[Any],
            List[MessageHandler[RoutedAgent, Any, Any]],
        ] = DefaultDict(list)

        handlers = self._discover_handlers()
        for message_handler in handlers:
            for target_type in message_handler.target_types:
                self._handlers[target_type].append(message_handler)

        super().__init__(description)

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any | None:
        """Handle a message by routing it to the appropriate message handler.
        Do not override this method in subclasses. Instead, add message handlers as methods decorated with
        either the :func:`event` or :func:`rpc` decorator."""
        key_type: Type[Any] = type(message)  # type: ignore
        handlers = self._handlers.get(key_type)  # type: ignore
        if handlers is not None:
            # Iterate over all handlers for this matching message type.
            # Call the first handler whose router returns True and then return the result.
            for h in handlers:
                if h.router(message, ctx):
                    return await h(self, message, ctx)
        return await self.on_unhandled_message(message, ctx)  # type: ignore

    async def on_unhandled_message(self, message: Any, ctx: MessageContext) -> None:
        """Called when a message is received that does not have a matching message handler.
        The default implementation logs an info message."""
        logger.info(f"Unhandled message: {message}")

    @classmethod
    def _discover_handlers(cls) -> Sequence[MessageHandler[Any, Any, Any]]:
        handlers: List[MessageHandler[Any, Any, Any]] = []
        for attr in dir(cls):
            if callable(getattr(cls, attr, None)):
                # Since we are getting it from the class, self is not bound
                handler = getattr(cls, attr)
                if hasattr(handler, "is_message_handler"):
                    handlers.append(cast(MessageHandler[Any, Any, Any], handler))
        return handlers

    @classmethod
    def _handles_types(cls) -> List[Tuple[Type[Any], List[MessageSerializer[Any]]]]:
        # TODO handle deduplication
        handlers = cls._discover_handlers()
        types: List[Tuple[Type[Any], List[MessageSerializer[Any]]]] = []
        types.extend(cls.internal_extra_handles_types)
        for handler in handlers:
            for t in handler.target_types:
                # TODO: support different serializers
                serializers = try_get_known_serializers_for_type(t)
                if len(serializers) == 0:
                    raise ValueError(f"No serializers found for type {t}.")

                types.append((t, try_get_known_serializers_for_type(t)))
        return types

# From autogen_core/_routed_agent.py
def message_handler(
    func: Callable[[AgentT, ReceivesT, MessageContext], Coroutine[Any, Any, ProducesT]],
) -> MessageHandler[AgentT, ReceivesT, ProducesT]: ...

# From autogen_core/_routed_agent.py
def event(
    func: Callable[[AgentT, ReceivesT, MessageContext], Coroutine[Any, Any, None]],
) -> MessageHandler[AgentT, ReceivesT, None]: ...

# From autogen_core/_routed_agent.py
def rpc(
    func: Callable[[AgentT, ReceivesT, MessageContext], Coroutine[Any, Any, ProducesT]],
) -> MessageHandler[AgentT, ReceivesT, ProducesT]: ...


# From autogen_core/_closure_agent.py
class ClosureContext(Protocol):
    @property
    def id(self) -> AgentId: ...

    async def send_message(
        self,
        message: Any,
        recipient: AgentId,
        *,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any: ...

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> None: ...

# From autogen_core/_closure_agent.py
class ClosureAgent(BaseAgent, ClosureContext):
    def __init__(
        self,
        description: str,
        closure: Callable[[ClosureContext, T, MessageContext], Awaitable[Any]],
        *,
        unknown_type_policy: Literal["error", "warn", "ignore"] = "warn",
    ) -> None:
        try:
            runtime = AgentInstantiationContext.current_runtime()
            id = AgentInstantiationContext.current_agent_id()
        except Exception as e:
            raise RuntimeError(
                "ClosureAgent must be instantiated within the context of an AgentRuntime. It cannot be directly instantiated."
            ) from e

        self._runtime: AgentRuntime = runtime
        self._id: AgentId = id
        self._description = description
        handled_types = get_handled_types_from_closure(closure)
        self._expected_types = handled_types
        self._closure = closure
        self._unknown_type_policy = unknown_type_policy
        super().__init__(description)

    @property
    def metadata(self) -> AgentMetadata:
        assert self._id is not None
        return AgentMetadata(
            key=self._id.key,
            type=self._id.type,
            description=self._description,
        )

    @property
    def id(self) -> AgentId:
        return self._id

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        if type(message) not in self._expected_types:
            if self._unknown_type_policy == "warn":
                warnings.warn(
                    f"Message type {type(message)} not in target types {self._expected_types} of {self.id}. Set unknown_type_policy to 'error' to raise an exception, or 'ignore' to suppress this warning.",
                    stacklevel=1,
                )
                return None
            elif self._unknown_type_policy == "error":
                raise CantHandleException(
                    f"Message type {type(message)} not in target types {self._expected_types} of {self.id}. Set unknown_type_policy to 'warn' to suppress this exception, or 'ignore' to suppress this warning."
                )

        return await self._closure(self, message, ctx)

    async def save_state(self) -> Mapping[str, Any]:
        """Closure agents do not have state. So this method always returns an empty dictionary."""
        return {}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Closure agents do not have state. So this method does nothing."""
        pass

    @classmethod
    async def register_closure(
        cls,
        runtime: AgentRuntime,
        type: str,
        closure: Callable[[ClosureContext, T, MessageContext], Awaitable[Any]],
        *,
        unknown_type_policy: Literal["error", "warn", "ignore"] = "warn",
        skip_direct_message_subscription: bool = False,
        description: str = "",
        subscriptions: Callable[[], list[Subscription] | Awaitable[list[Subscription]]] | None = None,
    ) -> AgentType:
        """The closure agent allows you to define an agent using a closure, or function without needing to define a class. It allows values to be extracted out of the runtime.

        The closure can define the type of message which is expected, or `Any` can be used to accept any type of message.

        Example:

        .. code-block:: python

            import asyncio
            from autogen_core import SingleThreadedAgentRuntime, MessageContext, ClosureAgent, ClosureContext
            from dataclasses import dataclass

            from autogen_core._default_subscription import DefaultSubscription
            from autogen_core._default_topic import DefaultTopicId


            @dataclass
            class MyMessage:
                content: str


            async def main():
                queue = asyncio.Queue[MyMessage]()

                async def output_result(_ctx: ClosureContext, message: MyMessage, ctx: MessageContext) -> None:
                    await queue.put(message)

                runtime = SingleThreadedAgentRuntime()
                await ClosureAgent.register_closure(
                    runtime, "output_result", output_result, subscriptions=lambda: [DefaultSubscription()]
                )

                runtime.start()
                await runtime.publish_message(MyMessage("Hello, world!"), DefaultTopicId())
                await runtime.stop_when_idle()

                result = await queue.get()
                print(result)


            asyncio.run(main())


        Args:
            runtime (AgentRuntime): Runtime to register the agent to
            type (str): Agent type of registered agent
            closure (Callable[[ClosureContext, T, MessageContext], Awaitable[Any]]): Closure to handle messages
            unknown_type_policy (Literal["error", "warn", "ignore"], optional): What to do if a type is encountered that does not match the closure type. Defaults to "warn".
            skip_direct_message_subscription (bool, optional): Do not add direct message subscription for this agent. Defaults to False.
            description (str, optional): Description of what agent does. Defaults to "".
            subscriptions (Callable[[], list[Subscription]  |  Awaitable[list[Subscription]]] | None, optional): List of subscriptions for this closure agent. Defaults to None.

        Returns:
            AgentType: Type of the agent that was registered
        """

        def factory() -> ClosureAgent:
            return ClosureAgent(description=description, closure=closure, unknown_type_policy=unknown_type_policy)

        assert len(cls._unbound_subscriptions()) == 0, "Closure agents are expected to have no class subscriptions"
        agent_type = await cls.register(
            runtime=runtime,
            type=type,
            factory=factory,  # type: ignore
            # There should be no need to process class subscriptions, as the closure agent does not have any subscriptions.s
            skip_class_subscriptions=True,
            skip_direct_message_subscription=skip_direct_message_subscription,
        )

        subscriptions_list: List[Subscription] = []
        if subscriptions is not None:
            with SubscriptionInstantiationContext.populate_context(agent_type):
                subscriptions_list_result = subscriptions()
                if inspect.isawaitable(subscriptions_list_result):
                    subscriptions_list.extend(await subscriptions_list_result)
                else:
                    # just ignore mypy here
                    subscriptions_list.extend(subscriptions_list_result)  # type: ignore

        for subscription in subscriptions_list:
            await runtime.add_subscription(subscription)

        handled_types = get_handled_types_from_closure(closure)
        for message_type in handled_types:
            # TODO: support custom serializers
            serializer = try_get_known_serializers_for_type(message_type)
            runtime.add_message_serializer(serializer)

        return agent_type

# From autogen_core/_closure_agent.py
def get_handled_types_from_closure(
    closure: Callable[[ClosureAgent, T, MessageContext], Awaitable[Any]],
) -> Sequence[type]:
    args = inspect.getfullargspec(closure)[0]
    if len(args) != 3:
        raise AssertionError("Closure must have 4 arguments")

    message_arg_name = args[1]

    type_hints = get_type_hints(closure)

    if "return" not in type_hints:
        raise AssertionError("return not found in function signature")

    # Get the type of the message parameter
    target_types = get_types(type_hints[message_arg_name])
    if target_types is None:
        raise AssertionError("Message type not found")

    # print(type_hints)
    return_types = get_types(type_hints["return"])

    if return_types is None:
        raise AssertionError("Return type not found")

    return target_types

# From autogen_core/_closure_agent.py
def factory() -> ClosureAgent:
            return ClosureAgent(description=description, closure=closure, unknown_type_policy=unknown_type_policy)

from collections.abc import Generator
from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind

# From _telemetry/_genai.py
class GenAiOperationNameValues(Enum):
    """Enum for GenAI operation name values."""

    CHAT = "chat"
    CREATE_AGENT = "create_agent"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"
    GENERATE_CONTENT = "generate_content"
    INVOKE_AGENT = "invoke_agent"
    TEXT_COMPLETION = "text_completion"

# From _telemetry/_genai.py
def trace_tool_span(
    tool_name: str,
    *,
    tracer: Optional[trace.Tracer] = None,
    parent: Optional[Span] = None,
    tool_description: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Generator[Span, Any, None]:
    """Context manager to create a span for tool execution following the
    OpenTelemetry Semantic conventions for generative AI systems.

    See the GenAI semantic conventions documentation:
    `OpenTelemetry GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`__

    .. warning::

        The GenAI Semantic Conventions are still in incubation and
        subject to changes in future releases.


    Args:
        tool_name (str): The name of the tool being executed.
        tracer (Optional[trace.Tracer]): The tracer to use for creating the span.
        parent (Optional[Span]): The parent span to link this span to.
        tool_description (Optional[str]): A description of the tool.
        tool_call_id (Optional[str]): A unique identifier for the tool call.
    """
    if tracer is None:
        tracer = trace.get_tracer("autogen-core")
    span_attributes = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        GEN_AI_SYSTEM: GENAI_SYSTEM_AUTOGEN,
        GEN_AI_TOOL_NAME: tool_name,
    }
    if tool_description is not None:
        span_attributes[GEN_AI_TOOL_DESCRIPTION] = tool_description
    if tool_call_id is not None:
        span_attributes[GEN_AI_TOOL_CALL_ID] = tool_call_id
    with tracer.start_as_current_span(
        f"{GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}",
        kind=SpanKind.INTERNAL,
        context=trace.set_span_in_context(parent) if parent else None,
        attributes=span_attributes,
    ) as span:
        try:
            yield span
        except Exception as e:
            # Set the exception details on the span if an error occurs
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            raise

# From _telemetry/_genai.py
def trace_create_agent_span(
    agent_name: str,
    *,
    tracer: Optional[trace.Tracer] = None,
    parent: Optional[Span] = None,
    agent_id: Optional[str] = None,
    agent_description: Optional[str] = None,
) -> Generator[Span, Any, None]:
    """Context manager to create a span for agent creation following the
    OpenTelemetry Semantic conventions for generative AI systems.

    See the GenAI semantic conventions documentation:
    `OpenTelemetry GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`__

    .. warning::

        The GenAI Semantic Conventions are still in incubation and
        subject to changes in future releases.

    Args:
        agent_name (str): The name of the agent being created.
        tracer (Optional[trace.Tracer]): The tracer to use for creating the span.
        parent (Optional[Span]): The parent span to link this span to.
        agent_id (Optional[str]): The unique identifier for the agent.
        agent_description (Optional[str]): A description of the agent.
    """
    if tracer is None:
        tracer = trace.get_tracer("autogen-core")
    span_attributes = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CREATE_AGENT.value,
        GEN_AI_SYSTEM: GENAI_SYSTEM_AUTOGEN,
        GEN_AI_AGENT_NAME: agent_name,
    }
    if agent_id is None:
        # Try to see if we can get the agent ID from the current context
        try:
            agent_id = str(AgentInstantiationContext.current_agent_id())
        except RuntimeError:
            agent_id = None
    if agent_id is not None:
        span_attributes[GEN_AI_AGENT_ID] = agent_id
    if agent_description is not None:
        span_attributes[GEN_AI_AGENT_DESCRIPTION] = agent_description
    with tracer.start_as_current_span(
        f"{GenAiOperationNameValues.CREATE_AGENT.value} {agent_name}",
        kind=SpanKind.CLIENT,
        context=trace.set_span_in_context(parent) if parent else None,
        attributes=span_attributes,
    ) as span:
        try:
            yield span
        except Exception as e:
            # Set the exception details on the span if an error occurs
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            raise

# From _telemetry/_genai.py
def trace_invoke_agent_span(
    agent_name: str,
    *,
    tracer: Optional[trace.Tracer] = None,
    parent: Optional[Span] = None,
    agent_id: Optional[str] = None,
    agent_description: Optional[str] = None,
) -> Generator[Span, Any, None]:
    """Context manager to create a span for invoking an agent following the
    OpenTelemetry Semantic conventions for generative AI systems.

    See the GenAI semantic conventions documentation:
    `OpenTelemetry GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`__

    .. warning::

        The GenAI Semantic Conventions are still in incubation and
        subject to changes in future releases.

    Args:
        agent_name (str): The name of the agent being invoked.
        tracer (Optional[trace.Tracer]): The tracer to use for creating the span.
        parent (Optional[Span]): The parent span to link this span to.
        agent_id (Optional[str]): The unique identifier for the agent.
        agent_description (Optional[str]): A description of the agent.
    """
    if tracer is None:
        tracer = trace.get_tracer("autogen-core")
    span_attributes = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        GEN_AI_SYSTEM: GENAI_SYSTEM_AUTOGEN,
        GEN_AI_AGENT_NAME: agent_name,
    }
    if agent_id is not None:
        span_attributes[GEN_AI_AGENT_ID] = agent_id
    if agent_description is not None:
        span_attributes[GEN_AI_AGENT_DESCRIPTION] = agent_description
    with tracer.start_as_current_span(
        f"{GenAiOperationNameValues.INVOKE_AGENT.value} {agent_name}",
        kind=SpanKind.CLIENT,
        context=trace.set_span_in_context(parent) if parent else None,
        attributes=span_attributes,
    ) as span:
        try:
            yield span
        except Exception as e:
            # Set the exception details on the span if an error occurs
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            raise

from  import FunctionCall
from  import MessageContext
from  import RoutedAgent
from  import message_handler
from models import FunctionExecutionResult
from tools import Tool

# From tool_agent/_tool_agent.py
class ToolException(BaseException):
    call_id: str
    content: str
    name: str

# From tool_agent/_tool_agent.py
class ToolNotFoundException(ToolException):
    pass

# From tool_agent/_tool_agent.py
class InvalidToolArgumentsException(ToolException):
    pass

# From tool_agent/_tool_agent.py
class ToolExecutionException(ToolException):
    pass

# From tool_agent/_tool_agent.py
class ToolAgent(RoutedAgent):
    """A tool agent accepts direct messages of the type `FunctionCall`,
    executes the requested tool with the provided arguments, and returns the
    result as `FunctionExecutionResult` messages.

    Args:
        description (str): The description of the agent.
        tools (List[Tool]): The list of tools that the agent can execute.
    """

    def __init__(
        self,
        description: str,
        tools: List[Tool],
    ) -> None:
        super().__init__(description)
        self._tools = tools

    @property
    def tools(self) -> List[Tool]:
        return self._tools

    @message_handler
    async def handle_function_call(self, message: FunctionCall, ctx: MessageContext) -> FunctionExecutionResult:
        """Handles a `FunctionCall` message by executing the requested tool with the provided arguments.

        Args:
            message (FunctionCall): The function call message.
            cancellation_token (CancellationToken): The cancellation token.

        Returns:
            FunctionExecutionResult: The result of the function execution.

        Raises:
            ToolNotFoundException: If the tool is not found.
            InvalidToolArgumentsException: If the tool arguments are invalid.
            ToolExecutionException: If the tool execution fails.
        """
        tool = next((tool for tool in self._tools if tool.name == message.name), None)
        if tool is None:
            raise ToolNotFoundException(
                call_id=message.id, content=f"Error: Tool not found: {message.name}", name=message.name
            )
        else:
            try:
                arguments = json.loads(message.arguments)
                result = await tool.run_json(
                    args=arguments, cancellation_token=ctx.cancellation_token, call_id=message.id
                )
                result_as_str = tool.return_value_as_string(result)
            except json.JSONDecodeError as e:
                raise InvalidToolArgumentsException(
                    call_id=message.id, content=f"Error: Invalid arguments: {message.arguments}", name=message.name
                ) from e
            except Exception as e:
                raise ToolExecutionException(call_id=message.id, content=f"Error: {e}", name=message.name) from e
        return FunctionExecutionResult(content=result_as_str, call_id=message.id, is_error=False, name=message.name)

# From tool_agent/_tool_agent.py
def tools(self) -> List[Tool]:
        return self._tools

from datetime import datetime
from datetime import timezone
from typing import Generic
from autogen_core import Component
from autogen_core import ComponentBase
from autogen_core import FunctionCall
from autogen_core import Image
from autogen_core.code_executor import CodeBlock
from autogen_core.code_executor import CodeResult
from autogen_core.memory import MemoryContent
from autogen_core.models import FunctionExecutionResult
from autogen_core.models import LLMMessage
from autogen_core.models import RequestUsage
from autogen_core.models import UserMessage
from autogen_core.utils import schema_to_pydantic_model
from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field
from typing_extensions import Annotated

# From autogen_agentchat/messages.py
class BaseMessage(BaseModel, ABC):
    """Abstract base class for all message types in AgentChat.

    .. warning::

        If you want to create a new message type, do not inherit from this class.
        Instead, inherit from :class:`BaseChatMessage` or :class:`BaseAgentEvent`
        to clarify the purpose of the message type.

    """

    @abstractmethod
    def to_text(self) -> str:
        """Convert the message content to a string-only representation
        that can be rendered in the console and inspected by the user or conditions.
        This is not used for creating text-only content for models.
        For :class:`BaseChatMessage` types, use :meth:`to_model_text` instead."""
        ...

    def dump(self) -> Mapping[str, Any]:
        """Convert the message to a JSON-serializable dictionary.

        The default implementation uses the Pydantic model's
        :meth:`model_dump` method to convert the message to a dictionary.
        Datetime objects are automatically converted to ISO format strings
        to ensure JSON serialization compatibility.
        Override this method if you want to customize the serialization
        process or add additional fields to the output.
        """
        return self.model_dump(mode="json")

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> Self:
        """Create a message from a dictionary of JSON-serializable data.

        The default implementation uses the Pydantic model's
        :meth:`model_validate` method to create the message from the data.
        Override this method if you want to customize the deserialization
        process or add additional fields to the input data."""
        return cls.model_validate(data)

# From autogen_agentchat/messages.py
class BaseChatMessage(BaseMessage, ABC):
    """Abstract base class for chat messages.

    .. note::

        If you want to create a new message type that is used for agent-to-agent
        communication, inherit from this class, or simply use
        :class:`StructuredMessage` if your content type is a subclass of
        Pydantic BaseModel.

    This class is used for messages that are sent between agents in a chat
    conversation. Agents are expected to process the content of the
    message using models and return a response as another :class:`BaseChatMessage`.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this message."""

    source: str
    """The name of the agent that sent this message."""

    models_usage: RequestUsage | None = None
    """The model client usage incurred when producing this message."""

    metadata: Dict[str, str] = {}
    """Additional metadata about the message."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """The time when the message was created."""

    @abstractmethod
    def to_model_text(self) -> str:
        """Convert the content of the message to text-only representation.
        This is used for creating text-only content for models.

        This is not used for rendering the message in console. For that, use
        :meth:`~BaseMessage.to_text`.

        The difference between this and :meth:`to_model_message` is that this
        is used to construct parts of the a message for the model client,
        while :meth:`to_model_message` is used to create a complete message
        for the model client.
        """
        ...

    @abstractmethod
    def to_model_message(self) -> UserMessage:
        """Convert the message content to a :class:`~autogen_core.models.UserMessage`
        for use with model client, e.g., :class:`~autogen_core.models.ChatCompletionClient`.
        """
        ...

# From autogen_agentchat/messages.py
class BaseTextChatMessage(BaseChatMessage, ABC):
    """Base class for all text-only :class:`BaseChatMessage` types.
    It has implementations for :meth:`to_text`, :meth:`to_model_text`,
    and :meth:`to_model_message` methods.

    Inherit from this class if your message content type is a string.
    """

    content: str
    """The content of the message."""

    def to_text(self) -> str:
        return self.content

    def to_model_text(self) -> str:
        return self.content

    def to_model_message(self) -> UserMessage:
        return UserMessage(content=self.content, source=self.source)

# From autogen_agentchat/messages.py
class BaseAgentEvent(BaseMessage, ABC):
    """Base class for agent events.

    .. note::

        If you want to create a new message type for signaling observable events
        to user and application, inherit from this class.

    Agent events are used to signal actions and thoughts produced by agents
    and teams to user and applications. They are not used for agent-to-agent
    communication and are not expected to be processed by other agents.

    You should override the :meth:`to_text` method if you want to provide
    a custom rendering of the content.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this event."""

    source: str
    """The name of the agent that sent this message."""

    models_usage: RequestUsage | None = None
    """The model client usage incurred when producing this message."""

    metadata: Dict[str, str] = {}
    """Additional metadata about the message."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """The time when the message was created."""

# From autogen_agentchat/messages.py
class StructuredMessage(BaseChatMessage, Generic[StructuredContentType]):
    """A :class:`BaseChatMessage` type with an unspecified content type.

    To create a new structured message type, specify the content type
    as a subclass of `Pydantic BaseModel <https://docs.pydantic.dev/latest/concepts/models/>`_.

    .. code-block:: python

        from pydantic import BaseModel
        from autogen_agentchat.messages import StructuredMessage


        class MyMessageContent(BaseModel):
            text: str
            number: int


        message = StructuredMessage[MyMessageContent](
            content=MyMessageContent(text="Hello", number=42),
            source="agent1",
        )

        print(message.to_text())  # {"text": "Hello", "number": 42}

    .. code-block:: python

        from pydantic import BaseModel
        from autogen_agentchat.messages import StructuredMessage


        class MyMessageContent(BaseModel):
            text: str
            number: int


        message = StructuredMessage[MyMessageContent](
            content=MyMessageContent(text="Hello", number=42),
            source="agent",
            format_string="Hello, {text} {number}!",
        )

        print(message.to_text())  # Hello, agent 42!

    """

    content: StructuredContentType
    """The content of the message. Must be a subclass of
    `Pydantic BaseModel <https://docs.pydantic.dev/latest/concepts/models/>`_."""

    format_string: Optional[str] = None
    """(Experimental) An optional format string to render the content into a human-readable format.
    The format string can use the fields of the content model as placeholders.
    For example, if the content model has a field `name`, you can use
    `{name}` in the format string to include the value of that field.
    The format string is used in the :meth:`to_text` method to create a
    human-readable representation of the message.
    This setting is experimental and will change in the future.
    """

    @computed_field
    def type(self) -> str:
        return self.__class__.__name__

    def to_text(self) -> str:
        if self.format_string is not None:
            return self.format_string.format(**self.content.model_dump())
        else:
            return self.content.model_dump_json()

    def to_model_text(self) -> str:
        if self.format_string is not None:
            return self.format_string.format(**self.content.model_dump())
        else:
            return self.content.model_dump_json()

    def to_model_message(self) -> UserMessage:
        return UserMessage(
            content=self.content.model_dump_json(),
            source=self.source,
        )

# From autogen_agentchat/messages.py
class StructureMessageConfig(BaseModel):
    """The declarative configuration for the structured output."""

    json_schema: Dict[str, Any]
    format_string: Optional[str] = None
    content_model_name: str

# From autogen_agentchat/messages.py
class StructuredMessageFactory(ComponentBase[StructureMessageConfig], Component[StructureMessageConfig]):
    """:meta private:

    A component that creates structured chat messages from Pydantic models or JSON schemas.

    This component helps you generate strongly-typed chat messages with content defined using a Pydantic model.
    It can be used in declarative workflows where message structure must be validated, formatted, and serialized.

    You can initialize the component directly using a `BaseModel` subclass, or dynamically from a configuration
    object (e.g., loaded from disk or a database).

    ### Example 1: Create from a Pydantic Model

    .. code-block:: python

        from pydantic import BaseModel
        from autogen_agentchat.messages import StructuredMessageFactory


        class TestContent(BaseModel):
            field1: str
            field2: int


        format_string = "This is a string {field1} and this is an int {field2}"
        sm_component = StructuredMessageFactory(input_model=TestContent, format_string=format_string)

        message = sm_component.StructuredMessage(
            source="test_agent", content=TestContent(field1="Hello", field2=42), format_string=format_string
        )

        print(message.to_model_text())  # Output: This is a string Hello and this is an int 42

        config = sm_component.dump_component()

        s_m_dyn = StructuredMessageFactory.load_component(config)
        message = s_m_dyn.StructuredMessage(
            source="test_agent",
            content=s_m_dyn.ContentModel(field1="dyn agent", field2=43),
            format_string=s_m_dyn.format_string,
        )
        print(type(message))  # StructuredMessage[GeneratedModel]
        print(message.to_model_text())  # Output: This is a string dyn agent and this is an int 43

    Attributes:
        component_config_schema (StructureMessageConfig): Defines the configuration structure for this component.
        component_provider_override (str): Path used to reference this component in external tooling.
        component_type (str): Identifier used for categorization (e.g., "structured_message").

    Raises:
        ValueError: If neither `json_schema` nor `input_model` is provided.

    Args:
        json_schema (Optional[str]): JSON schema to dynamically create a Pydantic model.
        input_model (Optional[Type[BaseModel]]): A subclass of `BaseModel` that defines the expected message structure.
        format_string (Optional[str]): Optional string to render content into a human-readable format.
        content_model_name (Optional[str]): Optional name for the generated Pydantic model.
    """

    component_config_schema = StructureMessageConfig
    component_provider_override = "autogen_agentchat.messages.StructuredMessageFactory"
    component_type = "structured_message"

    def __init__(
        self,
        json_schema: Optional[Dict[str, Any]] = None,
        input_model: Optional[Type[BaseModel]] = None,
        format_string: Optional[str] = None,
        content_model_name: Optional[str] = None,
    ) -> None:
        self.format_string = format_string

        if json_schema:
            self.ContentModel = schema_to_pydantic_model(
                json_schema, model_name=content_model_name or "GeneratedContentModel"
            )
        elif input_model:
            self.ContentModel = input_model
        else:
            raise ValueError("Either `json_schema` or `input_model` must be provided.")

        self.StructuredMessage = StructuredMessage[self.ContentModel]  # type: ignore[name-defined]

    def _to_config(self) -> StructureMessageConfig:
        return StructureMessageConfig(
            json_schema=self.ContentModel.model_json_schema(),
            format_string=self.format_string,
            content_model_name=self.ContentModel.__name__,
        )

    @classmethod
    def _from_config(cls, config: StructureMessageConfig) -> "StructuredMessageFactory":
        return cls(
            json_schema=config.json_schema,
            format_string=config.format_string,
            content_model_name=config.content_model_name,
        )

# From autogen_agentchat/messages.py
class TextMessage(BaseTextChatMessage):
    """A text message with string-only content."""

    type: Literal["TextMessage"] = "TextMessage"

# From autogen_agentchat/messages.py
class MultiModalMessage(BaseChatMessage):
    """A multimodal message."""

    content: List[str | Image]
    """The content of the message."""

    type: Literal["MultiModalMessage"] = "MultiModalMessage"

    def to_model_text(self, image_placeholder: str | None = "[image]") -> str:
        """Convert the content of the message to a string-only representation.
        If an image is present, it will be replaced with the image placeholder
        by default, otherwise it will be a base64 string when set to None.
        """
        text = ""
        for c in self.content:
            if isinstance(c, str):
                text += c
            elif isinstance(c, Image):
                if image_placeholder is not None:
                    text += f" {image_placeholder}"
                else:
                    text += f" {c.to_base64()}"
        return text

    def to_text(self, iterm: bool = False) -> str:
        result: List[str] = []
        for c in self.content:
            if isinstance(c, str):
                result.append(c)
            else:
                if iterm:
                    # iTerm2 image rendering protocol: https://iterm2.com/documentation-images.html
                    image_data = c.to_base64()
                    result.append(f"\033]1337;File=inline=1:{image_data}\a\n")
                else:
                    result.append("<image>")
        return "\n".join(result)

    def to_model_message(self) -> UserMessage:
        return UserMessage(content=self.content, source=self.source)

# From autogen_agentchat/messages.py
class StopMessage(BaseTextChatMessage):
    """A message requesting stop of a conversation."""

    type: Literal["StopMessage"] = "StopMessage"

# From autogen_agentchat/messages.py
class HandoffMessage(BaseTextChatMessage):
    """A message requesting handoff of a conversation to another agent."""

    target: str
    """The name of the target agent to handoff to."""

    context: List[LLMMessage] = []
    """The model context to be passed to the target agent."""

    type: Literal["HandoffMessage"] = "HandoffMessage"

# From autogen_agentchat/messages.py
class ToolCallSummaryMessage(BaseTextChatMessage):
    """A message signaling the summary of tool call results."""

    type: Literal["ToolCallSummaryMessage"] = "ToolCallSummaryMessage"

    tool_calls: List[FunctionCall]
    """The tool calls that were made."""

    results: List[FunctionExecutionResult]
    """The results of the tool calls."""

# From autogen_agentchat/messages.py
class ToolCallRequestEvent(BaseAgentEvent):
    """An event signaling a request to use tools."""

    content: List[FunctionCall]
    """The tool calls."""

    type: Literal["ToolCallRequestEvent"] = "ToolCallRequestEvent"

    def to_text(self) -> str:
        return str(self.content)

# From autogen_agentchat/messages.py
class CodeGenerationEvent(BaseAgentEvent):
    """An event signaling code generation event."""

    retry_attempt: int
    "Retry number, 0 means first generation"

    content: str
    "The complete content as string."

    code_blocks: List[CodeBlock]
    "List of code blocks present in content"

    type: Literal["CodeGenerationEvent"] = "CodeGenerationEvent"

    def to_text(self) -> str:
        return self.content

# From autogen_agentchat/messages.py
class CodeExecutionEvent(BaseAgentEvent):
    """An event signaling code execution event."""

    retry_attempt: int
    "Retry number, 0 means first execution"

    result: CodeResult
    "Code Execution Result"

    type: Literal["CodeExecutionEvent"] = "CodeExecutionEvent"

    def to_text(self) -> str:
        return self.result.output

# From autogen_agentchat/messages.py
class ToolCallExecutionEvent(BaseAgentEvent):
    """An event signaling the execution of tool calls."""

    content: List[FunctionExecutionResult]
    """The tool call results."""

    type: Literal["ToolCallExecutionEvent"] = "ToolCallExecutionEvent"

    def to_text(self) -> str:
        return str(self.content)

# From autogen_agentchat/messages.py
class UserInputRequestedEvent(BaseAgentEvent):
    """An event signaling a that the user proxy has requested user input. Published prior to invoking the input callback."""

    request_id: str
    """Identifier for the user input request."""

    content: Literal[""] = ""
    """Empty content for compat with consumers expecting a content field."""

    type: Literal["UserInputRequestedEvent"] = "UserInputRequestedEvent"

    def to_text(self) -> str:
        return str(self.content)

# From autogen_agentchat/messages.py
class MemoryQueryEvent(BaseAgentEvent):
    """An event signaling the results of memory queries."""

    content: List[MemoryContent]
    """The memory query results."""

    type: Literal["MemoryQueryEvent"] = "MemoryQueryEvent"

    def to_text(self) -> str:
        return str(self.content)

# From autogen_agentchat/messages.py
class ModelClientStreamingChunkEvent(BaseAgentEvent):
    """An event signaling a text output chunk from a model client in streaming mode."""

    content: str
    """A string chunk from the model client."""

    full_message_id: str | None = None
    """Optional reference to the complete message that may come after the chunks.
    This allows consumers of the stream to correlate chunks with the eventual completed message."""

    type: Literal["ModelClientStreamingChunkEvent"] = "ModelClientStreamingChunkEvent"

    def to_text(self) -> str:
        return self.content

# From autogen_agentchat/messages.py
class ThoughtEvent(BaseAgentEvent):
    """An event signaling the thought process of a model.
    It is used to communicate the reasoning tokens generated by a reasoning model,
    or the extra text content generated by a function call."""

    content: str
    """The thought process of the model."""

    type: Literal["ThoughtEvent"] = "ThoughtEvent"

    def to_text(self) -> str:
        return self.content

# From autogen_agentchat/messages.py
class SelectSpeakerEvent(BaseAgentEvent):
    """An event signaling the selection of speakers for a conversation."""

    content: List[str]
    """The names of the selected speakers."""

    type: Literal["SelectSpeakerEvent"] = "SelectSpeakerEvent"

    def to_text(self) -> str:
        return str(self.content)

# From autogen_agentchat/messages.py
class SelectorEvent(BaseAgentEvent):
    """An event emitted from the `SelectorGroupChat`."""

    content: str
    """The content of the event."""

    type: Literal["SelectorEvent"] = "SelectorEvent"

    def to_text(self) -> str:
        return str(self.content)

# From autogen_agentchat/messages.py
class MessageFactory:
    """:meta private:

    A factory for creating messages from JSON-serializable dictionaries.

    This is useful for deserializing messages from JSON data.
    """

    def __init__(self) -> None:
        self._message_types: Dict[str, type[BaseAgentEvent | BaseChatMessage]] = {}
        # Register all message types.
        self._message_types[TextMessage.__name__] = TextMessage
        self._message_types[MultiModalMessage.__name__] = MultiModalMessage
        self._message_types[StopMessage.__name__] = StopMessage
        self._message_types[ToolCallSummaryMessage.__name__] = ToolCallSummaryMessage
        self._message_types[HandoffMessage.__name__] = HandoffMessage
        self._message_types[ToolCallRequestEvent.__name__] = ToolCallRequestEvent
        self._message_types[ToolCallExecutionEvent.__name__] = ToolCallExecutionEvent
        self._message_types[MemoryQueryEvent.__name__] = MemoryQueryEvent
        self._message_types[UserInputRequestedEvent.__name__] = UserInputRequestedEvent
        self._message_types[ModelClientStreamingChunkEvent.__name__] = ModelClientStreamingChunkEvent
        self._message_types[ThoughtEvent.__name__] = ThoughtEvent
        self._message_types[SelectSpeakerEvent.__name__] = SelectSpeakerEvent
        self._message_types[CodeGenerationEvent.__name__] = CodeGenerationEvent
        self._message_types[CodeExecutionEvent.__name__] = CodeExecutionEvent

    def is_registered(self, message_type: type[BaseAgentEvent | BaseChatMessage]) -> bool:
        """Check if a message type is registered with the factory."""
        # Get the class name of the message type.
        class_name = message_type.__name__
        # Check if the class name is already registered.
        return class_name in self._message_types

    def register(self, message_type: type[BaseAgentEvent | BaseChatMessage]) -> None:
        """Register a new message type with the factory."""
        if self.is_registered(message_type):
            raise ValueError(f"Message type {message_type} is already registered.")
        if not issubclass(message_type, BaseChatMessage) and not issubclass(message_type, BaseAgentEvent):
            raise ValueError(f"Message type {message_type} must be a subclass of BaseChatMessage or BaseAgentEvent.")
        # Get the class name of the
        class_name = message_type.__name__
        # Check if the class name is already registered.
        # Register the message type.
        self._message_types[class_name] = message_type

    def create(self, data: Mapping[str, Any]) -> BaseAgentEvent | BaseChatMessage:
        """Create a message from a dictionary of JSON-serializable data."""
        # Get the type of the message from the dictionary.
        message_type = data.get("type")
        if message_type is None:
            raise ValueError("Field 'type' is required in the message data to recover the message type.")
        if message_type not in self._message_types:
            raise ValueError(f"Unknown message type: {message_type}")
        if not isinstance(message_type, str):
            raise ValueError(f"Message type must be a string, got {type(message_type)}")

        # Get the class for the message type.
        message_class = self._message_types[message_type]

        # Create an instance of the message class.
        assert issubclass(message_class, BaseChatMessage) or issubclass(message_class, BaseAgentEvent)
        return message_class.load(data)

# From autogen_agentchat/messages.py
def to_text(self) -> str:
        """Convert the message content to a string-only representation
        that can be rendered in the console and inspected by the user or conditions.
        This is not used for creating text-only content for models.
        For :class:`BaseChatMessage` types, use :meth:`to_model_text` instead."""
        ...

# From autogen_agentchat/messages.py
def dump(self) -> Mapping[str, Any]:
        """Convert the message to a JSON-serializable dictionary.

        The default implementation uses the Pydantic model's
        :meth:`model_dump` method to convert the message to a dictionary.
        Datetime objects are automatically converted to ISO format strings
        to ensure JSON serialization compatibility.
        Override this method if you want to customize the serialization
        process or add additional fields to the output.
        """
        return self.model_dump(mode="json")

# From autogen_agentchat/messages.py
def load(cls, data: Mapping[str, Any]) -> Self:
        """Create a message from a dictionary of JSON-serializable data.

        The default implementation uses the Pydantic model's
        :meth:`model_validate` method to create the message from the data.
        Override this method if you want to customize the deserialization
        process or add additional fields to the input data."""
        return cls.model_validate(data)

# From autogen_agentchat/messages.py
def to_model_text(self) -> str:
        """Convert the content of the message to text-only representation.
        This is used for creating text-only content for models.

        This is not used for rendering the message in console. For that, use
        :meth:`~BaseMessage.to_text`.

        The difference between this and :meth:`to_model_message` is that this
        is used to construct parts of the a message for the model client,
        while :meth:`to_model_message` is used to create a complete message
        for the model client.
        """
        ...

# From autogen_agentchat/messages.py
def to_model_message(self) -> UserMessage:
        """Convert the message content to a :class:`~autogen_core.models.UserMessage`
        for use with model client, e.g., :class:`~autogen_core.models.ChatCompletionClient`.
        """
        ...

# From autogen_agentchat/messages.py
def is_registered(self, message_type: type[BaseAgentEvent | BaseChatMessage]) -> bool:
        """Check if a message type is registered with the factory."""
        # Get the class name of the message type.
        class_name = message_type.__name__
        # Check if the class name is already registered.
        return class_name in self._message_types

# From autogen_agentchat/messages.py
def register(self, message_type: type[BaseAgentEvent | BaseChatMessage]) -> None:
        """Register a new message type with the factory."""
        if self.is_registered(message_type):
            raise ValueError(f"Message type {message_type} is already registered.")
        if not issubclass(message_type, BaseChatMessage) and not issubclass(message_type, BaseAgentEvent):
            raise ValueError(f"Message type {message_type} must be a subclass of BaseChatMessage or BaseAgentEvent.")
        # Get the class name of the
        class_name = message_type.__name__
        # Check if the class name is already registered.
        # Register the message type.
        self._message_types[class_name] = message_type

# From autogen_agentchat/messages.py
def create(self, data: Mapping[str, Any]) -> BaseAgentEvent | BaseChatMessage:
        """Create a message from a dictionary of JSON-serializable data."""
        # Get the type of the message from the dictionary.
        message_type = data.get("type")
        if message_type is None:
            raise ValueError("Field 'type' is required in the message data to recover the message type.")
        if message_type not in self._message_types:
            raise ValueError(f"Unknown message type: {message_type}")
        if not isinstance(message_type, str):
            raise ValueError(f"Message type must be a string, got {type(message_type)}")

        # Get the class for the message type.
        message_class = self._message_types[message_type]

        # Create an instance of the message class.
        assert issubclass(message_class, BaseChatMessage) or issubclass(message_class, BaseAgentEvent)
        return message_class.load(data)

from autogen_agentchat.agents import BaseChatAgent
from _task_runner_tool import TaskRunnerTool

# From tools/_agent.py
class AgentToolConfig(BaseModel):
    """Configuration for the AgentTool."""

    agent: ComponentModel
    """The agent to be used for running the task."""

    return_value_as_last_message: bool = False
    """Whether to return the value as the last message of the task result."""

# From tools/_agent.py
class AgentTool(TaskRunnerTool, Component[AgentToolConfig]):
    """Tool that can be used to run a task using an agent.

    The tool returns the result of the task execution as a :class:`~autogen_agentchat.base.TaskResult` object.

    .. important::
        When using AgentTool, you **must** disable parallel tool calls in the model client configuration
        to avoid concurrency issues. Agents cannot run concurrently as they maintain internal state
        that would conflict with parallel execution. For example, set ``parallel_tool_calls=False``
        for :class:`~autogen_ext.models.openai.OpenAIChatCompletionClient` and
        :class:`~autogen_ext.models.openai.AzureOpenAIChatCompletionClient`.

    Args:
        agent (BaseChatAgent): The agent to be used for running the task.
        return_value_as_last_message (bool): Whether to use the last message content of the task result
            as the return value of the tool in :meth:`~autogen_agentchat.tools.TaskRunnerTool.return_value_as_string`.
            If set to True, the last message content will be returned as a string.
            If set to False, the tool will return all messages in the task result as a string concatenated together,
            with each message prefixed by its source (e.g., "writer: ...", "assistant: ...").

    Example:

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.tools import AgentTool
            from autogen_agentchat.ui import Console
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4.1")
                writer = AssistantAgent(
                    name="writer",
                    description="A writer agent for generating text.",
                    model_client=model_client,
                    system_message="Write well.",
                )
                writer_tool = AgentTool(agent=writer)

                # Create model client with parallel tool calls disabled for the main agent
                main_model_client = OpenAIChatCompletionClient(model="gpt-4.1", parallel_tool_calls=False)
                assistant = AssistantAgent(
                    name="assistant",
                    model_client=main_model_client,
                    tools=[writer_tool],
                    system_message="You are a helpful assistant.",
                )
                await Console(assistant.run_stream(task="Write a poem about the sea."))


            asyncio.run(main())
    """

    component_config_schema = AgentToolConfig
    component_provider_override = "autogen_agentchat.tools.AgentTool"

    def __init__(self, agent: BaseChatAgent, return_value_as_last_message: bool = False) -> None:
        self._agent = agent
        super().__init__(
            agent, agent.name, agent.description, return_value_as_last_message=return_value_as_last_message
        )

    def _to_config(self) -> AgentToolConfig:
        return AgentToolConfig(
            agent=self._agent.dump_component(),
            return_value_as_last_message=self._return_value_as_last_message,
        )

    @classmethod
    def _from_config(cls, config: AgentToolConfig) -> Self:
        return cls(BaseChatAgent.load_component(config.agent), config.return_value_as_last_message)

from typing import AsyncGenerator
from autogen_core import CancellationToken
from pydantic import SerializeAsAny
from messages import BaseAgentEvent
from messages import BaseChatMessage
from _task import TaskRunner

# From base/_chat_agent.py
class Response:
    """A response from calling :meth:`ChatAgent.on_messages`."""

    chat_message: SerializeAsAny[BaseChatMessage]
    """A chat message produced by the agent as the response."""

    inner_messages: Sequence[SerializeAsAny[BaseAgentEvent | BaseChatMessage]] | None = None
    """Inner messages produced by the agent, they can be :class:`BaseAgentEvent`
    or :class:`BaseChatMessage`."""

# From base/_chat_agent.py
class ChatAgent(ABC, TaskRunner, ComponentBase[BaseModel]):
    """Protocol for a chat agent."""

    component_type = "agent"

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the agent. This is used by team to uniquely identify
        the agent. It should be unique within the team."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """The description of the agent. This is used by team to
        make decisions about which agents to use. The description should
        describe the agent's capabilities and how to interact with it."""
        ...

    @property
    @abstractmethod
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the agent produces in the
        :attr:`Response.chat_message` field. They must be :class:`BaseChatMessage` types."""
        ...

    @abstractmethod
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Handles incoming messages and returns a response."""
        ...

    @abstractmethod
    def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Handles incoming messages and returns a stream of inner messages and
        and the final item is the response."""
        ...

    @abstractmethod
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Resets the agent to its initialization state."""
        ...

    @abstractmethod
    async def on_pause(self, cancellation_token: CancellationToken) -> None:
        """Called when the agent is paused. The agent may be running in :meth:`on_messages` or
        :meth:`on_messages_stream` when this method is called."""
        ...

    @abstractmethod
    async def on_resume(self, cancellation_token: CancellationToken) -> None:
        """Called when the agent is resumed. The agent may be running in :meth:`on_messages` or
        :meth:`on_messages_stream` when this method is called."""
        ...

    @abstractmethod
    async def save_state(self) -> Mapping[str, Any]:
        """Save agent state for later restoration"""
        ...

    @abstractmethod
    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore agent from saved state"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the agent."""
        ...

# From base/_chat_agent.py
def name(self) -> str:
        """The name of the agent. This is used by team to uniquely identify
        the agent. It should be unique within the team."""
        ...

# From base/_chat_agent.py
def description(self) -> str:
        """The description of the agent. This is used by team to
        make decisions about which agents to use. The description should
        describe the agent's capabilities and how to interact with it."""
        ...

# From base/_chat_agent.py
def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the agent produces in the
        :attr:`Response.chat_message` field. They must be :class:`BaseChatMessage` types."""
        ...

# From base/_chat_agent.py
def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Handles incoming messages and returns a stream of inner messages and
        and the final item is the response."""
        ...

from typing import Union
from autogen_core.memory import Memory
from autogen_core.model_context import ChatCompletionContext
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage
from autogen_core.models import ChatCompletionClient
from autogen_core.models import CreateResult
from autogen_core.models import FunctionExecutionResultMessage
from autogen_core.models import SystemMessage
from autogen_core.tools import BaseTool
from autogen_core.tools import FunctionTool
from autogen_core.tools import StaticStreamWorkbench
from autogen_core.tools import ToolResult
from autogen_core.tools import Workbench
from  import EVENT_LOGGER_NAME
from base import Handoff
from base import Response
from messages import HandoffMessage
from messages import MemoryQueryEvent
from messages import ModelClientStreamingChunkEvent
from messages import StructuredMessage
from messages import StructuredMessageFactory
from messages import TextMessage
from messages import ThoughtEvent
from messages import ToolCallExecutionEvent
from messages import ToolCallRequestEvent
from messages import ToolCallSummaryMessage
from state import AssistantAgentState
from utils import remove_images
from _base_chat_agent import BaseChatAgent

# From agents/_assistant_agent.py
class AssistantAgentConfig(BaseModel):
    """The declarative configuration for the assistant agent."""

    name: str
    model_client: ComponentModel
    tools: List[ComponentModel] | None = None
    workbench: List[ComponentModel] | None = None
    handoffs: List[HandoffBase | str] | None = None
    model_context: ComponentModel | None = None
    memory: List[ComponentModel] | None = None
    description: str
    system_message: str | None = None
    model_client_stream: bool = False
    reflect_on_tool_use: bool
    tool_call_summary_format: str
    max_tool_iterations: int = Field(default=1, ge=1)
    metadata: Dict[str, str] | None = None
    structured_message_factory: ComponentModel | None = None

# From agents/_assistant_agent.py
class AssistantAgent(BaseChatAgent, Component[AssistantAgentConfig]):
    """An agent that provides assistance with tool use.
    The :meth:`on_messages` returns a :class:`~autogen_agentchat.base.Response`
    in which :attr:`~autogen_agentchat.base.Response.chat_message` is the final
    response message.

    The :meth:`on_messages_stream` creates an async generator that produces
    the inner messages as they are created, and the :class:`~autogen_agentchat.base.Response`
    object as the last item before closing the generator.

    The :meth:`BaseChatAgent.run` method returns a :class:`~autogen_agentchat.base.TaskResult`
    containing the messages produced by the agent. In the list of messages,
    :attr:`~autogen_agentchat.base.TaskResult.messages`,
    the last message is the final response message.

    The :meth:`BaseChatAgent.run_stream` method creates an async generator that produces
    the inner messages as they are created, and the :class:`~autogen_agentchat.base.TaskResult`
    object as the last item before closing the generator.

    .. attention::

        The caller must only pass the new messages to the agent on each call
        to the :meth:`on_messages`, :meth:`on_messages_stream`, :meth:`BaseChatAgent.run`,
        or :meth:`BaseChatAgent.run_stream` methods.
        The agent maintains its state between calls to these methods.
        Do not pass the entire conversation history to the agent on each call.

    .. warning::
        The assistant agent is not thread-safe or coroutine-safe.
        It should not be shared between multiple tasks or coroutines, and it should
        not call its methods concurrently.

    The following diagram shows how the assistant agent works:

    .. image:: ../../images/assistant-agent.svg

    **Structured output:**

    If the `output_content_type` is set, the agent will respond with a :class:`~autogen_agentchat.messages.StructuredMessage`
    instead of a :class:`~autogen_agentchat.messages.TextMessage` in the final response by default.

    .. note::

        Currently, setting `output_content_type` prevents the agent from being
        able to call `load_component` and `dum_component` methods for serializable
        configuration. This will be fixed soon in the future.

    **Tool call behavior:**

    * If the model returns no tool call, then the response is immediately returned as a :class:`~autogen_agentchat.messages.TextMessage` or a :class:`~autogen_agentchat.messages.StructuredMessage` (when using structured output) in :attr:`~autogen_agentchat.base.Response.chat_message`. This ends the tool call iteration loop regardless of the `max_tool_iterations` setting.
    * When the model returns tool calls, they will be executed right away:
        - When `reflect_on_tool_use` is False, the tool call results are returned as a :class:`~autogen_agentchat.messages.ToolCallSummaryMessage` in :attr:`~autogen_agentchat.base.Response.chat_message`. You can customise the summary with either a static format string (`tool_call_summary_format`) **or** a callable (`tool_call_summary_formatter`); the callable is evaluated once per tool call.
        - When `reflect_on_tool_use` is True, the another model inference is made using the tool calls and results, and final response is returned as a :class:`~autogen_agentchat.messages.TextMessage` or a :class:`~autogen_agentchat.messages.StructuredMessage` (when using structured output) in :attr:`~autogen_agentchat.base.Response.chat_message`.
        - `reflect_on_tool_use` is set to `True` by default when `output_content_type` is set.
        - `reflect_on_tool_use` is set to `False` by default when `output_content_type` is not set.
    * If the model returns multiple tool calls, they will be executed concurrently. To disable parallel tool calls you need to configure the model client. For example, set `parallel_tool_calls=False` for :class:`~autogen_ext.models.openai.OpenAIChatCompletionClient` and :class:`~autogen_ext.models.openai.AzureOpenAIChatCompletionClient`.
    * The `max_tool_iterations` parameter controls how many sequential tool call iterations the agent can perform in a single run. When set to 1 (default), the agent executes tool calls once and returns the result. When set higher, the agent can make additional model calls to execute more tool calls if the model continues to request them, enabling multi-step tool-based workflows. The agent stops when either the model returns a text response (instead of tool calls) or the maximum number of iterations is reached.

    .. tip::

        By default, the tool call results are returned as the response when tool
        calls are made, so pay close attention to how the tools' return values
        are formatted—especially if another agent expects a specific schema.

        * Use **`tool_call_summary_format`** for a simple static template.
        * Use **`tool_call_summary_formatter`** for full programmatic control
          (e.g., "hide large success payloads, show full details on error").

        *Note*: `tool_call_summary_formatter` is **not serializable** and will
        be ignored when an agent is loaded from, or exported to, YAML/JSON
        configuration files.


    **Hand off behavior:**

    * If a handoff is triggered, a :class:`~autogen_agentchat.messages.HandoffMessage` will be returned in :attr:`~autogen_agentchat.base.Response.chat_message`.
    * If there are tool calls, they will also be executed right away before returning the handoff.
    * The tool calls and results are passed to the target agent through :attr:`~autogen_agentchat.messages.HandoffMessage.context`.


    .. note::
        If multiple handoffs are detected, only the first handoff is executed.
        To avoid this, disable parallel tool calls in the model client configuration.


    **Limit context size sent to the model:**

    You can limit the number of messages sent to the model by setting
    the `model_context` parameter to a :class:`~autogen_core.model_context.BufferedChatCompletionContext`.
    This will limit the number of recent messages sent to the model and can be useful
    when the model has a limit on the number of tokens it can process.
    Another option is to use a :class:`~autogen_core.model_context.TokenLimitedChatCompletionContext`
    which will limit the number of tokens sent to the model.
    You can also create your own model context by subclassing
    :class:`~autogen_core.model_context.ChatCompletionContext`.

    **Streaming mode:**

    The assistant agent can be used in streaming mode by setting `model_client_stream=True`.
    In this mode, the :meth:`on_messages_stream` and :meth:`BaseChatAgent.run_stream` methods will also yield
    :class:`~autogen_agentchat.messages.ModelClientStreamingChunkEvent`
    messages as the model client produces chunks of response.
    The chunk messages will not be included in the final response's inner messages.

    Args:
        name (str): The name of the agent.
        model_client (ChatCompletionClient): The model client to use for inference.
        tools (List[BaseTool[Any, Any]  | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None, optional): The tools to register with the agent.
        workbench (Workbench | Sequence[Workbench] | None, optional): The workbench or list of workbenches to use for the agent.
            Tools cannot be used when workbench is set and vice versa.
        handoffs (List[HandoffBase | str] | None, optional): The handoff configurations for the agent,
            allowing it to transfer to other agents by responding with a :class:`HandoffMessage`.
            The transfer is only executed when the team is in :class:`~autogen_agentchat.teams.Swarm`.
            If a handoff is a string, it should represent the target agent's name.
        model_context (ChatCompletionContext | None, optional): The model context for storing and retrieving :class:`~autogen_core.models.LLMMessage`. It can be preloaded with initial messages. The initial messages will be cleared when the agent is reset.
        description (str, optional): The description of the agent.
        system_message (str, optional): The system message for the model. If provided, it will be prepended to the messages in the model context when making an inference. Set to `None` to disable.
        model_client_stream (bool, optional): If `True`, the model client will be used in streaming mode.
            :meth:`on_messages_stream` and :meth:`BaseChatAgent.run_stream` methods will also yield :class:`~autogen_agentchat.messages.ModelClientStreamingChunkEvent`
            messages as the model client produces chunks of response. Defaults to `False`.
        reflect_on_tool_use (bool, optional): If `True`, the agent will make another model inference using the tool call and result
            to generate a response. If `False`, the tool call result will be returned as the response. By default, if `output_content_type` is set, this will be `True`;
            if `output_content_type` is not set, this will be `False`.
        output_content_type (type[BaseModel] | None, optional): The output content type for :class:`~autogen_agentchat.messages.StructuredMessage` response as a Pydantic model.
            This will be used with the model client to generate structured output.
            If this is set, the agent will respond with a :class:`~autogen_agentchat.messages.StructuredMessage` instead of a :class:`~autogen_agentchat.messages.TextMessage`
            in the final response, unless `reflect_on_tool_use` is `False` and a tool call is made.
        output_content_type_format (str | None, optional): (Experimental) The format string used for the content of a :class:`~autogen_agentchat.messages.StructuredMessage` response.
        max_tool_iterations (int, optional): The maximum number of tool iterations to perform until the model stops making tool calls. Defaults to `1`, which means the agent will
            only execute the tool calls made by the model once, and return the result as a :class:`~autogen_agentchat.messages.ToolCallSummaryMessage`,
            or a :class:`~autogen_agentchat.messages.TextMessage` or a :class:`~autogen_agentchat.messages.StructuredMessage` (when using structured output)
            in :attr:`~autogen_agentchat.base.Response.chat_message` as the final response.
            As soon as the model stops making tool calls, the agent will stop executing tool calls and return the result as the final response.
            The value must be greater than or equal to 1.
        tool_call_summary_format (str, optional): Static format string applied to each tool call result when composing the :class:`~autogen_agentchat.messages.ToolCallSummaryMessage`.
            Defaults to ``"{result}"``. Ignored if `tool_call_summary_formatter` is provided. When `reflect_on_tool_use` is ``False``, the summaries for all tool
            calls are concatenated with a newline ('\\n') and returned as the response.  Placeholders available in the template:
            `{tool_name}`, `{arguments}`, `{result}`, `{is_error}`.
        tool_call_summary_formatter (Callable[[FunctionCall, FunctionExecutionResult], str] | None, optional):
            Callable that receives the ``FunctionCall`` and its ``FunctionExecutionResult`` and returns the summary string.
            Overrides `tool_call_summary_format` when supplied and allows conditional logic — for example, emitting static string like
            ``"Tool FooBar executed successfully."`` on success and a full payload (including all passed arguments etc.) only on failure.

            **Limitation**: The callable is *not serializable*; values provided via YAML/JSON configs are ignored.

    .. note::

        `tool_call_summary_formatter` is intended for in-code use only. It cannot currently be saved or restored via
        configuration files.

        memory (Sequence[Memory] | None, optional): The memory store to use for the agent. Defaults to `None`.
        metadata (Dict[str, str] | None, optional): Optional metadata for tracking.

    Raises:
        ValueError: If tool names are not unique.
        ValueError: If handoff names are not unique.
        ValueError: If handoff names are not unique from tool names.
        ValueError: If maximum number of tool iterations is less than 1.

    Examples:

        **Example 1: basic agent**

        The following example demonstrates how to create an assistant agent with
        a model client and generate a response to a simple task.

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    # api_key = "your_openai_api_key"
                )
                agent = AssistantAgent(name="assistant", model_client=model_client)

                result = await agent.run(task="Name two cities in North America.")
                print(result)


            asyncio.run(main())

        **Example 2: model client token streaming**

        This example demonstrates how to create an assistant agent with
        a model client and generate a token stream by setting `model_client_stream=True`.

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    # api_key = "your_openai_api_key"
                )
                agent = AssistantAgent(
                    name="assistant",
                    model_client=model_client,
                    model_client_stream=True,
                )

                stream = agent.run_stream(task="Name two cities in North America.")
                async for message in stream:
                    print(message)


            asyncio.run(main())

        .. code-block:: text

            source='user' models_usage=None metadata={} content='Name two cities in North America.' type='TextMessage'
            source='assistant' models_usage=None metadata={} content='Two' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' cities' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' in' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' North' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' America' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' are' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' New' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' York' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' City' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' and' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' Toronto' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content='.' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content=' TERMIN' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=None metadata={} content='ATE' type='ModelClientStreamingChunkEvent'
            source='assistant' models_usage=RequestUsage(prompt_tokens=0, completion_tokens=0) metadata={} content='Two cities in North America are New York City and Toronto. TERMINATE' type='TextMessage'
            messages=[TextMessage(source='user', models_usage=None, metadata={}, content='Name two cities in North America.', type='TextMessage'), TextMessage(source='assistant', models_usage=RequestUsage(prompt_tokens=0, completion_tokens=0), metadata={}, content='Two cities in North America are New York City and Toronto. TERMINATE', type='TextMessage')] stop_reason=None


        **Example 3: agent with tools**

        The following example demonstrates how to create an assistant agent with
        a model client and a tool, generate a stream of messages for a task, and
        print the messages to the console using :class:`~autogen_agentchat.ui.Console`.

        The tool is a simple function that returns the current time.
        Under the hood, the function is wrapped in a :class:`~autogen_core.tools.FunctionTool`
        and used with the agent's model client. The doc string of the function
        is used as the tool description, the function name is used as the tool name,
        and the function signature including the type hints is used as the tool arguments.

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console


            async def get_current_time() -> str:
                return "The current time is 12:00 PM."


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    # api_key = "your_openai_api_key"
                )
                agent = AssistantAgent(name="assistant", model_client=model_client, tools=[get_current_time])
                await Console(agent.run_stream(task="What is the current time?"))


            asyncio.run(main())

        **Example 4: agent with max_tool_iterations**

        The following example demonstrates how to use the `max_tool_iterations` parameter
        to control how many times the agent can execute tool calls in a single run.
        This is useful when you want the agent to perform multiple sequential tool
        operations to reach a goal.

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console


            # Global counter state
            counter = 0


            def increment_counter() -> str:
                \"\"\"Increment the counter by 1 and return the current value.\"\"\"
                global counter
                counter += 1
                return f"Counter incremented to: {counter}"


            def get_counter() -> str:
                \"\"\"Get the current counter value.\"\"\"
                global counter
                return f"Current counter value: {counter}"


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    # api_key = "your_openai_api_key"
                )

                # Create agent with max_tool_iterations=5 to allow multiple tool calls
                agent = AssistantAgent(
                    name="assistant",
                    model_client=model_client,
                    tools=[increment_counter, get_counter],
                    max_tool_iterations=5,  # Allow up to 5 tool call iterations
                    reflect_on_tool_use=True,  # Get a final summary after tool calls
                )

                await Console(agent.run_stream(task="Increment the counter 3 times and then tell me the final value."))


            asyncio.run(main())

        **Example 5: agent with Model-Context Protocol (MCP) workbench**

        The following example demonstrates how to create an assistant agent with
        a model client and an :class:`~autogen_ext.tools.mcp.McpWorkbench` for
        interacting with a Model-Context Protocol (MCP) server.

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench


            async def main() -> None:
                params = StdioServerParams(
                    command="uvx",
                    args=["mcp-server-fetch"],
                    read_timeout_seconds=60,
                )

                # You can also use `start()` and `stop()` to manage the session.
                async with McpWorkbench(server_params=params) as workbench:
                    model_client = OpenAIChatCompletionClient(model="gpt-4.1-nano")
                    assistant = AssistantAgent(
                        name="Assistant",
                        model_client=model_client,
                        workbench=workbench,
                        reflect_on_tool_use=True,
                    )
                    await Console(
                        assistant.run_stream(task="Go to https://github.com/microsoft/autogen and tell me what you see.")
                    )


            asyncio.run(main())

        **Example 6: agent with structured output and tool**

        The following example demonstrates how to create an assistant agent with
        a model client configured to use structured output and a tool.
        Note that you need to use :class:`~autogen_core.tools.FunctionTool` to create the tool
        and the `strict=True` is required for structured output mode.
        Because the model is configured to use structured output, the output
        reflection response will be a JSON formatted string.

        .. code-block:: python

            import asyncio
            from typing import Literal

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.ui import Console
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

            # Create an OpenAIChatCompletionClient instance that supports structured output.
            model_client = OpenAIChatCompletionClient(
                model="gpt-4o-mini",
            )

            # Create an AssistantAgent instance that uses the tool and model client.
            agent = AssistantAgent(
                name="assistant",
                model_client=model_client,
                tools=[tool],
                system_message="Use the tool to analyze sentiment.",
                output_content_type=AgentResponse,
            )


            async def main() -> None:
                stream = agent.run_stream(task="I am happy today!")
                await Console(stream)


            asyncio.run(main())

        .. code-block:: text

            ---------- assistant ----------
            [FunctionCall(id='call_tIZjAVyKEDuijbBwLY6RHV2p', arguments='{"text":"I am happy today!"}', name='sentiment_analysis')]
            ---------- assistant ----------
            [FunctionExecutionResult(content='happy', call_id='call_tIZjAVyKEDuijbBwLY6RHV2p', is_error=False)]
            ---------- assistant ----------
            {"thoughts":"The user expresses a clear positive emotion by stating they are happy today, suggesting an upbeat mood.","response":"happy"}

        **Example 7: agent with bounded model context**

        The following example shows how to use a
        :class:`~autogen_core.model_context.BufferedChatCompletionContext`
        that only keeps the last 2 messages (1 user + 1 assistant).
        Bounded model context is useful when the model has a limit on the
        number of tokens it can process.

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.model_context import BufferedChatCompletionContext
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                # Create a model client.
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    # api_key = "your_openai_api_key"
                )

                # Create a model context that only keeps the last 2 messages (1 user + 1 assistant).
                model_context = BufferedChatCompletionContext(buffer_size=2)

                # Create an AssistantAgent instance with the model client and context.
                agent = AssistantAgent(
                    name="assistant",
                    model_client=model_client,
                    model_context=model_context,
                    system_message="You are a helpful assistant.",
                )

                result = await agent.run(task="Name two cities in North America.")
                print(result.messages[-1].content)  # type: ignore

                result = await agent.run(task="My favorite color is blue.")
                print(result.messages[-1].content)  # type: ignore

                result = await agent.run(task="Did I ask you any question?")
                print(result.messages[-1].content)  # type: ignore


            asyncio.run(main())

        .. code-block:: text

            Two cities in North America are New York City and Toronto.
            That's great! Blue is often associated with calmness and serenity. Do you have a specific shade of blue that you like, or any particular reason why it's your favorite?
            No, you didn't ask a question. I apologize for any misunderstanding. If you have something specific you'd like to discuss or ask, feel free to let me know!

        **Example 8: agent with memory**

        The following example shows how to use a list-based memory with the assistant agent.
        The memory is preloaded with some initial content.
        Under the hood, the memory is used to update the model context
        before making an inference, using the :meth:`~autogen_core.memory.Memory.update_context` method.

        .. code-block:: python

            import asyncio

            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.memory import ListMemory, MemoryContent
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                # Create a model client.
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    # api_key = "your_openai_api_key"
                )

                # Create a list-based memory with some initial content.
                memory = ListMemory()
                await memory.add(MemoryContent(content="User likes pizza.", mime_type="text/plain"))
                await memory.add(MemoryContent(content="User dislikes cheese.", mime_type="text/plain"))

                # Create an AssistantAgent instance with the model client and memory.
                agent = AssistantAgent(
                    name="assistant",
                    model_client=model_client,
                    memory=[memory],
                    system_message="You are a helpful assistant.",
                )

                result = await agent.run(task="What is a good dinner idea?")
                print(result.messages[-1].content)  # type: ignore


            asyncio.run(main())

        .. code-block:: text

            How about making a delicious pizza without cheese? You can create a flavorful veggie pizza with a variety of toppings. Here's a quick idea:

            **Veggie Tomato Sauce Pizza**
            - Start with a pizza crust (store-bought or homemade).
            - Spread a layer of marinara or tomato sauce evenly over the crust.
            - Top with your favorite vegetables like bell peppers, mushrooms, onions, olives, and spinach.
            - Add some protein if you'd like, such as grilled chicken or pepperoni (ensure it's cheese-free).
            - Sprinkle with herbs like oregano and basil, and maybe a drizzle of olive oil.
            - Bake according to the crust instructions until the edges are golden and the veggies are cooked.

            Serve it with a side salad or some garlic bread to complete the meal! Enjoy your dinner!

        **Example 9: agent with `o1-mini`**

        The following example shows how to use `o1-mini` model with the assistant agent.

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(
                    model="o1-mini",
                    # api_key = "your_openai_api_key"
                )
                # The system message is not supported by the o1 series model.
                agent = AssistantAgent(name="assistant", model_client=model_client, system_message=None)

                result = await agent.run(task="What is the capital of France?")
                print(result.messages[-1].content)  # type: ignore


            asyncio.run(main())

        .. note::

            The `o1-preview` and `o1-mini` models do not support system message and function calling.
            So the `system_message` should be set to `None` and the `tools` and `handoffs` should not be set.
            See `o1 beta limitations <https://platform.openai.com/docs/guides/reasoning#beta-limitations>`_ for more details.


        **Example 10: agent using reasoning model with custom model context.**

        The following example shows how to use a reasoning model (DeepSeek R1) with the assistant agent.
        The model context is used to filter out the thought field from the assistant message.

        .. code-block:: python

            import asyncio
            from typing import List

            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.model_context import UnboundedChatCompletionContext
            from autogen_core.models import AssistantMessage, LLMMessage, ModelFamily
            from autogen_ext.models.ollama import OllamaChatCompletionClient


            class ReasoningModelContext(UnboundedChatCompletionContext):
                \"\"\"A model context for reasoning models.\"\"\"

                async def get_messages(self) -> List[LLMMessage]:
                    messages = await super().get_messages()
                    # Filter out thought field from AssistantMessage.
                    messages_out: List[LLMMessage] = []
                    for message in messages:
                        if isinstance(message, AssistantMessage):
                            message.thought = None
                        messages_out.append(message)
                    return messages_out


            # Create an instance of the model client for DeepSeek R1 hosted locally on Ollama.
            model_client = OllamaChatCompletionClient(
                model="deepseek-r1:8b",
                model_info={
                    "vision": False,
                    "function_calling": False,
                    "json_output": False,
                    "family": ModelFamily.R1,
                    "structured_output": True,
                },
            )

            agent = AssistantAgent(
                "reasoning_agent",
                model_client=model_client,
                model_context=ReasoningModelContext(),  # Use the custom model context.
            )


            async def run_reasoning_agent() -> None:
                result = await agent.run(task="What is the capital of France?")
                print(result)


            asyncio.run(run_reasoning_agent())

    For detailed examples and usage, see the Examples section below.
    """

    component_version = 2
    component_config_schema = AssistantAgentConfig
    component_provider_override = "autogen_agentchat.agents.AssistantAgent"

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        tools: List[BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
        workbench: Workbench | Sequence[Workbench] | None = None,
        handoffs: List[HandoffBase | str] | None = None,
        model_context: ChatCompletionContext | None = None,
        description: str = "An agent that provides assistance with ability to use tools.",
        system_message: (
            str | None
        ) = "You are a helpful AI assistant. Solve tasks using your tools. Reply with TERMINATE when the task has been completed.",
        model_client_stream: bool = False,
        reflect_on_tool_use: bool | None = None,
        max_tool_iterations: int = 1,
        tool_call_summary_format: str = "{result}",
        tool_call_summary_formatter: Callable[[FunctionCall, FunctionExecutionResult], str] | None = None,
        output_content_type: type[BaseModel] | None = None,
        output_content_type_format: str | None = None,
        memory: Sequence[Memory] | None = None,
        metadata: Dict[str, str] | None = None,
    ):
        super().__init__(name=name, description=description)
        self._metadata = metadata or {}
        self._model_client = model_client
        self._model_client_stream = model_client_stream
        self._output_content_type: type[BaseModel] | None = output_content_type
        self._output_content_type_format = output_content_type_format
        self._structured_message_factory: StructuredMessageFactory | None = None
        if output_content_type is not None:
            self._structured_message_factory = StructuredMessageFactory(
                input_model=output_content_type, format_string=output_content_type_format
            )

        self._memory = None
        if memory is not None:
            if isinstance(memory, list):
                self._memory = memory
            else:
                raise TypeError(f"Expected Memory, List[Memory], or None, got {type(memory)}")

        self._system_messages: List[SystemMessage] = []
        if system_message is None:
            self._system_messages = []
        else:
            self._system_messages = [SystemMessage(content=system_message)]
        self._tools: List[BaseTool[Any, Any]] = []
        if tools is not None:
            if model_client.model_info["function_calling"] is False:
                raise ValueError("The model does not support function calling.")
            for tool in tools:
                if isinstance(tool, BaseTool):
                    self._tools.append(tool)
                elif callable(tool):
                    if hasattr(tool, "__doc__") and tool.__doc__ is not None:
                        description = tool.__doc__
                    else:
                        description = ""
                    self._tools.append(FunctionTool(tool, description=description))
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")
        # Check if tool names are unique.
        tool_names = [tool.name for tool in self._tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Tool names must be unique: {tool_names}")

        # Handoff tools.
        self._handoff_tools: List[BaseTool[Any, Any]] = []
        self._handoffs: Dict[str, HandoffBase] = {}
        if handoffs is not None:
            if model_client.model_info["function_calling"] is False:
                raise ValueError("The model does not support function calling, which is needed for handoffs.")
            for handoff in handoffs:
                if isinstance(handoff, str):
                    handoff = HandoffBase(target=handoff)
                if isinstance(handoff, HandoffBase):
                    self._handoff_tools.append(handoff.handoff_tool)
                    self._handoffs[handoff.name] = handoff
                else:
                    raise ValueError(f"Unsupported handoff type: {type(handoff)}")
        # Check if handoff tool names are unique.
        handoff_tool_names = [tool.name for tool in self._handoff_tools]
        if len(handoff_tool_names) != len(set(handoff_tool_names)):
            raise ValueError(f"Handoff names must be unique: {handoff_tool_names}")
        # Create sets for faster lookup
        tool_names_set = set(tool_names)
        handoff_tool_names_set = set(handoff_tool_names)

        # Check if there's any overlap between handoff tool names and tool names
        overlap = tool_names_set.intersection(handoff_tool_names_set)

        # Also check if any handoff target name matches a tool name
        # This handles the case where a handoff is specified directly with a string that matches a tool name
        for handoff in handoffs or []:
            if isinstance(handoff, str) and handoff in tool_names_set:
                raise ValueError("Handoff names must be unique from tool names")
            elif isinstance(handoff, HandoffBase) and handoff.target in tool_names_set:
                raise ValueError("Handoff names must be unique from tool names")

        if overlap:
            raise ValueError("Handoff names must be unique from tool names")

        if workbench is not None:
            if self._tools:
                raise ValueError("Tools cannot be used with a workbench.")
            if isinstance(workbench, Sequence):
                self._workbench = workbench
            else:
                self._workbench = [workbench]
        else:
            self._workbench = [StaticStreamWorkbench(self._tools)]

        if model_context is not None:
            self._model_context = model_context
        else:
            self._model_context = UnboundedChatCompletionContext()

        if self._output_content_type is not None and reflect_on_tool_use is None:
            # If output_content_type is set, we need to reflect on tool use by default.
            self._reflect_on_tool_use = True
        elif reflect_on_tool_use is None:
            self._reflect_on_tool_use = False
        else:
            self._reflect_on_tool_use = reflect_on_tool_use

        # Tool call loop
        self._max_tool_iterations = max_tool_iterations
        if self._max_tool_iterations < 1:
            raise ValueError(
                f"Maximum number of tool iterations must be greater than or equal to 1, got {max_tool_iterations}"
            )

        self._tool_call_summary_format = tool_call_summary_format
        self._tool_call_summary_formatter = tool_call_summary_formatter
        self._is_running = False

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Get the types of messages this agent can produce.

        Returns:
            Sequence of message types this agent can generate
        """
        types: List[type[BaseChatMessage]] = [TextMessage, ToolCallSummaryMessage, HandoffMessage]
        if self._structured_message_factory is not None:
            types.append(StructuredMessage)
        return types

    @property
    def model_context(self) -> ChatCompletionContext:
        """Get the model context used by this agent.

        Returns:
            The chat completion context for this agent
        """
        return self._model_context

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        """Process incoming messages and generate a response.

        Args:
            messages: Sequence of messages to process
            cancellation_token: Token for cancelling operation

        Returns:
            Response containing the agent's reply
        """
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        """Process messages and stream the response.

        Args:
            messages: Sequence of messages to process
            cancellation_token: Token for cancelling operation

        Yields:
            Events, messages and final response during processing
        """

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        memory = self._memory
        system_messages = self._system_messages
        workbench = self._workbench
        handoff_tools = self._handoff_tools
        handoffs = self._handoffs
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        reflect_on_tool_use = self._reflect_on_tool_use
        max_tool_iterations = self._max_tool_iterations
        tool_call_summary_format = self._tool_call_summary_format
        tool_call_summary_formatter = self._tool_call_summary_formatter
        output_content_type = self._output_content_type

        # STEP 1: Add new user/handoff messages to the model context
        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )

        # STEP 2: Update model context with any relevant memory
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        for event_msg in await self._update_model_context_with_memory(
            memory=memory,
            model_context=model_context,
            agent_name=agent_name,
        ):
            inner_messages.append(event_msg)
            yield event_msg

        # STEP 3: Generate a message ID for correlation between streaming chunks and final message
        message_id = str(uuid.uuid4())

        # STEP 4: Run the first inference
        model_result = None
        async for inference_output in self._call_llm(
            model_client=model_client,
            model_client_stream=model_client_stream,
            system_messages=system_messages,
            model_context=model_context,
            workbench=workbench,
            handoff_tools=handoff_tools,
            agent_name=agent_name,
            cancellation_token=cancellation_token,
            output_content_type=output_content_type,
            message_id=message_id,
        ):
            if isinstance(inference_output, CreateResult):
                model_result = inference_output
            else:
                # Streaming chunk event
                yield inference_output

        assert model_result is not None, "No model result was produced."

        # --- NEW: If the model produced a hidden "thought," yield it as an event ---
        if model_result.thought:
            thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # Add the assistant message to the model context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )

        # STEP 5: Process the model output
        async for output_event in self._process_model_result(
            model_result=model_result,
            inner_messages=inner_messages,
            cancellation_token=cancellation_token,
            agent_name=agent_name,
            system_messages=system_messages,
            model_context=model_context,
            workbench=workbench,
            handoff_tools=handoff_tools,
            handoffs=handoffs,
            model_client=model_client,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            max_tool_iterations=max_tool_iterations,
            tool_call_summary_format=tool_call_summary_format,
            tool_call_summary_formatter=tool_call_summary_formatter,
            output_content_type=output_content_type,
            message_id=message_id,
            format_string=self._output_content_type_format,
        ):
            yield output_event

    @staticmethod
    async def _add_messages_to_context(
        model_context: ChatCompletionContext,
        messages: Sequence[BaseChatMessage],
    ) -> None:
        """
        Add incoming messages to the model context.
        """
        for msg in messages:
            if isinstance(msg, HandoffMessage):
                for llm_msg in msg.context:
                    await model_context.add_message(llm_msg)
            await model_context.add_message(msg.to_model_message())

    @staticmethod
    async def _update_model_context_with_memory(
        memory: Optional[Sequence[Memory]],
        model_context: ChatCompletionContext,
        agent_name: str,
    ) -> List[MemoryQueryEvent]:
        """Update model context with memory content.

        Args:
            memory: Optional sequence of memory stores to query
            model_context: Context to update with memory content
            agent_name: Name of the agent for event tracking

        Returns:
            List of memory query events generated during update
        """
        events: List[MemoryQueryEvent] = []
        if memory:
            for mem in memory:
                update_context_result = await mem.update_context(model_context)
                if update_context_result and len(update_context_result.memories.results) > 0:
                    memory_query_event_msg = MemoryQueryEvent(
                        content=update_context_result.memories.results,
                        source=agent_name,
                    )
                    events.append(memory_query_event_msg)
        return events

    @classmethod
    async def _call_llm(
        cls,
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        workbench: Sequence[Workbench],
        handoff_tools: List[BaseTool[Any, Any]],
        agent_name: str,
        cancellation_token: CancellationToken,
        output_content_type: type[BaseModel] | None,
        message_id: str,
    ) -> AsyncGenerator[Union[CreateResult, ModelClientStreamingChunkEvent], None]:
        """Call the language model with given context and configuration.

        Args:
            model_client: Client for model inference
            model_client_stream: Whether to stream responses
            system_messages: System messages to include
            model_context: Context containing message history
            workbench: Available workbenches
            handoff_tools: Tools for handling handoffs
            agent_name: Name of the agent
            cancellation_token: Token for cancelling operation
            output_content_type: Optional type for structured output

        Returns:
            Generator yielding model results or streaming chunks
        """
        all_messages = await model_context.get_messages()
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=system_messages + all_messages)

        tools = [tool for wb in workbench for tool in await wb.list_tools()] + handoff_tools

        if model_client_stream:
            model_result: Optional[CreateResult] = None

            async for chunk in model_client.create_stream(
                llm_messages,
                tools=tools,
                json_output=output_content_type,
                cancellation_token=cancellation_token,
            ):
                if isinstance(chunk, CreateResult):
                    model_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source=agent_name, full_message_id=message_id)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
            if model_result is None:
                raise RuntimeError("No final model result in streaming mode.")
            yield model_result
        else:
            model_result = await model_client.create(
                llm_messages,
                tools=tools,
                cancellation_token=cancellation_token,
                json_output=output_content_type,
            )
            yield model_result

    @classmethod
    async def _process_model_result(
        cls,
        model_result: CreateResult,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        cancellation_token: CancellationToken,
        agent_name: str,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        workbench: Sequence[Workbench],
        handoff_tools: List[BaseTool[Any, Any]],
        handoffs: Dict[str, HandoffBase],
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        reflect_on_tool_use: bool,
        tool_call_summary_format: str,
        tool_call_summary_formatter: Callable[[FunctionCall, FunctionExecutionResult], str] | None,
        max_tool_iterations: int,
        output_content_type: type[BaseModel] | None,
        message_id: str,
        format_string: str | None = None,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Handle final or partial responses from model_result, including tool calls, handoffs,
        and reflection if needed. Supports tool call loops when enabled.
        """

        # Tool call loop implementation with streaming support
        current_model_result = model_result
        # This variable is needed for the final summary/reflection step
        executed_calls_and_results: List[Tuple[FunctionCall, FunctionExecutionResult]] = []

        for loop_iteration in range(max_tool_iterations):
            # If direct text response (string), we're done
            if isinstance(current_model_result.content, str):
                # Use the passed message ID for the final message
                if output_content_type:
                    content = output_content_type.model_validate_json(current_model_result.content)
                    yield Response(
                        chat_message=StructuredMessage[output_content_type](  # type: ignore[valid-type]
                            content=content,
                            source=agent_name,
                            models_usage=current_model_result.usage,
                            format_string=format_string,
                            id=message_id,
                        ),
                        inner_messages=inner_messages,
                    )
                else:
                    yield Response(
                        chat_message=TextMessage(
                            content=current_model_result.content,
                            source=agent_name,
                            models_usage=current_model_result.usage,
                            id=message_id,
                        ),
                        inner_messages=inner_messages,
                    )
                return

            # Otherwise, we have function calls
            assert isinstance(current_model_result.content, list) and all(
                isinstance(item, FunctionCall) for item in current_model_result.content
            )

            # STEP 4A: Yield ToolCallRequestEvent
            tool_call_msg = ToolCallRequestEvent(
                content=current_model_result.content,
                source=agent_name,
                models_usage=current_model_result.usage,
            )
            event_logger.debug(tool_call_msg)
            inner_messages.append(tool_call_msg)
            yield tool_call_msg

            # STEP 4B: Execute tool calls with streaming support
            # Use a queue to handle streaming results from tool calls.
            stream = asyncio.Queue[BaseAgentEvent | BaseChatMessage | None]()

            async def _execute_tool_calls(
                function_calls: List[FunctionCall],
                stream_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | None],
            ) -> List[Tuple[FunctionCall, FunctionExecutionResult]]:
                results = await asyncio.gather(
                    *[
                        cls._execute_tool_call(
                            tool_call=call,
                            workbench=workbench,
                            handoff_tools=handoff_tools,
                            agent_name=agent_name,
                            cancellation_token=cancellation_token,
                            stream=stream_queue,
                        )
                        for call in function_calls
                    ]
                )
                # Signal the end of streaming by putting None in the queue.
                stream_queue.put_nowait(None)
                return results

            task = asyncio.create_task(_execute_tool_calls(current_model_result.content, stream))

            while True:
                event = await stream.get()
                if event is None:
                    # End of streaming, break the loop.
                    break
                if isinstance(event, BaseAgentEvent) or isinstance(event, BaseChatMessage):
                    yield event
                    inner_messages.append(event)
                else:
                    raise RuntimeError(f"Unexpected event type: {type(event)}")

            # Wait for all tool calls to complete.
            executed_calls_and_results = await task
            exec_results = [result for _, result in executed_calls_and_results]

            # Yield ToolCallExecutionEvent
            tool_call_result_msg = ToolCallExecutionEvent(
                content=exec_results,
                source=agent_name,
            )
            event_logger.debug(tool_call_result_msg)
            await model_context.add_message(FunctionExecutionResultMessage(content=exec_results))
            inner_messages.append(tool_call_result_msg)
            yield tool_call_result_msg

            # STEP 4C: Check for handoff
            handoff_output = cls._check_and_handle_handoff(
                model_result=current_model_result,
                executed_calls_and_results=executed_calls_and_results,
                inner_messages=inner_messages,
                handoffs=handoffs,
                agent_name=agent_name,
            )
            if handoff_output:
                yield handoff_output
                return

            # STEP 4D: Check if we should continue the loop.
            # If we are on the last iteration, break to the summary/reflection step.
            if loop_iteration == max_tool_iterations - 1:
                break

            # Continue the loop: make another model call using _call_llm
            next_model_result: Optional[CreateResult] = None
            async for llm_output in cls._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                workbench=workbench,
                handoff_tools=handoff_tools,
                agent_name=agent_name,
                cancellation_token=cancellation_token,
                output_content_type=output_content_type,
                message_id=message_id,  # Use same message ID for consistency
            ):
                if isinstance(llm_output, CreateResult):
                    next_model_result = llm_output
                else:
                    # Streaming chunk event
                    yield llm_output

            assert next_model_result is not None, "No model result was produced in tool call loop."
            current_model_result = next_model_result

            # Yield thought event if present
            if current_model_result.thought:
                thought_event = ThoughtEvent(content=current_model_result.thought, source=agent_name)
                yield thought_event
                inner_messages.append(thought_event)

            # Add the assistant message to the model context (including thought if present)
            await model_context.add_message(
                AssistantMessage(
                    content=current_model_result.content,
                    source=agent_name,
                    thought=getattr(current_model_result, "thought", None),
                )
            )

        # After the loop, reflect or summarize tool results
        if reflect_on_tool_use:
            async for reflection_response in cls._reflect_on_tool_use_flow(
                system_messages=system_messages,
                model_client=model_client,
                model_client_stream=model_client_stream,
                model_context=model_context,
                workbench=workbench,
                handoff_tools=handoff_tools,
                agent_name=agent_name,
                inner_messages=inner_messages,
                output_content_type=output_content_type,
                cancellation_token=cancellation_token,
            ):
                yield reflection_response
        else:
            yield cls._summarize_tool_use(
                executed_calls_and_results=executed_calls_and_results,
                inner_messages=inner_messages,
                handoffs=handoffs,
                tool_call_summary_format=tool_call_summary_format,
                tool_call_summary_formatter=tool_call_summary_formatter,
                agent_name=agent_name,
            )
        return

    @staticmethod
    def _check_and_handle_handoff(
        model_result: CreateResult,
        executed_calls_and_results: List[Tuple[FunctionCall, FunctionExecutionResult]],
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        handoffs: Dict[str, HandoffBase],
        agent_name: str,
    ) -> Optional[Response]:
        """Check for and handle any handoff requests in the model result.

        Args:
            model_result: Result from model inference
            executed_calls_and_results: List of executed tool calls and their results
            inner_messages: List of messages generated during processing
            handoffs: Dictionary of available handoff configurations
            agent_name: Name of the agent

        Returns:
            Optional response containing handoff message if handoff detected
        """
        handoff_reqs = [
            call for call in model_result.content if isinstance(call, FunctionCall) and call.name in handoffs
        ]
        if len(handoff_reqs) > 0:
            # We have at least one handoff function call
            selected_handoff = handoffs[handoff_reqs[0].name]

            if len(handoff_reqs) > 1:
                warnings.warn(
                    (
                        f"Multiple handoffs detected. Only the first is executed: "
                        f"{[handoffs[c.name].name for c in handoff_reqs]}. "
                        "Disable parallel tool calls in the model client to avoid this warning."
                    ),
                    stacklevel=2,
                )

            # Collect normal tool calls (not handoff) into the handoff context
            tool_calls: List[FunctionCall] = []
            tool_call_results: List[FunctionExecutionResult] = []
            # Collect the results returned by handoff_tool. By default, the message attribute will returned.
            selected_handoff_message = selected_handoff.message
            for exec_call, exec_result in executed_calls_and_results:
                if exec_call.name not in handoffs:
                    tool_calls.append(exec_call)
                    tool_call_results.append(exec_result)
                elif exec_call.name == selected_handoff.name:
                    selected_handoff_message = exec_result.content

            handoff_context: List[LLMMessage] = []
            if len(tool_calls) > 0:
                # Include the thought in the AssistantMessage if model_result has it
                handoff_context.append(
                    AssistantMessage(
                        content=tool_calls,
                        source=agent_name,
                        thought=getattr(model_result, "thought", None),
                    )
                )
                handoff_context.append(FunctionExecutionResultMessage(content=tool_call_results))
            elif model_result.thought:
                # If no tool calls, but a thought exists, include it in the context
                handoff_context.append(
                    AssistantMessage(
                        content=model_result.thought,
                        source=agent_name,
                    )
                )

            # Return response for the first handoff
            return Response(
                chat_message=HandoffMessage(
                    content=selected_handoff_message,
                    target=selected_handoff.target,
                    source=agent_name,
                    context=handoff_context,
                ),
                inner_messages=inner_messages,
            )
        return None

    @classmethod
    async def _reflect_on_tool_use_flow(
        cls,
        system_messages: List[SystemMessage],
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        model_context: ChatCompletionContext,
        workbench: Sequence[Workbench],
        handoff_tools: List[BaseTool[Any, Any]],
        agent_name: str,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        output_content_type: type[BaseModel] | None,
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[Response | ModelClientStreamingChunkEvent | ThoughtEvent, None]:
        """
        If reflect_on_tool_use=True, we do another inference based on tool results
        and yield the final text response (or streaming chunks).
        """
        all_messages = system_messages + await model_context.get_messages()
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=all_messages)

        reflection_result: Optional[CreateResult] = None

        # Generate a message ID for correlation between chunks and final message in reflection flow
        reflection_message_id = str(uuid.uuid4())

        if model_client_stream:
            async for chunk in model_client.create_stream(
                llm_messages,
                json_output=output_content_type,
                cancellation_token=cancellation_token,
                tool_choice="none",  # Do not use tools in reflection flow.
            ):
                if isinstance(chunk, CreateResult):
                    reflection_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(
                        content=chunk, source=agent_name, full_message_id=reflection_message_id
                    )
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
        else:
            reflection_result = await model_client.create(
                llm_messages,
                json_output=output_content_type,
                cancellation_token=cancellation_token,
                tool_choice="none",  # Do not use tools in reflection flow.
            )

        if not reflection_result or not isinstance(reflection_result.content, str):
            raise RuntimeError("Reflect on tool use produced no valid text response.")

        # --- NEW: If the reflection produced a thought, yield it ---
        if reflection_result.thought:
            thought_event = ThoughtEvent(content=reflection_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # Add to context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=reflection_result.content,
                source=agent_name,
                thought=getattr(reflection_result, "thought", None),
            )
        )

        if output_content_type:
            content = output_content_type.model_validate_json(reflection_result.content)
            yield Response(
                chat_message=StructuredMessage[output_content_type](  # type: ignore[valid-type]
                    content=content,
                    source=agent_name,
                    models_usage=reflection_result.usage,
                    id=reflection_message_id,
                ),
                inner_messages=inner_messages,
            )
        else:
            yield Response(
                chat_message=TextMessage(
                    content=reflection_result.content,
                    source=agent_name,
                    models_usage=reflection_result.usage,
                    id=reflection_message_id,
                ),
                inner_messages=inner_messages,
            )

    @staticmethod
    def _summarize_tool_use(
        executed_calls_and_results: List[Tuple[FunctionCall, FunctionExecutionResult]],
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        handoffs: Dict[str, HandoffBase],
        tool_call_summary_format: str,
        tool_call_summary_formatter: Callable[[FunctionCall, FunctionExecutionResult], str] | None,
        agent_name: str,
    ) -> Response:
        """
        If reflect_on_tool_use=False, create a summary message of all tool calls.
        """
        # Filter out calls which were actually handoffs
        normal_tool_calls = [(call, result) for call, result in executed_calls_and_results if call.name not in handoffs]

        def default_tool_call_summary_formatter(call: FunctionCall, result: FunctionExecutionResult) -> str:
            return tool_call_summary_format.format(
                tool_name=call.name,
                arguments=call.arguments,
                result=result.content,
                is_error=result.is_error,
            )

        summary_formatter = tool_call_summary_formatter or default_tool_call_summary_formatter

        tool_call_summaries = [summary_formatter(call, result) for call, result in normal_tool_calls]

        tool_call_summary = "\n".join(tool_call_summaries)
        return Response(
            chat_message=ToolCallSummaryMessage(
                content=tool_call_summary,
                source=agent_name,
                tool_calls=[call for call, _ in normal_tool_calls],
                results=[result for _, result in normal_tool_calls],
            ),
            inner_messages=inner_messages,
        )

    @staticmethod
    async def _execute_tool_call(
        tool_call: FunctionCall,
        workbench: Sequence[Workbench],
        handoff_tools: List[BaseTool[Any, Any]],
        agent_name: str,
        cancellation_token: CancellationToken,
        stream: asyncio.Queue[BaseAgentEvent | BaseChatMessage | None],
    ) -> Tuple[FunctionCall, FunctionExecutionResult]:
        """Execute a single tool call and return the result."""
        # Load the arguments from the tool call.
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            return (
                tool_call,
                FunctionExecutionResult(
                    content=f"Error: {e}",
                    call_id=tool_call.id,
                    is_error=True,
                    name=tool_call.name,
                ),
            )

        # Check if the tool call is a handoff.
        # TODO: consider creating a combined workbench to handle both handoff and normal tools.
        for handoff_tool in handoff_tools:
            if tool_call.name == handoff_tool.name:
                # Run handoff tool call.
                result = await handoff_tool.run_json(arguments, cancellation_token, call_id=tool_call.id)
                result_as_str = handoff_tool.return_value_as_string(result)
                return (
                    tool_call,
                    FunctionExecutionResult(
                        content=result_as_str,
                        call_id=tool_call.id,
                        is_error=False,
                        name=tool_call.name,
                    ),
                )

        # Handle normal tool call using workbench.
        for wb in workbench:
            tools = await wb.list_tools()
            if any(t["name"] == tool_call.name for t in tools):
                if isinstance(wb, StaticStreamWorkbench):
                    tool_result: ToolResult | None = None
                    async for event in wb.call_tool_stream(
                        name=tool_call.name,
                        arguments=arguments,
                        cancellation_token=cancellation_token,
                        call_id=tool_call.id,
                    ):
                        if isinstance(event, ToolResult):
                            tool_result = event
                        elif isinstance(event, BaseAgentEvent) or isinstance(event, BaseChatMessage):
                            await stream.put(event)
                        else:
                            warnings.warn(
                                f"Unexpected event type: {type(event)} in tool call streaming.",
                                UserWarning,
                                stacklevel=2,
                            )
                    assert isinstance(tool_result, ToolResult), "Tool result should not be None in streaming mode."
                else:
                    tool_result = await wb.call_tool(
                        name=tool_call.name,
                        arguments=arguments,
                        cancellation_token=cancellation_token,
                        call_id=tool_call.id,
                    )
                return (
                    tool_call,
                    FunctionExecutionResult(
                        content=tool_result.to_text(),
                        call_id=tool_call.id,
                        is_error=tool_result.is_error,
                        name=tool_call.name,
                    ),
                )

        return (
            tool_call,
            FunctionExecutionResult(
                content=f"Error: tool '{tool_call.name}' not found in any workbench",
                call_id=tool_call.id,
                is_error=True,
                name=tool_call.name,
            ),
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant agent to its initialization state."""
        await self._model_context.clear()

    async def save_state(self) -> Mapping[str, Any]:
        """Save the current state of the assistant agent."""
        model_context_state = await self._model_context.save_state()
        return AssistantAgentState(llm_context=model_context_state).model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the state of the assistant agent"""
        assistant_agent_state = AssistantAgentState.model_validate(state)
        # Load the model context state.
        await self._model_context.load_state(assistant_agent_state.llm_context)

    @staticmethod
    def _get_compatible_context(model_client: ChatCompletionClient, messages: List[LLMMessage]) -> Sequence[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)

    def _to_config(self) -> AssistantAgentConfig:
        """Convert the assistant agent to a declarative config."""

        return AssistantAgentConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            tools=None,  # versionchanged:: v0.5.5  Now tools are not serialized, Cause they are part of the workbench.
            workbench=[wb.dump_component() for wb in self._workbench] if self._workbench else None,
            handoffs=list(self._handoffs.values()) if self._handoffs else None,
            model_context=self._model_context.dump_component(),
            memory=[memory.dump_component() for memory in self._memory] if self._memory else None,
            description=self.description,
            system_message=self._system_messages[0].content
            if self._system_messages and isinstance(self._system_messages[0].content, str)
            else None,
            model_client_stream=self._model_client_stream,
            reflect_on_tool_use=self._reflect_on_tool_use,
            max_tool_iterations=self._max_tool_iterations,
            tool_call_summary_format=self._tool_call_summary_format,
            structured_message_factory=self._structured_message_factory.dump_component()
            if self._structured_message_factory
            else None,
            metadata=self._metadata,
        )

    @classmethod
    def _from_config(cls, config: AssistantAgentConfig) -> Self:
        """Create an assistant agent from a declarative config."""
        if config.structured_message_factory:
            structured_message_factory = StructuredMessageFactory.load_component(config.structured_message_factory)
            format_string = structured_message_factory.format_string
            output_content_type = structured_message_factory.ContentModel

        else:
            format_string = None
            output_content_type = None

        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            workbench=[Workbench.load_component(wb) for wb in config.workbench] if config.workbench else None,
            handoffs=config.handoffs,
            model_context=ChatCompletionContext.load_component(config.model_context) if config.model_context else None,
            tools=[BaseTool.load_component(tool) for tool in config.tools] if config.tools else None,
            memory=[Memory.load_component(memory) for memory in config.memory] if config.memory else None,
            description=config.description,
            system_message=config.system_message,
            model_client_stream=config.model_client_stream,
            reflect_on_tool_use=config.reflect_on_tool_use,
            max_tool_iterations=config.max_tool_iterations,
            tool_call_summary_format=config.tool_call_summary_format,
            output_content_type=output_content_type,
            output_content_type_format=format_string,
            metadata=config.metadata,
        )

# From agents/_assistant_agent.py
def model_context(self) -> ChatCompletionContext:
        """Get the model context used by this agent.

        Returns:
            The chat completion context for this agent
        """
        return self._model_context

# From agents/_assistant_agent.py
def default_tool_call_summary_formatter(call: FunctionCall, result: FunctionExecutionResult) -> str:
            return tool_call_summary_format.format(
                tool_name=call.name,
                arguments=call.arguments,
                result=result.content,
                is_error=result.is_error,
            )

from autogen_core import trace_create_agent_span
from autogen_core import trace_invoke_agent_span
from base import ChatAgent
from base import TaskResult
from state import BaseState

# From agents/_base_chat_agent.py
class BaseChatAgent(ChatAgent, ABC, ComponentBase[BaseModel]):
    """Base class for a chat agent.

    This abstract class provides a base implementation for a :class:`ChatAgent`.
    To create a new chat agent, subclass this class and implement the
    :meth:`on_messages`, :meth:`on_reset`, and :attr:`produced_message_types`.
    If streaming is required, also implement the :meth:`on_messages_stream` method.

    An agent is considered stateful and maintains its state between calls to
    the :meth:`on_messages` or :meth:`on_messages_stream` methods.
    The agent should store its state in the
    agent instance. The agent should also implement the :meth:`on_reset` method
    to reset the agent to its initialization state.

    .. note::

        The caller should only pass the new messages to the agent on each call
        to the :meth:`on_messages` or :meth:`on_messages_stream` method.
        Do not pass the entire conversation history to the agent on each call.
        This design principle must be followed when creating a new agent.
    """

    component_type = "agent"

    def __init__(self, name: str, description: str) -> None:
        """Initialize the agent with a name and description."""
        with trace_create_agent_span(
            agent_name=name,
            agent_description=description,
        ):
            self._name = name
            if self._name.isidentifier() is False:
                raise ValueError("The agent name must be a valid Python identifier.")
            self._description = description

    @property
    def name(self) -> str:
        """The name of the agent. This is used by team to uniquely identify
        the agent. It should be unique within the team."""
        return self._name

    @property
    def description(self) -> str:
        """The description of the agent. This is used by team to
        make decisions about which agents to use. The description should
        describe the agent's capabilities and how to interact with it."""
        return self._description

    @property
    @abstractmethod
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the agent produces in the
        :attr:`Response.chat_message` field. They must be :class:`BaseChatMessage` types."""
        ...

    @abstractmethod
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Handles incoming messages and returns a response.

        .. note::

            Agents are stateful and the messages passed to this method should
            be the new messages since the last call to this method. The agent
            should maintain its state between calls to this method. For example,
            if the agent needs to remember the previous messages to respond to
            the current message, it should store the previous messages in the
            agent state.

        """
        ...

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Handles incoming messages and returns a stream of messages and
        and the final item is the response. The base implementation in
        :class:`BaseChatAgent` simply calls :meth:`on_messages` and yields
        the messages in the response.

        .. note::

            Agents are stateful and the messages passed to this method should
            be the new messages since the last call to this method. The agent
            should maintain its state between calls to this method. For example,
            if the agent needs to remember the previous messages to respond to
            the current message, it should store the previous messages in the
            agent state.

        """
        response = await self.on_messages(messages, cancellation_token)
        for inner_message in response.inner_messages or []:
            yield inner_message
        yield response

    async def run(
        self,
        *,
        task: str | BaseChatMessage | Sequence[BaseChatMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
        output_task_messages: bool = True,
    ) -> TaskResult:
        """Run the agent with the given task and return the result."""
        with trace_invoke_agent_span(
            agent_name=self.name,
            agent_description=self.description,
        ):
            if cancellation_token is None:
                cancellation_token = CancellationToken()
            input_messages: List[BaseChatMessage] = []
            output_messages: List[BaseAgentEvent | BaseChatMessage] = []
            if task is None:
                pass
            elif isinstance(task, str):
                text_msg = TextMessage(content=task, source="user")
                input_messages.append(text_msg)
                if output_task_messages:
                    output_messages.append(text_msg)
            elif isinstance(task, BaseChatMessage):
                input_messages.append(task)
                if output_task_messages:
                    output_messages.append(task)
            else:
                if not task:
                    raise ValueError("Task list cannot be empty.")
                # Task is a sequence of messages.
                for msg in task:
                    if isinstance(msg, BaseChatMessage):
                        input_messages.append(msg)
                        if output_task_messages:
                            output_messages.append(msg)
                    else:
                        raise ValueError(f"Invalid message type in sequence: {type(msg)}")
            response = await self.on_messages(input_messages, cancellation_token)
            if response.inner_messages is not None:
                output_messages += response.inner_messages
            output_messages.append(response.chat_message)
            return TaskResult(messages=output_messages)

    async def run_stream(
        self,
        *,
        task: str | BaseChatMessage | Sequence[BaseChatMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
        output_task_messages: bool = True,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]:
        """Run the agent with the given task and return a stream of messages
        and the final task result as the last item in the stream.

        Args:
            task: The task to run. Can be a string, a single message, or a sequence of messages.
            cancellation_token: The cancellation token to kill the task immediately.
            output_task_messages: Whether to include task messages in the output stream. Defaults to True for backward compatibility.
        """
        with trace_invoke_agent_span(
            agent_name=self.name,
            agent_description=self.description,
        ):
            if cancellation_token is None:
                cancellation_token = CancellationToken()
            input_messages: List[BaseChatMessage] = []
            output_messages: List[BaseAgentEvent | BaseChatMessage] = []
            if task is None:
                pass
            elif isinstance(task, str):
                text_msg = TextMessage(content=task, source="user")
                input_messages.append(text_msg)
                if output_task_messages:
                    output_messages.append(text_msg)
                    yield text_msg
            elif isinstance(task, BaseChatMessage):
                input_messages.append(task)
                if output_task_messages:
                    output_messages.append(task)
                    yield task
            else:
                if not task:
                    raise ValueError("Task list cannot be empty.")
                for msg in task:
                    if isinstance(msg, BaseChatMessage):
                        input_messages.append(msg)
                        if output_task_messages:
                            output_messages.append(msg)
                            yield msg
                    else:
                        raise ValueError(f"Invalid message type in sequence: {type(msg)}")
            async for message in self.on_messages_stream(input_messages, cancellation_token):
                if isinstance(message, Response):
                    yield message.chat_message
                    output_messages.append(message.chat_message)
                    yield TaskResult(messages=output_messages)
                else:
                    yield message
                    if isinstance(message, ModelClientStreamingChunkEvent):
                        # Skip the model client streaming chunk events.
                        continue
                    output_messages.append(message)

    @abstractmethod
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Resets the agent to its initialization state."""
        ...

    async def on_pause(self, cancellation_token: CancellationToken) -> None:
        """Called when the agent is paused while running in its :meth:`on_messages` or
        :meth:`on_messages_stream` method. This is a no-op by default in the
        :class:`BaseChatAgent` class. Subclasses can override this method to
        implement custom pause behavior."""
        pass

    async def on_resume(self, cancellation_token: CancellationToken) -> None:
        """Called when the agent is resumed from a pause while running in
        its :meth:`on_messages` or :meth:`on_messages_stream` method.
        This is a no-op by default in the :class:`BaseChatAgent` class.
        Subclasses can override this method to implement custom resume behavior."""
        pass

    async def save_state(self) -> Mapping[str, Any]:
        """Export state. Default implementation for stateless agents."""
        return BaseState().model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore agent from saved state. Default implementation for stateless agents."""
        BaseState.model_validate(state)

    async def close(self) -> None:
        """Release any resources held by the agent. This is a no-op by default in the
        :class:`BaseChatAgent` class. Subclasses can override this method to
        implement custom close behavior."""
        pass

from autogen_agentchat.base import Response
from autogen_agentchat.state import SocietyOfMindAgentState
from base import Team

# From agents/_society_of_mind_agent.py
class SocietyOfMindAgentConfig(BaseModel):
    """The declarative configuration for a SocietyOfMindAgent."""

    name: str
    team: ComponentModel
    model_client: ComponentModel
    description: str | None = None
    instruction: str | None = None
    response_prompt: str | None = None
    model_context: ComponentModel | None = None

# From agents/_society_of_mind_agent.py
class SocietyOfMindAgent(BaseChatAgent, Component[SocietyOfMindAgentConfig]):
    """An agent that uses an inner team of agents to generate responses.

    Each time the agent's :meth:`on_messages` or :meth:`on_messages_stream`
    method is called, it runs the inner team of agents and then uses the
    model client to generate a response based on the inner team's messages.
    Once the response is generated, the agent resets the inner team by
    calling :meth:`Team.reset`.

    Limit context size sent to the model:

    You can limit the number of messages sent to the model by setting
    the `model_context` parameter to a :class:`~autogen_core.model_context.BufferedChatCompletionContext`.
    This will limit the number of recent messages sent to the model and can be useful
    when the model has a limit on the number of tokens it can process.
    You can also create your own model context by subclassing
    :class:`~autogen_core.model_context.ChatCompletionContext`.


    Args:
        name (str): The name of the agent.
        team (Team): The team of agents to use.
        model_client (ChatCompletionClient): The model client to use for preparing responses.
        description (str, optional): The description of the agent.
        instruction (str, optional): The instruction to use when generating a response using the inner team's messages.
            Defaults to :attr:`DEFAULT_INSTRUCTION`. It assumes the role of 'system'.
        response_prompt (str, optional): The response prompt to use when generating a response using the inner team's messages.
            Defaults to :attr:`DEFAULT_RESPONSE_PROMPT`. It assumes the role of 'system'.
        model_context (ChatCompletionContext | None, optional): The model context for storing and retrieving :class:`~autogen_core.models.LLMMessage`. It can be preloaded with initial messages. The initial messages will be cleared when the agent is reset.



    Example:

    .. code-block:: python

        import asyncio
        from autogen_agentchat.ui import Console
        from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.conditions import TextMentionTermination


        async def main() -> None:
            model_client = OpenAIChatCompletionClient(model="gpt-4o")

            agent1 = AssistantAgent("assistant1", model_client=model_client, system_message="You are a writer, write well.")
            agent2 = AssistantAgent(
                "assistant2",
                model_client=model_client,
                system_message="You are an editor, provide critical feedback. Respond with 'APPROVE' if the text addresses all feedbacks.",
            )
            inner_termination = TextMentionTermination("APPROVE")
            inner_team = RoundRobinGroupChat([agent1, agent2], termination_condition=inner_termination)

            society_of_mind_agent = SocietyOfMindAgent("society_of_mind", team=inner_team, model_client=model_client)

            agent3 = AssistantAgent(
                "assistant3", model_client=model_client, system_message="Translate the text to Spanish."
            )
            team = RoundRobinGroupChat([society_of_mind_agent, agent3], max_turns=2)

            stream = team.run_stream(task="Write a short story with a surprising ending.")
            await Console(stream)


        asyncio.run(main())
    """

    component_config_schema = SocietyOfMindAgentConfig
    component_provider_override = "autogen_agentchat.agents.SocietyOfMindAgent"

    DEFAULT_INSTRUCTION = "Earlier you were asked to fulfill a request. You and your team worked diligently to address that request. Here is a transcript of that conversation:"
    """str: The default instruction to use when generating a response using the
    inner team's messages. The instruction will be prepended to the inner team's
    messages when generating a response using the model. It assumes the role of
    'system'."""

    DEFAULT_RESPONSE_PROMPT = (
        "Output a standalone response to the original request, without mentioning any of the intermediate discussion."
    )
    """str: The default response prompt to use when generating a response using
    the inner team's messages. It assumes the role of 'system'."""

    DEFAULT_DESCRIPTION = "An agent that uses an inner team of agents to generate responses."
    """str: The default description for a SocietyOfMindAgent."""

    def __init__(
        self,
        name: str,
        team: Team,
        model_client: ChatCompletionClient,
        *,
        description: str = DEFAULT_DESCRIPTION,
        instruction: str = DEFAULT_INSTRUCTION,
        response_prompt: str = DEFAULT_RESPONSE_PROMPT,
        model_context: ChatCompletionContext | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self._team = team
        self._model_client = model_client
        self._instruction = instruction
        self._response_prompt = response_prompt

        if model_context is not None:
            self._model_context = model_context
        else:
            self._model_context = UnboundedChatCompletionContext()

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    @property
    def model_context(self) -> ChatCompletionContext:
        """
        The model context in use by the agent.
        """
        return self._model_context

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        # Call the stream method and collect the messages.
        response: Response | None = None
        async for msg in self.on_messages_stream(messages, cancellation_token):
            if isinstance(msg, Response):
                response = msg
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        # Prepare the task for the team of agents.
        task_messages = list(messages)

        # Run the team of agents.
        result: TaskResult | None = None
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        model_context = self._model_context

        prev_content = await model_context.get_messages()
        if len(prev_content) > 0:
            prev_message = HandoffMessage(
                content="relevant previous messages",
                source=self.name,
                target="",
                context=prev_content,
            )
            task_messages = [prev_message] + task_messages

        if len(task_messages) == 0:
            task = None
        else:
            task = task_messages

        # Use the new output_task_messages parameter to avoid fragile count-based logic
        async for inner_msg in self._team.run_stream(
            task=task, cancellation_token=cancellation_token, output_task_messages=False
        ):
            if isinstance(inner_msg, TaskResult):
                result = inner_msg
            else:
                yield inner_msg
                if isinstance(inner_msg, ModelClientStreamingChunkEvent):
                    # Skip the model client streaming chunk events.
                    continue
                inner_messages.append(inner_msg)
        assert result is not None

        if len(inner_messages) == 0:
            yield Response(
                chat_message=TextMessage(source=self.name, content="No response."),
                inner_messages=[],
                # Response's inner_messages should be empty. Cause that mean is response to outer world.
            )
        else:
            llm_messages: List[LLMMessage] = []

            if self._model_client.model_info.get("multiple_system_messages", False):
                # The model client supports multiple system messages, so we
                llm_messages.append(SystemMessage(content=self._instruction))
            else:
                # The model client does not support multiple system messages, so we
                llm_messages.append(UserMessage(content=self._instruction, source="user"))

            # Generate a response using the model client.
            for message in inner_messages:
                if isinstance(message, BaseChatMessage):
                    llm_messages.append(message.to_model_message())

            if self._model_client.model_info.get("multiple_system_messages", False):
                # The model client supports multiple system messages, so we
                llm_messages.append(SystemMessage(content=self._response_prompt))
            else:
                # The model client does not support multiple system messages, so we
                llm_messages.append(UserMessage(content=self._response_prompt, source="user"))
            completion = await self._model_client.create(messages=llm_messages, cancellation_token=cancellation_token)
            assert isinstance(completion.content, str)
            yield Response(
                chat_message=TextMessage(source=self.name, content=completion.content, models_usage=completion.usage),
                inner_messages=[],
                # Response's inner_messages should be empty. Cause that mean is response to outer world.
            )

        # Add new user/handoff messages to the model context
        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )

        # Reset the team.
        await self._team.reset()

    @staticmethod
    async def _add_messages_to_context(
        model_context: ChatCompletionContext,
        messages: Sequence[BaseChatMessage],
    ) -> None:
        """
        Add incoming messages to the model context.
        """
        for msg in messages:
            if isinstance(msg, HandoffMessage):
                for llm_msg in msg.context:
                    await model_context.add_message(llm_msg)
            await model_context.add_message(msg.to_model_message())

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._team.reset()
        await self._model_context.clear()

    async def save_state(self) -> Mapping[str, Any]:
        team_state = await self._team.save_state()
        state = SocietyOfMindAgentState(inner_team_state=team_state)
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        society_of_mind_state = SocietyOfMindAgentState.model_validate(state)
        await self._team.load_state(society_of_mind_state.inner_team_state)

    def _to_config(self) -> SocietyOfMindAgentConfig:
        return SocietyOfMindAgentConfig(
            name=self.name,
            team=self._team.dump_component(),
            model_client=self._model_client.dump_component(),
            description=self.description,
            instruction=self._instruction,
            response_prompt=self._response_prompt,
            model_context=self._model_context.dump_component(),
        )

    @classmethod
    def _from_config(cls, config: SocietyOfMindAgentConfig) -> Self:
        model_client = ChatCompletionClient.load_component(config.model_client)
        team = Team.load_component(config.team)
        return cls(
            name=config.name,
            team=team,
            model_client=model_client,
            description=config.description or cls.DEFAULT_DESCRIPTION,
            instruction=config.instruction or cls.DEFAULT_INSTRUCTION,
            response_prompt=config.response_prompt or cls.DEFAULT_RESPONSE_PROMPT,
            model_context=ChatCompletionContext.load_component(config.model_context) if config.model_context else None,
        )

from autogen_agentchat.messages import BaseAgentEvent
from autogen_agentchat.messages import BaseChatMessage

# From agents/_message_filter_agent.py
class PerSourceFilter(BaseModel):
    source: str
    position: Optional[Literal["first", "last"]] = None
    count: Optional[int] = None

# From agents/_message_filter_agent.py
class MessageFilterConfig(BaseModel):
    per_source: List[PerSourceFilter]

# From agents/_message_filter_agent.py
class MessageFilterAgentConfig(BaseModel):
    name: str
    wrapped_agent: ComponentModel
    filter: MessageFilterConfig

# From agents/_message_filter_agent.py
class MessageFilterAgent(BaseChatAgent, Component[MessageFilterAgentConfig]):
    """
    A wrapper agent that filters incoming messages before passing them to the inner agent.

    .. warning::

        This is an experimental feature, and the API will change in the future releases.

    This is useful in scenarios like multi-agent workflows where an agent should only
    process a subset of the full message history—for example, only the last message
    from each upstream agent, or only the first message from a specific source.

    Filtering is configured using :class:`MessageFilterConfig`, which supports:
    - Filtering by message source (e.g., only messages from "user" or another agent)
    - Selecting the first N or last N messages from each source
    - If position is `None`, all messages from that source are included

    This agent is compatible with both direct message passing and team-based execution
    such as :class:`~autogen_agentchat.teams.GraphFlow`.

    Example:
        >>> agent_a = MessageFilterAgent(
        ...     name="A",
        ...     wrapped_agent=some_other_agent,
        ...     filter=MessageFilterConfig(
        ...         per_source=[
        ...             PerSourceFilter(source="user", position="first", count=1),
        ...             PerSourceFilter(source="B", position="last", count=2),
        ...         ]
        ...     ),
        ... )

    Example use case with Graph:
        Suppose you have a looping multi-agent graph: A → B → A → B → C.

        You want:
        - A to only see the user message and the last message from B
        - B to see the user message, last message from A, and its own prior responses (for reflection)
        - C to see the user message and the last message from B

        Wrap the agents like so:

        >>> agent_a = MessageFilterAgent(
        ...     name="A",
        ...     wrapped_agent=agent_a_inner,
        ...     filter=MessageFilterConfig(
        ...         per_source=[
        ...             PerSourceFilter(source="user", position="first", count=1),
        ...             PerSourceFilter(source="B", position="last", count=1),
        ...         ]
        ...     ),
        ... )

        >>> agent_b = MessageFilterAgent(
        ...     name="B",
        ...     wrapped_agent=agent_b_inner,
        ...     filter=MessageFilterConfig(
        ...         per_source=[
        ...             PerSourceFilter(source="user", position="first", count=1),
        ...             PerSourceFilter(source="A", position="last", count=1),
        ...             PerSourceFilter(source="B", position="last", count=10),
        ...         ]
        ...     ),
        ... )

        >>> agent_c = MessageFilterAgent(
        ...     name="C",
        ...     wrapped_agent=agent_c_inner,
        ...     filter=MessageFilterConfig(
        ...         per_source=[
        ...             PerSourceFilter(source="user", position="first", count=1),
        ...             PerSourceFilter(source="B", position="last", count=1),
        ...         ]
        ...     ),
        ... )

        Then define the graph:

        >>> graph = DiGraph(
        ...     nodes={
        ...         "A": DiGraphNode(name="A", edges=[DiGraphEdge(target="B")]),
        ...         "B": DiGraphNode(
        ...             name="B",
        ...             edges=[
        ...                 DiGraphEdge(target="C", condition="exit"),
        ...                 DiGraphEdge(target="A", condition="loop"),
        ...             ],
        ...         ),
        ...         "C": DiGraphNode(name="C", edges=[]),
        ...     },
        ...     default_start_node="A",
        ... )

        This will ensure each agent sees only what is needed for its decision or action logic.
    """

    component_config_schema = MessageFilterAgentConfig
    component_provider_override = "autogen_agentchat.agents.MessageFilterAgent"

    def __init__(
        self,
        name: str,
        wrapped_agent: BaseChatAgent,
        filter: MessageFilterConfig,
    ):
        super().__init__(name=name, description=f"{wrapped_agent.description} (with message filtering)")
        self._wrapped_agent = wrapped_agent
        self._filter = filter

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return self._wrapped_agent.produced_message_types

    def _apply_filter(self, messages: Sequence[BaseChatMessage]) -> Sequence[BaseChatMessage]:
        result: List[BaseChatMessage] = []

        for source_filter in self._filter.per_source:
            msgs = [m for m in messages if m.source == source_filter.source]

            if source_filter.position == "first" and source_filter.count:
                msgs = msgs[: source_filter.count]
            elif source_filter.position == "last" and source_filter.count:
                msgs = msgs[-source_filter.count :]

            result.extend(msgs)

        return result

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        filtered = self._apply_filter(messages)
        return await self._wrapped_agent.on_messages(filtered, cancellation_token)

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        filtered = self._apply_filter(messages)
        async for item in self._wrapped_agent.on_messages_stream(filtered, cancellation_token):
            yield item

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._wrapped_agent.on_reset(cancellation_token)

    def _to_config(self) -> MessageFilterAgentConfig:
        return MessageFilterAgentConfig(
            name=self.name,
            wrapped_agent=self._wrapped_agent.dump_component(),
            filter=self._filter,
        )

    @classmethod
    def _from_config(cls, config: MessageFilterAgentConfig) -> "MessageFilterAgent":
        wrapped = BaseChatAgent.load_component(config.wrapped_agent)
        return cls(
            name=config.name,
            wrapped_agent=wrapped,
            filter=config.filter,
        )

from inspect import iscoroutinefunction
from messages import UserInputRequestedEvent

# From agents/_user_proxy_agent.py
class UserProxyAgentConfig(BaseModel):
    """Declarative configuration for the UserProxyAgent."""

    name: str
    description: str = "A human user"
    input_func: str | None = None

# From agents/_user_proxy_agent.py
class UserProxyAgent(BaseChatAgent, Component[UserProxyAgentConfig]):
    """An agent that can represent a human user through an input function.

    This agent can be used to represent a human user in a chat system by providing a custom input function.

    .. note::

        Using :class:`UserProxyAgent` puts a running team in a temporary blocked
        state until the user responds. So it is important to time out the user input
        function and cancel using the :class:`~autogen_core.CancellationToken` if the user does not respond.
        The input function should also handle exceptions and return a default response if needed.

        For typical use cases that involve
        slow human responses, it is recommended to use termination conditions
        such as :class:`~autogen_agentchat.conditions.HandoffTermination` or :class:`~autogen_agentchat.conditions.SourceMatchTermination`
        to stop the running team and return the control to the application.
        You can run the team again with the user input. This way, the state of the team
        can be saved and restored when the user responds.

        See `Human-in-the-loop <https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html>`_ for more information.

    Args:
        name (str): The name of the agent.
        description (str, optional): A description of the agent.
        input_func (Optional[Callable[[str], str]], Callable[[str, Optional[CancellationToken]], Awaitable[str]]): A function that takes a prompt and returns a user input string.

    For examples of integrating with web and UI frameworks, see the following:

    * `FastAPI <https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_fastapi>`_
    * `ChainLit <https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_chainlit>`_

    Example:
        Simple usage case::

            import asyncio
            from autogen_core import CancellationToken
            from autogen_agentchat.agents import UserProxyAgent
            from autogen_agentchat.messages import TextMessage


            async def simple_user_agent():
                agent = UserProxyAgent("user_proxy")
                response = await asyncio.create_task(
                    agent.on_messages(
                        [TextMessage(content="What is your name? ", source="user")],
                        cancellation_token=CancellationToken(),
                    )
                )
                assert isinstance(response.chat_message, TextMessage)
                print(f"Your name is {response.chat_message.content}")

    Example:
        Cancellable usage case::

            import asyncio
            from typing import Any
            from autogen_core import CancellationToken
            from autogen_agentchat.agents import UserProxyAgent
            from autogen_agentchat.messages import TextMessage


            token = CancellationToken()
            agent = UserProxyAgent("user_proxy")


            async def timeout(delay: float):
                await asyncio.sleep(delay)


            def cancellation_callback(task: asyncio.Task[Any]):
                token.cancel()


            async def cancellable_user_agent():
                try:
                    timeout_task = asyncio.create_task(timeout(3))
                    timeout_task.add_done_callback(cancellation_callback)
                    agent_task = asyncio.create_task(
                        agent.on_messages(
                            [TextMessage(content="What is your name? ", source="user")],
                            cancellation_token=token,
                        )
                    )
                    response = await agent_task
                    assert isinstance(response.chat_message, TextMessage)
                    print(f"Your name is {response.chat_message.content}")
                except Exception as e:
                    print(f"Exception: {e}")
                except BaseException as e:
                    print(f"BaseException: {e}")
    """

    component_type = "agent"
    component_provider_override = "autogen_agentchat.agents.UserProxyAgent"
    component_config_schema = UserProxyAgentConfig

    class InputRequestContext:
        def __init__(self) -> None:
            raise RuntimeError(
                "InputRequestContext cannot be instantiated. It is a static class that provides context management for user input requests."
            )

        _INPUT_REQUEST_CONTEXT_VAR: ClassVar[ContextVar[str]] = ContextVar("_INPUT_REQUEST_CONTEXT_VAR")

        @classmethod
        @contextmanager
        def populate_context(cls, ctx: str) -> Generator[None, Any, None]:
            """:meta private:"""
            token = UserProxyAgent.InputRequestContext._INPUT_REQUEST_CONTEXT_VAR.set(ctx)
            try:
                yield
            finally:
                UserProxyAgent.InputRequestContext._INPUT_REQUEST_CONTEXT_VAR.reset(token)

        @classmethod
        def request_id(cls) -> str:
            try:
                return cls._INPUT_REQUEST_CONTEXT_VAR.get()
            except LookupError as e:
                raise RuntimeError(
                    "InputRequestContext.runtime() must be called within the input callback of a UserProxyAgent."
                ) from e

    def __init__(
        self,
        name: str,
        *,
        description: str = "A human user",
        input_func: Optional[InputFuncType] = None,
    ) -> None:
        """Initialize the UserProxyAgent."""
        super().__init__(name=name, description=description)
        self.input_func = input_func or cancellable_input
        self._is_async = iscoroutinefunction(self.input_func)

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Message types this agent can produce."""
        return (TextMessage, HandoffMessage)

    def _get_latest_handoff(self, messages: Sequence[BaseChatMessage]) -> Optional[HandoffMessage]:
        """Find the HandoffMessage in the message sequence that addresses this agent."""
        if len(messages) > 0 and isinstance(messages[-1], HandoffMessage):
            if messages[-1].target == self.name:
                return messages[-1]
            else:
                raise RuntimeError(f"Handoff message target does not match agent name: {messages[-1].source}")
        return None

    async def _get_input(self, prompt: str, cancellation_token: Optional[CancellationToken]) -> str:
        """Handle input based on function signature."""
        try:
            if self._is_async:
                # Cast to AsyncInputFunc for proper typing
                async_func = cast(AsyncInputFunc, self.input_func)
                return await async_func(prompt, cancellation_token)
            else:
                # Cast to SyncInputFunc for proper typing
                sync_func = cast(SyncInputFunc, self.input_func)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, sync_func, prompt)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get user input: {str(e)}") from e

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Handle incoming messages by requesting user input."""
        try:
            # Check for handoff first
            handoff = self._get_latest_handoff(messages)
            prompt = (
                f"Handoff received from {handoff.source}. Enter your response: " if handoff else "Enter your response: "
            )

            request_id = str(uuid.uuid4())

            input_requested_event = UserInputRequestedEvent(request_id=request_id, source=self.name)
            yield input_requested_event
            with UserProxyAgent.InputRequestContext.populate_context(request_id):
                user_input = await self._get_input(prompt, cancellation_token)

            # Return appropriate message type based on handoff presence
            if handoff:
                yield Response(chat_message=HandoffMessage(content=user_input, target=handoff.source, source=self.name))
            else:
                yield Response(chat_message=TextMessage(content=user_input, source=self.name))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get user input: {str(e)}") from e

    async def on_reset(self, cancellation_token: Optional[CancellationToken] = None) -> None:
        """Reset agent state."""
        pass

    def _to_config(self) -> UserProxyAgentConfig:
        # TODO: Add ability to serialie input_func
        return UserProxyAgentConfig(name=self.name, description=self.description, input_func=None)

    @classmethod
    def _from_config(cls, config: UserProxyAgentConfig) -> Self:
        return cls(name=config.name, description=config.description, input_func=None)

# From agents/_user_proxy_agent.py
class InputRequestContext:
        def __init__(self) -> None:
            raise RuntimeError(
                "InputRequestContext cannot be instantiated. It is a static class that provides context management for user input requests."
            )

        _INPUT_REQUEST_CONTEXT_VAR: ClassVar[ContextVar[str]] = ContextVar("_INPUT_REQUEST_CONTEXT_VAR")

        @classmethod
        @contextmanager
        def populate_context(cls, ctx: str) -> Generator[None, Any, None]:
            """:meta private:"""
            token = UserProxyAgent.InputRequestContext._INPUT_REQUEST_CONTEXT_VAR.set(ctx)
            try:
                yield
            finally:
                UserProxyAgent.InputRequestContext._INPUT_REQUEST_CONTEXT_VAR.reset(token)

        @classmethod
        def request_id(cls) -> str:
            try:
                return cls._INPUT_REQUEST_CONTEXT_VAR.get()
            except LookupError as e:
                raise RuntimeError(
                    "InputRequestContext.runtime() must be called within the input callback of a UserProxyAgent."
                ) from e

# From agents/_user_proxy_agent.py
def request_id(cls) -> str:
            try:
                return cls._INPUT_REQUEST_CONTEXT_VAR.get()
            except LookupError as e:
                raise RuntimeError(
                    "InputRequestContext.runtime() must be called within the input callback of a UserProxyAgent."
                ) from e

from autogen_core.code_executor import CodeExecutor
from messages import CodeExecutionEvent
from messages import CodeGenerationEvent

# From agents/_code_executor_agent.py
class CodeExecutorAgentConfig(BaseModel):
    """Configuration for CodeExecutorAgent"""

    name: str
    code_executor: ComponentModel
    model_client: ComponentModel | None = None
    description: str | None = None
    sources: List[str] | None = None
    system_message: str | None = None
    model_client_stream: bool = False
    model_context: ComponentModel | None = None
    supported_languages: List[str] | None = None

# From agents/_code_executor_agent.py
class RetryDecision(BaseModel):
    reason: str
    retry: bool

# From agents/_code_executor_agent.py
class ApprovalRequest(BaseModel):
    """Request for approval of code execution."""

    code: str
    context: List[LLMMessage]

# From agents/_code_executor_agent.py
class ApprovalResponse(BaseModel):
    """Response to approval request."""

    approved: bool
    reason: str

# From agents/_code_executor_agent.py
class CodeExecutorAgent(BaseChatAgent, Component[CodeExecutorAgentConfig]):
    """(Experimental) An agent that generates and executes code snippets based on user instructions.

    .. note::

        This agent is experimental and may change in future releases.

    It is typically used within a team with another agent that generates code snippets
    to be executed or alone with `model_client` provided so that it can generate code
    based on user query, execute it and reflect on the code result.

    When used with `model_client`, it will generate code snippets using the model
    and execute them using the provided `code_executor`. The model will also reflect on the
    code execution results. The agent will yield the final reflection result from the model
    as the final response.

    When used without `model_client`, it will only execute code blocks found in
    :class:`~autogen_agentchat.messages.TextMessage` messages and returns the output
    of the code execution.

    .. note::

        Using :class:`~autogen_agentchat.agents.AssistantAgent` with
        :class:`~autogen_ext.tools.code_execution.PythonCodeExecutionTool`
        is an alternative to this agent. However, the model for that agent will
        have to generate properly escaped code string as a parameter to the tool.

    Args:
        name (str): The name of the agent.
        code_executor (CodeExecutor): The code executor responsible for executing code received in messages
            (:py:class:`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` recommended. See example below)
        model_client (ChatCompletionClient, optional): The model client to use for inference and generating code.
            If not provided, the agent will only execute code blocks found in input messages.
            Currently, the model must support structured output mode, which is required for
            the automatic retry mechanism to work.
        model_client_stream (bool, optional): If `True`, the model client will be used in streaming mode.
            :meth:`on_messages_stream` and :meth:`BaseChatAgent.run_stream` methods will
            also yield :class:`~autogen_agentchat.messages.ModelClientStreamingChunkEvent`
            messages as the model client produces chunks of response. Defaults to `False`.
        description (str, optional): The description of the agent. If not provided,
            :class:`~autogen_agentchat.agents.CodeExecutorAgent.DEFAULT_AGENT_DESCRIPTION` will be used.
        system_message (str, optional): The system message for the model. If provided, it will be prepended to the messages in the model context when making an inference. Set to `None` to disable.
            Defaults to :class:`~autogen_agentchat.agents.CodeExecutorAgent.DEFAULT_SYSTEM_MESSAGE`. This is only used if `model_client` is provided.
        sources (Sequence[str], optional): Check only messages from the specified agents for the code to execute.
            This is useful when the agent is part of a group chat and you want to limit the code execution to messages from specific agents.
            If not provided, all messages will be checked for code blocks.
            This is only used if `model_client` is not provided.
        max_retries_on_error (int, optional): The maximum number of retries on error. If the code execution fails, the agent will retry up to this number of times.
            If the code execution fails after this number of retries, the agent will yield a reflection result.
        supported_languages (List[str], optional): List of programming languages that will be parsed and executed from agent response;
            others will be ignored. Defaults to DEFAULT_SUPPORTED_LANGUAGES.
        approval_func (Optional[Union[Callable[[ApprovalRequest], ApprovalResponse], Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]]], optional): A function that is called before each code execution to get approval.
            The function takes an ApprovalRequest containing the code to be executed and the current context, and returns an ApprovalResponse.
            The function can be either synchronous or asynchronous. If None (default), all code executions are automatically approved.
            If set, the agent cannot be serialized using :meth:`~autogen_agentchat.agents.CodeExecutorAgent.dump_component`.


    .. note::

        It is recommended that the `CodeExecutorAgent` agent uses a Docker container to execute code. This ensures that model-generated code is executed in an isolated environment. To use Docker, your environment must have Docker installed and running.
        Follow the installation instructions for `Docker <https://docs.docker.com/get-docker/>`_.

    .. note::

        The code executor only processes code that is properly formatted in markdown code blocks using triple backticks.
        For example:

        .. code-block:: text

            ```python
            print("Hello World")
            ```

            # or

            ```sh
            echo "Hello World"
            ```

    In this example, we show how to set up a `CodeExecutorAgent` agent that uses the
    :py:class:`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`
    to execute code snippets in a Docker container. The `work_dir` parameter indicates
    where all executed files are first saved locally before being executed in the Docker container.

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import CodeExecutorAgent, ApprovalRequest, ApprovalResponse
            from autogen_agentchat.messages import TextMessage
            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
            from autogen_core import CancellationToken


            def simple_approval_func(request: ApprovalRequest) -> ApprovalResponse:
                \"\"\"Simple approval function that requests user input for code execution approval.\"\"\"
                print("Code execution approval requested:")
                print("=" * 50)
                print(request.code)
                print("=" * 50)

                while True:
                    user_input = input("Do you want to execute this code? (y/n): ").strip().lower()
                    if user_input in ['y', 'yes']:
                        return ApprovalResponse(approved=True, reason='Approved by user')
                    elif user_input in ['n', 'no']:
                        return ApprovalResponse(approved=False, reason='Denied by user')
                    else:
                        print("Please enter 'y' for yes or 'n' for no.")


            async def run_code_executor_agent() -> None:
                # Create a code executor agent that uses a Docker container to execute code.
                code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
                await code_executor.start()
                code_executor_agent = CodeExecutorAgent(
                    "code_executor",
                    code_executor=code_executor,
                    approval_func=simple_approval_func
                )

                # Run the agent with a given code snippet.
                task = TextMessage(
                    content='''Here is some code
            ```python
            print('Hello world')
            ```
            ''',
                    source="user",
                )
                response = await code_executor_agent.on_messages([task], CancellationToken())
                print(response.chat_message)

                # Stop the code executor.
                await code_executor.stop()


            asyncio.run(run_code_executor_agent())

    In this example, we show how to set up a `CodeExecutorAgent` agent that uses the
    :py:class:`~docker.types.DeviceRequest` to expose a GPU to the container for cuda-accelerated code execution.

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import CodeExecutorAgent
            from autogen_agentchat.messages import TextMessage
            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
            from autogen_core import CancellationToken
            from docker.types import DeviceRequest


            async def run_code_executor_agent() -> None:
                # Create a code executor agent that uses a Docker container to execute code.
                code_executor = DockerCommandLineCodeExecutor(
                    work_dir="coding", device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
                )
                await code_executor.start()
                code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)

                # Display the GPU information
                task = TextMessage(
                    content='''Here is some code
            ```sh
            nvidia-smi
            ```
            ''',
                    source="user",
                )
                response = await code_executor_agent.on_messages([task], CancellationToken())
                print(response.chat_message)

                # Stop the code executor.
                await code_executor.stop()


            asyncio.run(run_code_executor_agent())

    In the following example, we show how to setup `CodeExecutorAgent` without `model_client` parameter for executing code blocks generated by other agents in a group chat using :py:class:`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`

        .. code-block:: python

            import asyncio

            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, ApprovalRequest, ApprovalResponse
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_agentchat.ui import Console

            termination_condition = MaxMessageTermination(3)


            def group_chat_approval_func(request: ApprovalRequest) -> ApprovalResponse:
                \"\"\"Approval function for group chat that allows basic Python operations.\"\"\"
                # Allow common safe operations
                safe_operations = ["print(", "import ", "def ", "class ", "if ", "for ", "while "]
                if any(op in request.code for op in safe_operations):
                    return ApprovalResponse(approved=True, reason='Safe Python operation')

                # Deny file system operations in group chat
                dangerous_operations = ["open(", "file(", "os.", "subprocess", "eval(", "exec("]
                if any(op in request.code for op in dangerous_operations):
                    return ApprovalResponse(approved=False, reason='File system or dangerous operation not allowed')

                return ApprovalResponse(approved=True, reason='Operation approved')


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                # define the Docker CLI Code Executor
                code_executor = DockerCommandLineCodeExecutor(work_dir="coding")

                # start the execution container
                await code_executor.start()

                code_executor_agent = CodeExecutorAgent(
                    "code_executor_agent",
                    code_executor=code_executor,
                    approval_func=group_chat_approval_func
                )
                coder_agent = AssistantAgent("coder_agent", model_client=model_client)

                groupchat = RoundRobinGroupChat(
                    participants=[coder_agent, code_executor_agent], termination_condition=termination_condition
                )

                task = "Write python code to print Hello World!"
                await Console(groupchat.run_stream(task=task))

                # stop the execution container
                await code_executor.stop()


            asyncio.run(main())

        .. code-block:: text

            ---------- user ----------
            Write python code to print Hello World!
            ---------- coder_agent ----------
            Certainly! Here's a simple Python code to print "Hello World!":

            ```python
            print("Hello World!")
            ```

            You can run this code in any Python environment to display the message.
            ---------- code_executor_agent ----------
            Hello World!

    In the following example, we show how to setup `CodeExecutorAgent` with `model_client`
    that can generate its own code without the help of any other agent and executing it in
    :py:class:`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`.
    It also demonstrates using a model-based approval function that reviews the code for safety before execution.

        .. code-block:: python

            import asyncio

            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_core.models import SystemMessage, UserMessage

            from autogen_agentchat.agents import CodeExecutorAgent, ApprovalRequest, ApprovalResponse
            from autogen_agentchat.conditions import TextMessageTermination
            from autogen_agentchat.ui import Console

            termination_condition = TextMessageTermination("code_executor_agent")


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                async def model_client_approval_func(request: ApprovalRequest) -> ApprovalResponse:
                    instruction = "Approve or reject the code in the last message based on whether it is dangerous or not. Use the following JSON format for your response: {approved: true/false, reason: 'your reason here'}"
                    response = await model_client.create(
                        messages=[SystemMessage(content=instruction)]
                        + request.context
                        + [UserMessage(content=request.code, source="user")],
                        json_output=ApprovalResponse,
                    )
                    assert isinstance(response.content, str)
                    return ApprovalResponse.model_validate_json(response.content)

                # define the Docker CLI Code Executor
                code_executor = DockerCommandLineCodeExecutor(work_dir="coding")

                # start the execution container
                await code_executor.start()

                code_executor_agent = CodeExecutorAgent(
                    "code_executor_agent",
                    code_executor=code_executor,
                    model_client=model_client,
                    approval_func=model_client_approval_func,
                )

                task = "Write python code to print Hello World!"
                await Console(code_executor_agent.run_stream(task=task))

                # stop the execution container
                await code_executor.stop()


            asyncio.run(main())


        .. code-block:: text

            ---------- user ----------
            Write python code to print Hello World!
            ---------- code_executor_agent ----------
            Certainly! Here is a simple Python code to print "Hello World!" to the console:

            ```python
            print("Hello World!")
            ```

            Let's execute it to confirm the output.
            ---------- code_executor_agent ----------
            Hello World!

            ---------- code_executor_agent ----------
            The code has been executed successfully, and it printed "Hello World!" as expected. If you have any more requests or questions, feel free to ask!

    """

    DEFAULT_TERMINAL_DESCRIPTION = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks)."
    DEFAULT_AGENT_DESCRIPTION = "A Code Execution Agent that generates and executes Python and shell scripts based on user instructions. It ensures correctness, efficiency, and minimal errors while gracefully handling edge cases."
    DEFAULT_SYSTEM_MESSAGE = "You are a Code Execution Agent. Your role is to generate and execute Python code and shell scripts based on user instructions, ensuring correctness, efficiency, and minimal errors. Handle edge cases gracefully. Python code should be provided in ```python code blocks, and sh shell scripts should be provided in ```sh code blocks for execution."
    NO_CODE_BLOCKS_FOUND_MESSAGE = "No code blocks found in the thread. Please provide at least one markdown-encoded code block to execute (i.e., quoting code in ```python or ```sh code blocks)."
    DEFAULT_SUPPORTED_LANGUAGES = ["python", "sh"]

    component_config_schema = CodeExecutorAgentConfig
    component_provider_override = "autogen_agentchat.agents.CodeExecutorAgent"

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        model_client: ChatCompletionClient | None = None,
        model_context: ChatCompletionContext | None = None,
        model_client_stream: bool = False,
        max_retries_on_error: int = 0,
        description: str | None = None,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
        sources: Sequence[str] | None = None,
        supported_languages: List[str] | None = None,
        approval_func: Optional[ApprovalFuncType] = None,
    ) -> None:
        if description is None:
            if model_client is None:
                description = CodeExecutorAgent.DEFAULT_TERMINAL_DESCRIPTION
            else:
                description = CodeExecutorAgent.DEFAULT_AGENT_DESCRIPTION

        super().__init__(name=name, description=description)
        self._code_executor = code_executor
        self._sources = sources
        self._model_client_stream = model_client_stream
        self._max_retries_on_error = max_retries_on_error
        self._approval_func = approval_func
        self._approval_func_is_async = approval_func is not None and iscoroutinefunction(approval_func)

        if supported_languages is not None:
            self._supported_languages = supported_languages
        else:
            self._supported_languages = CodeExecutorAgent.DEFAULT_SUPPORTED_LANGUAGES

        self._supported_languages_regex = "|".join(re.escape(lang) for lang in self._supported_languages)

        self._model_client = None
        if model_client is not None:
            self._model_client = model_client

        if model_context is not None:
            self._model_context = model_context
        else:
            self._model_context = UnboundedChatCompletionContext()

        self._system_messaages: List[SystemMessage] = []
        if system_message is None:
            self._system_messages = []
        else:
            self._system_messages = [SystemMessage(content=system_message)]

        if self._max_retries_on_error > 0:
            if not self._model_client or not self._model_client.model_info:
                raise ValueError("model_client.model_info must be provided when max_retries_on_error > 0")
            if not self._model_client.model_info["structured_output"]:
                raise ValueError("Specified model_client doesn't support structured output mode.")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the code executor agent produces."""
        return (TextMessage,)

    @property
    def model_context(self) -> ChatCompletionContext:
        """
        The model context in use by the agent.
        """
        return self._model_context

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        system_messages = self._system_messages
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        max_retries_on_error = self._max_retries_on_error

        execution_result: CodeResult | None = None
        if model_client is None:  # default behaviour for backward compatibility
            # execute generated code if present
            code_blocks: List[CodeBlock] = await self.extract_code_blocks_from_messages(messages)
            if not code_blocks:
                yield Response(
                    chat_message=TextMessage(
                        content=self.NO_CODE_BLOCKS_FOUND_MESSAGE,
                        source=agent_name,
                    )
                )
                return
            execution_result = await self.execute_code_block(code_blocks, cancellation_token)
            yield Response(chat_message=TextMessage(content=execution_result.output, source=self.name))
            return

        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []

        for nth_try in range(max_retries_on_error + 1):  # Do one default generation, execution and inference loop
            # Step 1: Add new user/handoff messages to the model context
            await self._add_messages_to_context(
                model_context=model_context,
                messages=messages,
            )

            # Step 2: Run inference with the model context
            model_result = None
            async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                agent_name=agent_name,
                cancellation_token=cancellation_token,
            ):
                if isinstance(inference_output, CreateResult):
                    model_result = inference_output
                else:
                    # Streaming chunk event
                    yield inference_output

            assert model_result is not None, "No model result was produced."

            # Step 3: [NEW] If the model produced a hidden "thought," yield it as an event
            if model_result.thought:
                thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
                yield thought_event
                inner_messages.append(thought_event)

            # Step 4: Add the assistant message to the model context (including thought if present)
            await model_context.add_message(
                AssistantMessage(
                    content=model_result.content,
                    source=agent_name,
                    thought=getattr(model_result, "thought", None),
                )
            )

            # Step 5: Extract the code blocks from inferred text
            assert isinstance(model_result.content, str), "Expected inferred model_result.content to be of type str."
            code_blocks = self._extract_markdown_code_blocks(str(model_result.content))

            # Step 6: Exit the loop if no code blocks found
            if not code_blocks:
                yield Response(
                    chat_message=TextMessage(
                        content=str(model_result.content),
                        source=agent_name,
                    )
                )
                return

            # Step 7: Yield a CodeGenerationEvent
            inferred_text_message: CodeGenerationEvent = CodeGenerationEvent(
                retry_attempt=nth_try,
                content=model_result.content,
                code_blocks=code_blocks,
                source=agent_name,
            )

            yield inferred_text_message

            # Step 8: Execute the extracted code blocks
            execution_result = await self.execute_code_block(inferred_text_message.code_blocks, cancellation_token)

            # Step 9: Update model context with the code execution result
            await model_context.add_message(
                UserMessage(
                    content=execution_result.output,
                    source=agent_name,
                )
            )

            # Step 10: Yield a CodeExecutionEvent
            yield CodeExecutionEvent(retry_attempt=nth_try, result=execution_result, source=self.name)

            # If execution was successful or last retry, then exit
            if execution_result.exit_code == 0 or nth_try == max_retries_on_error:
                break

            # Step 11: If exit code is non-zero and retries are available then
            #          make an inference asking if we should retry or not
            chat_context = await model_context.get_messages()

            retry_prompt = (
                f"The most recent code execution resulted in an error:\n{execution_result.output}\n\n"
                "Should we attempt to resolve it? Please respond with:\n"
                "- A boolean value for 'retry' indicating whether it should be retried.\n"
                "- A detailed explanation in 'reason' that identifies the issue, justifies your decision to retry or not, and outlines how you would resolve the error if a retry is attempted."
            )

            chat_context = chat_context + [
                UserMessage(
                    content=retry_prompt,
                    source=agent_name,
                )
            ]

            response = await model_client.create(messages=chat_context, json_output=RetryDecision)

            assert isinstance(
                response.content, str
            ), "Expected structured response for retry decision to be of type str."
            should_retry_generation = RetryDecision.model_validate_json(str(response.content))

            # Exit if no-retry is needed
            if not should_retry_generation.retry:
                break

            yield CodeGenerationEvent(
                retry_attempt=nth_try,
                content=f"Attempt number: {nth_try + 1}\nProposed correction: {should_retry_generation.reason}",
                code_blocks=[],
                source=agent_name,
            )

        # Always reflect on the execution result
        async for reflection_response in CodeExecutorAgent._reflect_on_code_block_results_flow(
            system_messages=system_messages,
            model_client=model_client,
            model_client_stream=model_client_stream,
            model_context=model_context,
            agent_name=agent_name,
            inner_messages=inner_messages,
        ):
            yield reflection_response  # Last reflection_response is of type Response so it will finish the routine

    async def extract_code_blocks_from_messages(self, messages: Sequence[BaseChatMessage]) -> List[CodeBlock]:
        # Extract code blocks from the messages.
        code_blocks: List[CodeBlock] = []
        for msg in messages:
            if self._sources is None or msg.source in self._sources:
                if isinstance(msg, TextMessage):
                    code_blocks.extend(self._extract_markdown_code_blocks(msg.content))
                # TODO: handle other message types if needed
        return code_blocks

    async def execute_code_block(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> CodeResult:
        # Check for approval before executing code blocks
        if self._approval_func is not None:
            # Combine all code blocks into a single string for approval
            combined_code = "\n\n".join([f"```{block.language}\n{block.code}\n```" for block in code_blocks])

            # Get the current context from model_context
            context_messages = await self._model_context.get_messages()

            # Create approval request
            approval_request = ApprovalRequest(code=combined_code, context=context_messages)

            # Get approval (handle both sync and async functions)
            if self._approval_func_is_async:
                # Cast to AsyncApprovalFunc for proper typing
                async_func = cast(AsyncApprovalFunc, self._approval_func)
                approval_response = await async_func(approval_request)
            else:
                # Cast to SyncApprovalFunc for proper typing
                sync_func = cast(SyncApprovalFunc, self._approval_func)
                approval_response = sync_func(approval_request)

            # If not approved, return error result
            if not approval_response.approved:
                return CodeResult(
                    exit_code=1, output=f"Code execution was not approved. Reason: {approval_response.reason}"
                )

        # Execute the code blocks.
        result = await self._code_executor.execute_code_blocks(code_blocks, cancellation_token=cancellation_token)

        if result.output.strip() == "":
            # No output
            result.output = f"The script ran but produced no output to console. The POSIX exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed to stdout."
        elif result.exit_code != 0:
            # Error
            result.output = f"The script ran, then exited with an error (POSIX exit code: {result.exit_code})\nIts output was:\n{result.output}"

        return result

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Its a no-op as the code executor agent has no mutable state."""
        pass

    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        pattern = re.compile(rf"```(?:\s*({self._supported_languages_regex}))\n([\s\S]*?)```", re.IGNORECASE)
        matches = pattern.findall(markdown_text)
        code_blocks: List[CodeBlock] = []
        for match in matches:
            language = match[0].strip() if match[0] else ""
            code_content = match[1]
            code_blocks.append(CodeBlock(code=code_content, language=language))
        return code_blocks

    def _to_config(self) -> CodeExecutorAgentConfig:
        if self._approval_func is not None:
            raise ValueError(
                "Cannot serialize CodeExecutorAgent with approval_func set. The approval function is not serializable."
            )

        return CodeExecutorAgentConfig(
            name=self.name,
            model_client=(self._model_client.dump_component() if self._model_client is not None else None),
            code_executor=self._code_executor.dump_component(),
            description=self.description,
            sources=list(self._sources) if self._sources is not None else None,
            system_message=(
                self._system_messages[0].content
                if self._system_messages and isinstance(self._system_messages[0].content, str)
                else None
            ),
            model_client_stream=self._model_client_stream,
            model_context=self._model_context.dump_component(),
            supported_languages=self._supported_languages,
        )

    @classmethod
    def _from_config(cls, config: CodeExecutorAgentConfig) -> Self:
        return cls(
            name=config.name,
            model_client=(
                ChatCompletionClient.load_component(config.model_client) if config.model_client is not None else None
            ),
            code_executor=CodeExecutor.load_component(config.code_executor),
            description=config.description,
            sources=config.sources,
            system_message=config.system_message,
            model_client_stream=config.model_client_stream,
            model_context=ChatCompletionContext.load_component(config.model_context) if config.model_context else None,
            supported_languages=config.supported_languages,
            approval_func=None,  # approval_func cannot be serialized, so it's always None when loading from config
        )

    @staticmethod
    def _get_compatible_context(model_client: ChatCompletionClient, messages: List[LLMMessage]) -> Sequence[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)

    @classmethod
    async def _call_llm(
        cls,
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        agent_name: str,
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[Union[CreateResult, ModelClientStreamingChunkEvent], None]:
        """
        Perform a model inference and yield either streaming chunk events or the final CreateResult.
        """
        all_messages = await model_context.get_messages()
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=system_messages + all_messages)

        if model_client_stream:
            model_result: Optional[CreateResult] = None
            async for chunk in model_client.create_stream(
                llm_messages, tools=[], cancellation_token=cancellation_token
            ):
                if isinstance(chunk, CreateResult):
                    model_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source=agent_name)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
            if model_result is None:
                raise RuntimeError("No final model result in streaming mode.")
            yield model_result
        else:
            model_result = await model_client.create(llm_messages, tools=[], cancellation_token=cancellation_token)
            yield model_result

    @staticmethod
    async def _add_messages_to_context(
        model_context: ChatCompletionContext,
        messages: Sequence[BaseChatMessage],
    ) -> None:
        """
        Add incoming messages to the model context.
        """
        for msg in messages:
            if isinstance(msg, HandoffMessage):
                for llm_msg in msg.context:
                    await model_context.add_message(llm_msg)
            await model_context.add_message(msg.to_model_message())

    @classmethod
    async def _reflect_on_code_block_results_flow(
        cls,
        system_messages: List[SystemMessage],
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        model_context: ChatCompletionContext,
        agent_name: str,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
    ) -> AsyncGenerator[Response | ModelClientStreamingChunkEvent | ThoughtEvent, None]:
        """
        If reflect_on_code_block_results=True, we do another inference based on tool results
        and yield the final text response (or streaming chunks).
        """
        all_messages = system_messages + await model_context.get_messages()
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=all_messages)

        reflection_result: Optional[CreateResult] = None

        if model_client_stream:
            async for chunk in model_client.create_stream(llm_messages):
                if isinstance(chunk, CreateResult):
                    reflection_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source=agent_name)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
        else:
            reflection_result = await model_client.create(llm_messages)

        if not reflection_result or not isinstance(reflection_result.content, str):
            raise RuntimeError("Reflect on tool use produced no valid text response.")

        # --- NEW: If the reflection produced a thought, yield it ---
        if reflection_result.thought:
            thought_event = ThoughtEvent(content=reflection_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # Add to context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=reflection_result.content,
                source=agent_name,
                thought=getattr(reflection_result, "thought", None),
            )
        )

        yield Response(
            chat_message=TextMessage(
                content=reflection_result.content,
                source=agent_name,
                models_usage=reflection_result.usage,
            ),
            inner_messages=inner_messages,
        )


# From state/_states.py
class BaseState(BaseModel):
    """Base class for all saveable state"""

    type: str = Field(default="BaseState")
    version: str = Field(default="1.0.0")

# From state/_states.py
class AssistantAgentState(BaseState):
    """State for an assistant agent."""

    llm_context: Mapping[str, Any] = Field(default_factory=lambda: dict([("messages", [])]))
    type: str = Field(default="AssistantAgentState")

# From state/_states.py
class TeamState(BaseState):
    """State for a team of agents."""

    agent_states: Mapping[str, Any] = Field(default_factory=dict)
    type: str = Field(default="TeamState")

# From state/_states.py
class BaseGroupChatManagerState(BaseState):
    """Base state for all group chat managers."""

    message_thread: List[Mapping[str, Any]] = Field(default_factory=list)
    current_turn: int = Field(default=0)
    type: str = Field(default="BaseGroupChatManagerState")

# From state/_states.py
class ChatAgentContainerState(BaseState):
    """State for a container of chat agents."""

    agent_state: Mapping[str, Any] = Field(default_factory=dict)
    message_buffer: List[Mapping[str, Any]] = Field(default_factory=list)
    type: str = Field(default="ChatAgentContainerState")

# From state/_states.py
class RoundRobinManagerState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.RoundRobinGroupChat` manager."""

    next_speaker_index: int = Field(default=0)
    type: str = Field(default="RoundRobinManagerState")

# From state/_states.py
class SelectorManagerState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.SelectorGroupChat` manager."""

    previous_speaker: Optional[str] = Field(default=None)
    type: str = Field(default="SelectorManagerState")

# From state/_states.py
class SwarmManagerState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.Swarm` manager."""

    current_speaker: str = Field(default="")
    type: str = Field(default="SwarmManagerState")

# From state/_states.py
class MagenticOneOrchestratorState(BaseGroupChatManagerState):
    """State for :class:`~autogen_agentchat.teams.MagneticOneGroupChat` orchestrator."""

    task: str = Field(default="")
    facts: str = Field(default="")
    plan: str = Field(default="")
    n_rounds: int = Field(default=0)
    n_stalls: int = Field(default=0)
    type: str = Field(default="MagenticOneOrchestratorState")

# From state/_states.py
class SocietyOfMindAgentState(BaseState):
    """State for a Society of Mind agent."""

    inner_team_state: Mapping[str, Any] = Field(default_factory=dict)
    type: str = Field(default="SocietyOfMindAgentState")

import traceback
from messages import StopMessage

# From _group_chat/_events.py
class SerializableException(BaseModel):
    """A serializable exception."""

    error_type: str
    """The type of error that occurred."""

    error_message: str
    """The error message that describes the error."""

    traceback: str | None = None
    """The traceback of the error, if available."""

    @classmethod
    def from_exception(cls, exc: Exception) -> "SerializableException":
        """Create a GroupChatError from an exception."""
        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback="\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    def __str__(self) -> str:
        """Return a string representation of the error, including the traceback if available."""
        if self.traceback:
            return f"{self.error_type}: {self.error_message}\nTraceback:\n{self.traceback}"
        return f"{self.error_type}: {self.error_message}"

# From _group_chat/_events.py
class GroupChatStart(BaseModel):
    """A request to start a group chat."""

    messages: List[SerializeAsAny[BaseChatMessage]] | None = None
    """An optional list of messages to start the group chat."""

    output_task_messages: bool = True
    """Whether to include task messages in the output. Defaults to True for backward compatibility."""

# From _group_chat/_events.py
class GroupChatAgentResponse(BaseModel):
    """A response published to a group chat."""

    response: SerializeAsAny[Response]
    """The response from an agent."""

    name: str
    """The name of the agent that produced the response."""

# From _group_chat/_events.py
class GroupChatTeamResponse(BaseModel):
    """A response published to a group chat from a team."""

    result: SerializeAsAny[TaskResult]
    """The result from a team."""

    name: str
    """The name of the team that produced the response."""

# From _group_chat/_events.py
class GroupChatRequestPublish(BaseModel):
    """A request to publish a message to a group chat."""

    ...

# From _group_chat/_events.py
class GroupChatMessage(BaseModel):
    """A message from a group chat."""

    message: SerializeAsAny[BaseAgentEvent | BaseChatMessage]
    """The message that was published."""

# From _group_chat/_events.py
class GroupChatTermination(BaseModel):
    """A message indicating that a group chat has terminated."""

    message: StopMessage
    """The stop message that indicates the reason of termination."""

    error: SerializableException | None = None
    """The error that occurred, if any."""

# From _group_chat/_events.py
class GroupChatReset(BaseModel):
    """A request to reset the agents in the group chat."""

    ...

# From _group_chat/_events.py
class GroupChatPause(BaseModel):
    """A request to pause the group chat."""

    ...

# From _group_chat/_events.py
class GroupChatResume(BaseModel):
    """A request to resume the group chat."""

    ...

# From _group_chat/_events.py
class GroupChatError(BaseModel):
    """A message indicating that an error occurred in the group chat."""

    error: SerializableException
    """The error that occurred."""

# From _group_chat/_events.py
def from_exception(cls, exc: Exception) -> "SerializableException":
        """Create a GroupChatError from an exception."""
        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback="\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

from autogen_core import AgentRuntime
from autogen_core.models import ModelFamily
from  import TRACE_LOGGER_NAME
from base import TerminationCondition
from messages import MessageFactory
from messages import SelectorEvent
from state import SelectorManagerState
from _base_group_chat import BaseGroupChat
from _base_group_chat_manager import BaseGroupChatManager
from _events import GroupChatTermination

# From _group_chat/_selector_group_chat.py
class SelectorGroupChatManager(BaseGroupChatManager):
    """A group chat manager that selects the next speaker using a ChatCompletion
    model and a custom selector function."""

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
        model_client: ChatCompletionClient,
        selector_prompt: str,
        allow_repeated_speaker: bool,
        selector_func: Optional[SelectorFuncType],
        max_selector_attempts: int,
        candidate_func: Optional[CandidateFuncType],
        emit_team_events: bool,
        model_context: ChatCompletionContext | None,
        model_client_streaming: bool = False,
    ) -> None:
        super().__init__(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
            message_factory,
            emit_team_events,
        )
        self._model_client = model_client
        self._selector_prompt = selector_prompt
        self._previous_speaker: str | None = None
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func
        self._is_selector_func_async = iscoroutinefunction(self._selector_func)
        self._max_selector_attempts = max_selector_attempts
        self._candidate_func = candidate_func
        self._is_candidate_func_async = iscoroutinefunction(self._candidate_func)
        self._model_client_streaming = model_client_streaming
        if model_context is not None:
            self._model_context = model_context
        else:
            self._model_context = UnboundedChatCompletionContext()
        self._cancellation_token = CancellationToken()

    async def validate_group_state(self, messages: List[BaseChatMessage] | None) -> None:
        pass

    async def reset(self) -> None:
        self._current_turn = 0
        self._message_thread.clear()
        await self._model_context.clear()
        if self._termination_condition is not None:
            await self._termination_condition.reset()
        self._previous_speaker = None

    async def save_state(self) -> Mapping[str, Any]:
        state = SelectorManagerState(
            message_thread=[msg.dump() for msg in self._message_thread],
            current_turn=self._current_turn,
            previous_speaker=self._previous_speaker,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        selector_state = SelectorManagerState.model_validate(state)
        self._message_thread = [self._message_factory.create(msg) for msg in selector_state.message_thread]
        await self._add_messages_to_context(
            self._model_context, [msg for msg in self._message_thread if isinstance(msg, BaseChatMessage)]
        )
        self._current_turn = selector_state.current_turn
        self._previous_speaker = selector_state.previous_speaker

    @staticmethod
    async def _add_messages_to_context(
        model_context: ChatCompletionContext,
        messages: Sequence[BaseChatMessage],
    ) -> None:
        """
        Add incoming messages to the model context.
        """
        for msg in messages:
            if isinstance(msg, HandoffMessage):
                for llm_msg in msg.context:
                    await model_context.add_message(llm_msg)
            await model_context.add_message(msg.to_model_message())

    async def update_message_thread(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> None:
        self._message_thread.extend(messages)
        base_chat_messages = [m for m in messages if isinstance(m, BaseChatMessage)]
        await self._add_messages_to_context(self._model_context, base_chat_messages)

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str] | str:
        """Selects the next speaker in a group chat using a ChatCompletion client,
        with the selector function as override if it returns a speaker name.

        .. note::

            This method always returns a single speaker name.

        A key assumption is that the agent type is the same as the topic type, which we use as the agent name.
        """
        # Use the selector function if provided.
        if self._selector_func is not None:
            if self._is_selector_func_async:
                async_selector_func = cast(AsyncSelectorFunc, self._selector_func)
                speaker = await async_selector_func(thread)
            else:
                sync_selector_func = cast(SyncSelectorFunc, self._selector_func)
                speaker = sync_selector_func(thread)
            if speaker is not None:
                if speaker not in self._participant_names:
                    raise ValueError(
                        f"Selector function returned an invalid speaker name: {speaker}. "
                        f"Expected one of: {self._participant_names}."
                    )
                # Skip the model based selection.
                return [speaker]

        # Use the candidate function to filter participants if provided
        if self._candidate_func is not None:
            if self._is_candidate_func_async:
                async_candidate_func = cast(AsyncCandidateFunc, self._candidate_func)
                participants = await async_candidate_func(thread)
            else:
                sync_candidate_func = cast(SyncCandidateFunc, self._candidate_func)
                participants = sync_candidate_func(thread)
            if not participants:
                raise ValueError("Candidate function must return a non-empty list of participant names.")
            if not all(p in self._participant_names for p in participants):
                raise ValueError(
                    f"Candidate function returned invalid participant names: {participants}. "
                    f"Expected one of: {self._participant_names}."
                )
        else:
            # Construct the candidate agent list to be selected from, skip the previous speaker if not allowed.
            if self._previous_speaker is not None and not self._allow_repeated_speaker:
                participants = [p for p in self._participant_names if p != self._previous_speaker]
            else:
                participants = list(self._participant_names)

        assert len(participants) > 0

        # Construct agent roles.
        # Each agent sould appear on a single line.
        roles = ""
        for topic_type, description in zip(self._participant_names, self._participant_descriptions, strict=True):
            roles += re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
        roles = roles.strip()

        # Select the next speaker.
        if len(participants) > 1:
            agent_name = await self._select_speaker(roles, participants, self._max_selector_attempts)
        else:
            agent_name = participants[0]
        self._previous_speaker = agent_name
        trace_logger.debug(f"Selected speaker: {agent_name}")
        return [agent_name]

    def construct_message_history(self, message_history: List[LLMMessage]) -> str:
        # Construct the history of the conversation.
        history_messages: List[str] = []
        for msg in message_history:
            if isinstance(msg, UserMessage) or isinstance(msg, AssistantMessage):
                message = f"{msg.source}: {msg.content}"
                history_messages.append(
                    message.rstrip() + "\n\n"
                )  # Create some consistency for how messages are separated in the transcript

        history: str = "\n".join(history_messages)
        return history

    async def _select_speaker(self, roles: str, participants: List[str], max_attempts: int) -> str:
        model_context_messages = await self._model_context.get_messages()
        model_context_history = self.construct_message_history(model_context_messages)

        select_speaker_prompt = self._selector_prompt.format(
            roles=roles, participants=str(participants), history=model_context_history
        )

        select_speaker_messages: List[SystemMessage | UserMessage | AssistantMessage]
        if ModelFamily.is_openai(self._model_client.model_info["family"]):
            select_speaker_messages = [SystemMessage(content=select_speaker_prompt)]
        else:
            # Many other models need a UserMessage to respond to
            select_speaker_messages = [UserMessage(content=select_speaker_prompt, source="user")]

        num_attempts = 0
        while num_attempts < max_attempts:
            num_attempts += 1
            if self._model_client_streaming:
                chunk: CreateResult | str = ""
                async for _chunk in self._model_client.create_stream(messages=select_speaker_messages):
                    chunk = _chunk
                    if self._emit_team_events:
                        if isinstance(chunk, str):
                            await self._output_message_queue.put(
                                ModelClientStreamingChunkEvent(content=cast(str, _chunk), source=self._name)
                            )
                        else:
                            assert isinstance(chunk, CreateResult)
                            assert isinstance(chunk.content, str)
                            await self._output_message_queue.put(
                                SelectorEvent(content=chunk.content, source=self._name)
                            )
                # The last chunk must be CreateResult.
                assert isinstance(chunk, CreateResult)
                response = chunk
            else:
                response = await self._model_client.create(messages=select_speaker_messages)
            assert isinstance(response.content, str)
            select_speaker_messages.append(AssistantMessage(content=response.content, source="selector"))
            # NOTE: we use all participant names to check for mentions, even if the previous speaker is not allowed.
            # This is because the model may still select the previous speaker, and we want to catch that.
            mentions = self._mentioned_agents(response.content, self._participant_names)
            if len(mentions) == 0:
                trace_logger.debug(f"Model failed to select a valid name: {response.content} (attempt {num_attempts})")
                feedback = f"No valid name was mentioned. Please select from: {str(participants)}."
                select_speaker_messages.append(UserMessage(content=feedback, source="user"))
            elif len(mentions) > 1:
                trace_logger.debug(f"Model selected multiple names: {str(mentions)} (attempt {num_attempts})")
                feedback = (
                    f"Expected exactly one name to be mentioned. Please select only one from: {str(participants)}."
                )
                select_speaker_messages.append(UserMessage(content=feedback, source="user"))
            else:
                agent_name = list(mentions.keys())[0]
                if (
                    not self._allow_repeated_speaker
                    and self._previous_speaker is not None
                    and agent_name == self._previous_speaker
                ):
                    trace_logger.debug(f"Model selected the previous speaker: {agent_name} (attempt {num_attempts})")
                    feedback = (
                        f"Repeated speaker is not allowed, please select a different name from: {str(participants)}."
                    )
                    select_speaker_messages.append(UserMessage(content=feedback, source="user"))
                else:
                    # Valid selection
                    trace_logger.debug(f"Model selected a valid name: {agent_name} (attempt {num_attempts})")
                    return agent_name

        if self._previous_speaker is not None:
            trace_logger.warning(f"Model failed to select a speaker after {max_attempts}, using the previous speaker.")
            return self._previous_speaker
        trace_logger.warning(
            f"Model failed to select a speaker after {max_attempts} and there was no previous speaker, using the first participant."
        )
        return participants[0]

    def _mentioned_agents(self, message_content: str, agent_names: List[str]) -> Dict[str, int]:
        """Counts the number of times each agent is mentioned in the provided message content.
        Agent names will match under any of the following conditions (all case-sensitive):
        - Exact name match
        - If the agent name has underscores it will match with spaces instead (e.g. 'Story_writer' == 'Story writer')
        - If the agent name has underscores it will match with '\\_' instead of '_' (e.g. 'Story_writer' == 'Story\\_writer')

        Args:
            message_content (Union[str, List]): The content of the message, either as a single string or a list of strings.
            agents (List[Agent]): A list of Agent objects, each having a 'name' attribute to be searched in the message content.

        Returns:
            Dict: a counter for mentioned agents.
        """
        mentions: Dict[str, int] = dict()
        for name in agent_names:
            # Finds agent mentions, taking word boundaries into account,
            # accommodates escaping underscores and underscores as spaces
            regex = (
                r"(?<=\W)("
                + re.escape(name)
                + r"|"
                + re.escape(name.replace("_", " "))
                + r"|"
                + re.escape(name.replace("_", r"\_"))
                + r")(?=\W)"
            )
            # Pad the message to help with matching
            count = len(re.findall(regex, f" {message_content} "))
            if count > 0:
                mentions[name] = count
        return mentions

# From _group_chat/_selector_group_chat.py
class SelectorGroupChatConfig(BaseModel):
    """The declarative configuration for SelectorGroupChat."""

    name: str | None = None
    description: str | None = None
    participants: List[ComponentModel]
    model_client: ComponentModel
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    selector_prompt: str
    allow_repeated_speaker: bool
    # selector_func: ComponentModel | None
    max_selector_attempts: int = 3
    emit_team_events: bool = False
    model_client_streaming: bool = False
    model_context: ComponentModel | None = None

# From _group_chat/_selector_group_chat.py
class SelectorGroupChat(BaseGroupChat, Component[SelectorGroupChatConfig]):
    """A group chat team that have participants takes turn to publish a message
    to all, using a ChatCompletion model to select the next speaker after each message.

    If an :class:`~autogen_agentchat.base.ChatAgent` is a participant,
    the :class:`~autogen_agentchat.messages.BaseChatMessage` from the agent response's
    :attr:`~autogen_agentchat.base.Response.chat_message` will be published
    to other participants in the group chat.

    If a :class:`~autogen_agentchat.base.Team` is a participant,
    the :class:`~autogen_agentchat.messages.BaseChatMessage`
    from the team result' :attr:`~autogen_agentchat.base.TaskResult.messages` will be published
    to other participants in the group chat.

    Args:
        participants (List[ChatAgent | Team]): The participants in the group chat,
            must have unique names and at least two participants.
        model_client (ChatCompletionClient): The ChatCompletion model client used
            to select the next speaker.
        name (str | None, optional): The name of the group chat, using
            :attr:`~autogen_agentchat.teams.SelectorGroupChat.DEFAULT_NAME` if not provided.
            The name is used by a parent team to identify this group chat so it must
            be unique within the parent team.
        description (str | None, optional): The description of the group chat, using
            :attr:`~autogen_agentchat.teams.SelectorGroupChat.DEFAULT_DESCRIPTION` if not provided.
        termination_condition (TerminationCondition, optional): The termination condition for the group chat. Defaults to None.
            Without a termination condition, the group chat will run indefinitely.
        max_turns (int, optional): The maximum number of turns in the group chat before stopping. Defaults to None, meaning no limit.
        selector_prompt (str, optional): The prompt template to use for selecting the next speaker.
            Available fields: '{roles}', '{participants}', and '{history}'.
            `{participants}` is the names of candidates for selection. The format is `["<name1>", "<name2>", ...]`.
            `{roles}` is a newline-separated list of names and descriptions of the candidate agents. The format for each line is: `"<name> : <description>"`.
            `{history}` is the conversation history formatted as a double newline separated of names and message content. The format for each message is: `"<name> : <message content>"`.
        allow_repeated_speaker (bool, optional): Whether to include the previous speaker in the list of candidates to be selected for the next turn.
            Defaults to False. The model may still select the previous speaker -- a warning will be logged if this happens.
        max_selector_attempts (int, optional): The maximum number of attempts to select a speaker using the model. Defaults to 3.
            If the model fails to select a speaker after the maximum number of attempts, the previous speaker will be used if available,
            otherwise the first participant will be used.
        selector_func (Callable[[Sequence[BaseAgentEvent | BaseChatMessage]], str | None], Callable[[Sequence[BaseAgentEvent | BaseChatMessage]], Awaitable[str | None]], optional): A custom selector
            function that takes the conversation history and returns the name of the next speaker.
            If provided, this function will be used to override the model to select the next speaker.
            If the function returns None, the model will be used to select the next speaker.
            NOTE: `selector_func` is not serializable and will be ignored during serialization and deserialization process.
        candidate_func (Callable[[Sequence[BaseAgentEvent | BaseChatMessage]], List[str]], Callable[[Sequence[BaseAgentEvent | BaseChatMessage]], Awaitable[List[str]]], optional):
            A custom function that takes the conversation history and returns a filtered list of candidates for the next speaker
            selection using model. If the function returns an empty list or `None`, `SelectorGroupChat` will raise a `ValueError`.
            This function is only used if `selector_func` is not set. The `allow_repeated_speaker` will be ignored if set.
        custom_message_types (List[type[BaseAgentEvent | BaseChatMessage]], optional): A list of custom message types that will be used in the group chat.
            If you are using custom message types or your agents produces custom message types, you need to specify them here.
            Make sure your custom message types are subclasses of :class:`~autogen_agentchat.messages.BaseAgentEvent` or :class:`~autogen_agentchat.messages.BaseChatMessage`.
        emit_team_events (bool, optional): Whether to emit team events through :meth:`BaseGroupChat.run_stream`. Defaults to False.
        model_client_streaming (bool, optional): Whether to use streaming for the model client. (This is useful for reasoning models like QwQ). Defaults to False.
        model_context (ChatCompletionContext | None, optional): The model context for storing and retrieving
            :class:`~autogen_core.models.LLMMessage`. It can be preloaded with initial messages. Messages stored in model context will be used for speaker selection. The initial messages will be cleared when the team is reset.

    Raises:
        ValueError: If the number of participants is less than two or if the selector prompt is invalid.

    Examples:

    A team with multiple participants:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import SelectorGroupChat
            from autogen_agentchat.conditions import TextMentionTermination
            from autogen_agentchat.ui import Console


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                async def lookup_hotel(location: str) -> str:
                    return f"Here are some hotels in {location}: hotel1, hotel2, hotel3."

                async def lookup_flight(origin: str, destination: str) -> str:
                    return f"Here are some flights from {origin} to {destination}: flight1, flight2, flight3."

                async def book_trip() -> str:
                    return "Your trip is booked!"

                travel_advisor = AssistantAgent(
                    "Travel_Advisor",
                    model_client,
                    tools=[book_trip],
                    description="Helps with travel planning.",
                )
                hotel_agent = AssistantAgent(
                    "Hotel_Agent",
                    model_client,
                    tools=[lookup_hotel],
                    description="Helps with hotel booking.",
                )
                flight_agent = AssistantAgent(
                    "Flight_Agent",
                    model_client,
                    tools=[lookup_flight],
                    description="Helps with flight booking.",
                )
                termination = TextMentionTermination("TERMINATE")
                team = SelectorGroupChat(
                    [travel_advisor, hotel_agent, flight_agent],
                    model_client=model_client,
                    termination_condition=termination,
                )
                await Console(team.run_stream(task="Book a 3-day trip to new york."))


            asyncio.run(main())

    A team with a custom selector function:

        .. code-block:: python

            import asyncio
            from typing import Sequence
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import SelectorGroupChat
            from autogen_agentchat.conditions import TextMentionTermination
            from autogen_agentchat.ui import Console
            from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                def check_calculation(x: int, y: int, answer: int) -> str:
                    if x + y == answer:
                        return "Correct!"
                    else:
                        return "Incorrect!"

                agent1 = AssistantAgent(
                    "Agent1",
                    model_client,
                    description="For calculation",
                    system_message="Calculate the sum of two numbers",
                )
                agent2 = AssistantAgent(
                    "Agent2",
                    model_client,
                    tools=[check_calculation],
                    description="For checking calculation",
                    system_message="Check the answer and respond with 'Correct!' or 'Incorrect!'",
                )

                def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
                    if len(messages) == 1 or messages[-1].to_text() == "Incorrect!":
                        return "Agent1"
                    if messages[-1].source == "Agent1":
                        return "Agent2"
                    return None

                termination = TextMentionTermination("Correct!")
                team = SelectorGroupChat(
                    [agent1, agent2],
                    model_client=model_client,
                    selector_func=selector_func,
                    termination_condition=termination,
                )

                await Console(team.run_stream(task="What is 1 + 1?"))


            asyncio.run(main())

    A team with custom model context:

        .. code-block:: python

            import asyncio

            from autogen_core.model_context import BufferedChatCompletionContext
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import TextMentionTermination
            from autogen_agentchat.teams import SelectorGroupChat
            from autogen_agentchat.ui import Console


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")
                model_context = BufferedChatCompletionContext(buffer_size=5)

                async def lookup_hotel(location: str) -> str:
                    return f"Here are some hotels in {location}: hotel1, hotel2, hotel3."

                async def lookup_flight(origin: str, destination: str) -> str:
                    return f"Here are some flights from {origin} to {destination}: flight1, flight2, flight3."

                async def book_trip() -> str:
                    return "Your trip is booked!"

                travel_advisor = AssistantAgent(
                    "Travel_Advisor",
                    model_client,
                    tools=[book_trip],
                    description="Helps with travel planning.",
                )
                hotel_agent = AssistantAgent(
                    "Hotel_Agent",
                    model_client,
                    tools=[lookup_hotel],
                    description="Helps with hotel booking.",
                )
                flight_agent = AssistantAgent(
                    "Flight_Agent",
                    model_client,
                    tools=[lookup_flight],
                    description="Helps with flight booking.",
                )
                termination = TextMentionTermination("TERMINATE")
                team = SelectorGroupChat(
                    [travel_advisor, hotel_agent, flight_agent],
                    model_client=model_client,
                    termination_condition=termination,
                    model_context=model_context,
                )
                await Console(team.run_stream(task="Book a 3-day trip to new york."))


            asyncio.run(main())
    """

    component_config_schema = SelectorGroupChatConfig
    component_provider_override = "autogen_agentchat.teams.SelectorGroupChat"

    DEFAULT_NAME = "SelectorGroupChat"
    DEFAULT_DESCRIPTION = "A team of agents."

    def __init__(
        self,
        participants: List[ChatAgent | Team],
        model_client: ChatCompletionClient,
        *,
        name: str | None = None,
        description: str | None = None,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        selector_prompt: str = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
""",
        allow_repeated_speaker: bool = False,
        max_selector_attempts: int = 3,
        selector_func: Optional[SelectorFuncType] = None,
        candidate_func: Optional[CandidateFuncType] = None,
        custom_message_types: List[type[BaseAgentEvent | BaseChatMessage]] | None = None,
        emit_team_events: bool = False,
        model_client_streaming: bool = False,
        model_context: ChatCompletionContext | None = None,
    ):
        super().__init__(
            name=name or self.DEFAULT_NAME,
            description=description or self.DEFAULT_DESCRIPTION,
            participants=participants,
            group_chat_manager_name="SelectorGroupChatManager",
            group_chat_manager_class=SelectorGroupChatManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types,
            emit_team_events=emit_team_events,
        )
        # Validate the participants.
        if len(participants) < 2:
            raise ValueError("At least two participants are required for SelectorGroupChat.")
        self._selector_prompt = selector_prompt
        self._model_client = model_client
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func
        self._max_selector_attempts = max_selector_attempts
        self._candidate_func = candidate_func
        self._model_client_streaming = model_client_streaming
        self._model_context = model_context

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
    ) -> Callable[[], BaseGroupChatManager]:
        return lambda: SelectorGroupChatManager(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
            message_factory,
            self._model_client,
            self._selector_prompt,
            self._allow_repeated_speaker,
            self._selector_func,
            self._max_selector_attempts,
            self._candidate_func,
            self._emit_team_events,
            self._model_context,
            self._model_client_streaming,
        )

    def _to_config(self) -> SelectorGroupChatConfig:
        return SelectorGroupChatConfig(
            name=self._name,
            description=self._description,
            participants=[participant.dump_component() for participant in self._participants],
            model_client=self._model_client.dump_component(),
            termination_condition=self._termination_condition.dump_component() if self._termination_condition else None,
            max_turns=self._max_turns,
            selector_prompt=self._selector_prompt,
            allow_repeated_speaker=self._allow_repeated_speaker,
            max_selector_attempts=self._max_selector_attempts,
            # selector_func=self._selector_func.dump_component() if self._selector_func else None,
            emit_team_events=self._emit_team_events,
            model_client_streaming=self._model_client_streaming,
            model_context=self._model_context.dump_component() if self._model_context else None,
        )

    @classmethod
    def _from_config(cls, config: SelectorGroupChatConfig) -> Self:
        participants: List[ChatAgent | Team] = []
        for participant in config.participants:
            if participant.component_type == ChatAgent.component_type:
                participants.append(ChatAgent.load_component(participant))
            elif participant.component_type == Team.component_type:
                participants.append(Team.load_component(participant))
            else:
                raise ValueError(
                    f"Invalid participant component type: {participant.component_type}. " "Expected ChatAgent or Team."
                )
        return cls(
            participants=participants,
            model_client=ChatCompletionClient.load_component(config.model_client),
            name=config.name,
            description=config.description,
            termination_condition=TerminationCondition.load_component(config.termination_condition)
            if config.termination_condition
            else None,
            max_turns=config.max_turns,
            selector_prompt=config.selector_prompt,
            allow_repeated_speaker=config.allow_repeated_speaker,
            max_selector_attempts=config.max_selector_attempts,
            # selector_func=ComponentLoader.load_component(config.selector_func, Callable[[Sequence[BaseAgentEvent | BaseChatMessage]], str | None])
            # if config.selector_func
            # else None,
            emit_team_events=config.emit_team_events,
            model_client_streaming=config.model_client_streaming,
            model_context=ChatCompletionContext.load_component(config.model_context) if config.model_context else None,
        )

# From _group_chat/_selector_group_chat.py
def construct_message_history(self, message_history: List[LLMMessage]) -> str:
        # Construct the history of the conversation.
        history_messages: List[str] = []
        for msg in message_history:
            if isinstance(msg, UserMessage) or isinstance(msg, AssistantMessage):
                message = f"{msg.source}: {msg.content}"
                history_messages.append(
                    message.rstrip() + "\n\n"
                )  # Create some consistency for how messages are separated in the transcript

        history: str = "\n".join(history_messages)
        return history

from autogen_core import MessageContext
from autogen_core import event
from autogen_core import rpc
from autogen_agentchat.messages import MessageFactory
from state import ChatAgentContainerState
from _events import GroupChatAgentResponse
from _events import GroupChatError
from _events import GroupChatMessage
from _events import GroupChatPause
from _events import GroupChatRequestPublish
from _events import GroupChatReset
from _events import GroupChatResume
from _events import GroupChatStart
from _events import GroupChatTeamResponse
from _events import SerializableException
from _sequential_routed_agent import SequentialRoutedAgent

# From _group_chat/_chat_agent_container.py
class ChatAgentContainer(SequentialRoutedAgent):
    """A core agent class that delegates message handling to an
    :class:`autogen_agentchat.base.ChatAgent` or :class:`autogen_agentchat.base.Team`
    so that it can be used in a group chat team.

    Args:
        parent_topic_type (str): The topic type of the parent orchestrator.
        output_topic_type (str): The topic type for the output.
        agent (ChatAgent | Team): The agent or team to delegate message handling to.
        message_factory (MessageFactory): The message factory to use for
            creating messages from JSON data.
    """

    def __init__(
        self, parent_topic_type: str, output_topic_type: str, agent: ChatAgent | Team, message_factory: MessageFactory
    ) -> None:
        super().__init__(
            description=agent.description,
            sequential_message_types=[
                GroupChatStart,
                GroupChatRequestPublish,
                GroupChatReset,
                GroupChatAgentResponse,
                GroupChatTeamResponse,
            ],
        )
        self._parent_topic_type = parent_topic_type
        self._output_topic_type = output_topic_type
        self._agent = agent
        self._message_buffer: List[BaseChatMessage] = []
        self._message_factory = message_factory

    @event
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:
        """Handle a start event by appending the content to the buffer."""
        if message.messages is not None:
            for msg in message.messages:
                self._buffer_message(msg)

    @event
    async def handle_agent_response(self, message: GroupChatAgentResponse, ctx: MessageContext) -> None:
        """Handle an agent response event by appending the content to the buffer."""
        self._buffer_message(message.response.chat_message)

    @event
    async def handle_team_response(self, message: GroupChatTeamResponse, ctx: MessageContext) -> None:
        """Handle a team response event by appending the content to the buffer."""
        for msg in message.result.messages:
            if isinstance(msg, BaseChatMessage):
                self._buffer_message(msg)

    @rpc
    async def handle_reset(self, message: GroupChatReset, ctx: MessageContext) -> None:
        """Handle a reset event by resetting the agent."""
        self._message_buffer.clear()
        if isinstance(self._agent, Team):
            # If the agent is a team, reset the team.
            await self._agent.reset()
        else:
            await self._agent.on_reset(ctx.cancellation_token)

    @event
    async def handle_request(self, message: GroupChatRequestPublish, ctx: MessageContext) -> None:
        """Handle a content request event by passing the messages in the buffer
        to the delegate agent and publish the response."""
        if isinstance(self._agent, Team):
            try:
                stream = self._agent.run_stream(
                    task=self._message_buffer,
                    cancellation_token=ctx.cancellation_token,
                    output_task_messages=False,
                )
                result: TaskResult | None = None
                async for team_event in stream:
                    if isinstance(team_event, TaskResult):
                        result = team_event
                    else:
                        await self._log_message(team_event)
                if result is None:
                    raise RuntimeError(
                        "The team did not produce a final TaskResult. Check the team's run_stream method."
                    )
                self._message_buffer.clear()
                # Publish the team response to the group chat.
                await self.publish_message(
                    GroupChatTeamResponse(result=result, name=self._agent.name),
                    topic_id=DefaultTopicId(type=self._parent_topic_type),
                    cancellation_token=ctx.cancellation_token,
                )
            except Exception as e:
                # Publish the error to the group chat.
                error_message = SerializableException.from_exception(e)
                await self.publish_message(
                    GroupChatError(error=error_message),
                    topic_id=DefaultTopicId(type=self._parent_topic_type),
                    cancellation_token=ctx.cancellation_token,
                )
                # Raise the error to the runtime.
                raise
        else:
            # If the agent is not a team, handle it as a single agent.
            with trace_invoke_agent_span(
                agent_name=self._agent.name,
                agent_description=self._agent.description,
                agent_id=str(self.id),
            ):
                try:
                    # Pass the messages in the buffer to the delegate agent.
                    response: Response | None = None
                    async for msg in self._agent.on_messages_stream(self._message_buffer, ctx.cancellation_token):
                        if isinstance(msg, Response):
                            await self._log_message(msg.chat_message)
                            response = msg
                        else:
                            await self._log_message(msg)
                    if response is None:
                        raise RuntimeError(
                            "The agent did not produce a final response. Check the agent's on_messages_stream method."
                        )
                    # Publish the response to the group chat.
                    self._message_buffer.clear()
                    await self.publish_message(
                        GroupChatAgentResponse(response=response, name=self._agent.name),
                        topic_id=DefaultTopicId(type=self._parent_topic_type),
                        cancellation_token=ctx.cancellation_token,
                    )
                except Exception as e:
                    # Publish the error to the group chat.
                    error_message = SerializableException.from_exception(e)
                    await self.publish_message(
                        GroupChatError(error=error_message),
                        topic_id=DefaultTopicId(type=self._parent_topic_type),
                        cancellation_token=ctx.cancellation_token,
                    )
                    # Raise the error to the runtime.
                    raise

    def _buffer_message(self, message: BaseChatMessage) -> None:
        if not self._message_factory.is_registered(message.__class__):
            raise ValueError(f"Message type {message.__class__} is not registered.")
        # Buffer the message.
        self._message_buffer.append(message)

    async def _log_message(self, message: BaseAgentEvent | BaseChatMessage) -> None:
        if not self._message_factory.is_registered(message.__class__):
            raise ValueError(f"Message type {message.__class__} is not registered.")
        # Log the message.
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )

    @rpc
    async def handle_pause(self, message: GroupChatPause, ctx: MessageContext) -> None:
        """Handle a pause event by pausing the agent."""
        if isinstance(self._agent, Team):
            # If the agent is a team, pause the team.
            await self._agent.pause()
        else:
            await self._agent.on_pause(ctx.cancellation_token)

    @rpc
    async def handle_resume(self, message: GroupChatResume, ctx: MessageContext) -> None:
        """Handle a resume event by resuming the agent."""
        if isinstance(self._agent, Team):
            # If the agent is a team, resume the team.
            await self._agent.resume()
        else:
            await self._agent.on_resume(ctx.cancellation_token)

    async def on_unhandled_message(self, message: Any, ctx: MessageContext) -> None:
        raise ValueError(f"Unhandled message in agent container: {type(message)}")

    async def save_state(self) -> Mapping[str, Any]:
        agent_state = await self._agent.save_state()
        state = ChatAgentContainerState(
            agent_state=agent_state, message_buffer=[message.dump() for message in self._message_buffer]
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        container_state = ChatAgentContainerState.model_validate(state)
        self._message_buffer = []
        for message_data in container_state.message_buffer:
            message = self._message_factory.create(message_data)
            if isinstance(message, BaseChatMessage):
                self._message_buffer.append(message)
            else:
                raise ValueError(f"Invalid message type in message buffer: {type(message)}")
        await self._agent.load_state(container_state.agent_state)

from autogen_core import RoutedAgent

# From _group_chat/_sequential_routed_agent.py
class FIFOLock:
    """A lock that ensures coroutines acquire the lock in the order they request it."""

    def __init__(self) -> None:
        self._queue = asyncio.Queue[asyncio.Event]()
        self._locked = False

    async def acquire(self) -> None:
        # If the lock is not held by any coroutine, set the lock to be held
        # by the current coroutine.
        if not self._locked:
            self._locked = True
            return

        # If the lock is held by another coroutine, create an event and put it
        # in the queue. Wait for the event to be set.
        event = asyncio.Event()
        await self._queue.put(event)
        await event.wait()

    def release(self) -> None:
        if not self._queue.empty():
            # If there are events in the queue, get the next event and set it.
            next_event = self._queue.get_nowait()
            next_event.set()
        else:
            # If there are no events in the queue, release the lock.
            self._locked = False

# From _group_chat/_sequential_routed_agent.py
class SequentialRoutedAgent(RoutedAgent):
    """A subclass of :class:`autogen_core.RoutedAgent` that ensures
    that messages of certain types are processed sequentially
    using a FIFO lock.

    This is useful for agents that need to maintain a strict order of
    processing messages, such as in a group chat scenario.



    Args:

        description (str): The description of the agent.
        sequential_message_types (Sequence[Type[Any]]): A sequence of message types that should be
            processed sequentially. If a message of one of these types is received,
            the agent will acquire a FIFO lock to ensure that it is processed
            before any later messages that are also one of these types.
    """

    def __init__(self, description: str, sequential_message_types: Sequence[type[Any]]) -> None:
        super().__init__(description=description)
        self._fifo_lock = FIFOLock()
        self._sequential_message_types = sequential_message_types

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any | None:
        if any(isinstance(message, sequential_type) for sequential_type in self._sequential_message_types):
            # Acquire the FIFO lock to ensure that this message is processed
            # in the order it was received.
            await self._fifo_lock.acquire()
            try:
                return await super().on_message_impl(message, ctx)
            finally:
                # Release the FIFO lock to allow the next message to be processed.
                self._fifo_lock.release()
        # If the message is not of a sequential type, process it normally.
        return await super().on_message_impl(message, ctx)

# From _group_chat/_sequential_routed_agent.py
def release(self) -> None:
        if not self._queue.empty():
            # If there are events in the queue, get the next event and set it.
            next_event = self._queue.get_nowait()
            next_event.set()
        else:
            # If there are no events in the queue, release the lock.
            self._locked = False

from _magentic_one_orchestrator import MagenticOneOrchestrator
from _prompts import ORCHESTRATOR_FINAL_ANSWER_PROMPT

# From _magentic_one/_magentic_one_group_chat.py
class MagenticOneGroupChatConfig(BaseModel):
    """The declarative configuration for a MagenticOneGroupChat."""

    name: str | None = None
    description: str | None = None
    participants: List[ComponentModel]
    model_client: ComponentModel
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    max_stalls: int
    final_answer_prompt: str
    emit_team_events: bool = False

# From _magentic_one/_magentic_one_group_chat.py
class MagenticOneGroupChat(BaseGroupChat, Component[MagenticOneGroupChatConfig]):
    """A team that runs a group chat with participants managed by the MagenticOneOrchestrator.

    The orchestrator handles the conversation flow, ensuring that the task is completed
    efficiently by managing the participants' interactions.

    The orchestrator is based on the Magentic-One architecture, which is a generalist multi-agent system for solving complex tasks (see references below).

    Unlike :class:`~autogen_agentchat.teams.RoundRobinGroupChat` and :class:`~autogen_agentchat.teams.SelectorGroupChat`,
    the MagenticOneGroupChat does not support using team as participant.

    Args:
        participants (List[ChatAgent]): The participants in the group chat.
        model_client (ChatCompletionClient): The model client used for generating responses.
        termination_condition (TerminationCondition, optional): The termination condition for the group chat. Defaults to None.
            Without a termination condition, the group chat will run based on the orchestrator logic or until the maximum number of turns is reached.
        max_turns (int, optional): The maximum number of turns in the group chat before stopping. Defaults to 20.
        max_stalls (int, optional): The maximum number of stalls allowed before re-planning. Defaults to 3.
        final_answer_prompt (str, optional): The LLM prompt used to generate the final answer or response from the team's transcript. A default (sensible for GPT-4o class models) is provided.
        custom_message_types (List[type[BaseAgentEvent | BaseChatMessage]], optional): A list of custom message types that will be used in the group chat.
            If you are using custom message types or your agents produces custom message types, you need to specify them here.
            Make sure your custom message types are subclasses of :class:`~autogen_agentchat.messages.BaseAgentEvent` or :class:`~autogen_agentchat.messages.BaseChatMessage`.
        emit_team_events (bool, optional): Whether to emit team events through :meth:`BaseGroupChat.run_stream`. Defaults to False.

    Raises:
        ValueError: In orchestration logic if progress ledger does not have required keys or if next speaker is not valid.

    Examples:

    MagenticOneGroupChat with one assistant agent:

        .. code-block:: python

            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import MagenticOneGroupChat
            from autogen_agentchat.ui import Console


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                assistant = AssistantAgent(
                    "Assistant",
                    model_client=model_client,
                )
                team = MagenticOneGroupChat([assistant], model_client=model_client)
                await Console(team.run_stream(task="Provide a different proof to Fermat last theorem"))


            asyncio.run(main())

    References:

        If you use the MagenticOneGroupChat in your work, please cite the following paper:

        .. code-block:: bibtex

            @article{fourney2024magentic,
                title={Magentic-one: A generalist multi-agent system for solving complex tasks},
                author={Fourney, Adam and Bansal, Gagan and Mozannar, Hussein and Tan, Cheng and Salinas, Eduardo and Niedtner, Friederike and Proebsting, Grace and Bassman, Griffin and Gerrits, Jack and Alber, Jacob and others},
                journal={arXiv preprint arXiv:2411.04468},
                year={2024}
            }
    """

    component_config_schema = MagenticOneGroupChatConfig
    component_provider_override = "autogen_agentchat.teams.MagenticOneGroupChat"

    DEFAULT_NAME = "MagenticOneGroupChat"
    DEFAULT_DESCRIPTION = "A team of agents."

    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        name: str | None = None,
        description: str | None = None,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = 20,
        runtime: AgentRuntime | None = None,
        max_stalls: int = 3,
        final_answer_prompt: str = ORCHESTRATOR_FINAL_ANSWER_PROMPT,
        custom_message_types: List[type[BaseAgentEvent | BaseChatMessage]] | None = None,
        emit_team_events: bool = False,
    ):
        for participant in participants:
            if not isinstance(participant, ChatAgent):
                raise TypeError(f"Participant {participant} must be a ChatAgent.")
        super().__init__(
            name=name or self.DEFAULT_NAME,
            description=description or self.DEFAULT_DESCRIPTION,
            participants=list(participants),
            group_chat_manager_name="MagenticOneOrchestrator",
            group_chat_manager_class=MagenticOneOrchestrator,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types,
            emit_team_events=emit_team_events,
        )

        # Validate the participants.
        if len(participants) == 0:
            raise ValueError("At least one participant is required for MagenticOneGroupChat.")
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt

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
    ) -> Callable[[], MagenticOneOrchestrator]:
        return lambda: MagenticOneOrchestrator(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            max_turns,
            message_factory,
            self._model_client,
            self._max_stalls,
            self._final_answer_prompt,
            output_message_queue,
            termination_condition,
            self._emit_team_events,
        )

    def _to_config(self) -> MagenticOneGroupChatConfig:
        participants = [participant.dump_component() for participant in self._participants]
        termination_condition = self._termination_condition.dump_component() if self._termination_condition else None
        return MagenticOneGroupChatConfig(
            name=self.name,
            description=self.description,
            participants=participants,
            model_client=self._model_client.dump_component(),
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            max_stalls=self._max_stalls,
            final_answer_prompt=self._final_answer_prompt,
            emit_team_events=self._emit_team_events,
        )

    @classmethod
    def _from_config(cls, config: MagenticOneGroupChatConfig) -> Self:
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        model_client = ChatCompletionClient.load_component(config.model_client)
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )
        return cls(
            participants=participants,
            name=config.name,
            description=config.description,
            model_client=model_client,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
            max_stalls=config.max_stalls,
            final_answer_prompt=config.final_answer_prompt,
            emit_team_events=config.emit_team_events,
        )

from autogen_core.utils import extract_json_from_str
from messages import MultiModalMessage
from messages import SelectSpeakerEvent
from state import MagenticOneOrchestratorState
from _prompts import ORCHESTRATOR_PROGRESS_LEDGER_PROMPT
from _prompts import ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT
from _prompts import ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT
from _prompts import ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT
from _prompts import ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT
from _prompts import ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT
from _prompts import LedgerEntry

# From _magentic_one/_magentic_one_orchestrator.py
class MagenticOneOrchestrator(BaseGroupChatManager):
    """The MagenticOneOrchestrator manages a group chat with ledger based orchestration."""

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        max_turns: int | None,
        message_factory: MessageFactory,
        model_client: ChatCompletionClient,
        max_stalls: int,
        final_answer_prompt: str,
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        emit_team_events: bool,
    ):
        super().__init__(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
            message_factory,
            emit_team_events=emit_team_events,
        )
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt
        self._max_json_retries = 10
        self._task = ""
        self._facts = ""
        self._plan = ""
        self._n_rounds = 0
        self._n_stalls = 0

        # Produce a team description. Each agent sould appear on a single line.
        self._team_description = ""
        for topic_type, description in zip(self._participant_names, self._participant_descriptions, strict=True):
            self._team_description += re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
        self._team_description = self._team_description.strip()

    def _get_task_ledger_facts_prompt(self, task: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT.format(task=task)

    def _get_task_ledger_plan_prompt(self, team: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT.format(team=team)

    def _get_task_ledger_full_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT.format(task=task, team=team, facts=facts, plan=plan)

    def _get_progress_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        return ORCHESTRATOR_PROGRESS_LEDGER_PROMPT.format(task=task, team=team, names=", ".join(names))

    def _get_task_ledger_facts_update_prompt(self, task: str, facts: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT.format(task=task, facts=facts)

    def _get_task_ledger_plan_update_prompt(self, team: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT.format(team=team)

    def _get_final_answer_prompt(self, task: str) -> str:
        if self._final_answer_prompt == ORCHESTRATOR_FINAL_ANSWER_PROMPT:
            return ORCHESTRATOR_FINAL_ANSWER_PROMPT.format(task=task)
        else:
            return self._final_answer_prompt

    async def _log_message(self, log_message: str) -> None:
        trace_logger.debug(log_message)

    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:  # type: ignore
        """Handle the start of a task."""

        # Check if the conversation has already terminated.
        if self._termination_condition is not None and self._termination_condition.terminated:
            early_stop_message = StopMessage(content="The group chat has already terminated.", source=self._name)
            # Signal termination.
            await self._signal_termination(early_stop_message)
            # Stop the group chat.
            return
        assert message is not None and message.messages is not None

        # Validate the group state given all the messages.
        await self.validate_group_state(message.messages)

        # Log the message to the output topic.
        await self.publish_message(message, topic_id=DefaultTopicId(type=self._output_topic_type))
        # Log the message to the output queue.
        for msg in message.messages:
            await self._output_message_queue.put(msg)

        # Outer Loop for first time
        # Create the initial task ledger
        #################################
        # Combine all message contents for task
        self._task = " ".join([msg.to_model_text() for msg in message.messages])
        planning_conversation: List[LLMMessage] = []

        # 1. GATHER FACTS
        # create a closed book task and generate a response and update the chat history
        planning_conversation.append(
            UserMessage(content=self._get_task_ledger_facts_prompt(self._task), source=self._name)
        )
        response = await self._model_client.create(
            self._get_compatible_context(planning_conversation), cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        self._facts = response.content
        planning_conversation.append(AssistantMessage(content=self._facts, source=self._name))

        # 2. CREATE A PLAN
        ## plan based on available information
        planning_conversation.append(
            UserMessage(content=self._get_task_ledger_plan_prompt(self._team_description), source=self._name)
        )
        response = await self._model_client.create(
            self._get_compatible_context(planning_conversation), cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        self._plan = response.content

        # Kick things off
        self._n_stalls = 0
        await self._reenter_outer_loop(ctx.cancellation_token)

    @event
    async def handle_agent_response(  # type: ignore
        self, message: GroupChatAgentResponse | GroupChatTeamResponse, ctx: MessageContext
    ) -> None:  # type: ignore
        try:
            if not isinstance(message, GroupChatAgentResponse):
                raise RuntimeError("MagenticOneOrchestrator does not support GroupChatTeamResponse messages.")
            delta: List[BaseAgentEvent | BaseChatMessage] = []
            if message.response.inner_messages is not None:
                for inner_message in message.response.inner_messages:
                    delta.append(inner_message)
            await self.update_message_thread([message.response.chat_message])
            delta.append(message.response.chat_message)

            if self._termination_condition is not None:
                stop_message = await self._termination_condition(delta)
                if stop_message is not None:
                    # Reset the termination conditions.
                    await self._termination_condition.reset()
                    # Signal termination.
                    await self._signal_termination(stop_message)
                    return

            await self._orchestrate_step(ctx.cancellation_token)
        except Exception as e:
            error = SerializableException.from_exception(e)
            await self._signal_termination_with_error(error)
            # Raise the error to the runtime.
            raise

    async def validate_group_state(self, messages: List[BaseChatMessage] | None) -> None:
        pass

    async def save_state(self) -> Mapping[str, Any]:
        state = MagenticOneOrchestratorState(
            message_thread=[msg.dump() for msg in self._message_thread],
            current_turn=self._current_turn,
            task=self._task,
            facts=self._facts,
            plan=self._plan,
            n_rounds=self._n_rounds,
            n_stalls=self._n_stalls,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = MagenticOneOrchestratorState.model_validate(state)
        self._message_thread = [self._message_factory.create(message) for message in orchestrator_state.message_thread]
        self._current_turn = orchestrator_state.current_turn
        self._task = orchestrator_state.task
        self._facts = orchestrator_state.facts
        self._plan = orchestrator_state.plan
        self._n_rounds = orchestrator_state.n_rounds
        self._n_stalls = orchestrator_state.n_stalls

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str] | str:
        """Not used in this orchestrator, we select next speaker in _orchestrate_step."""
        return [""]

    async def reset(self) -> None:
        """Reset the group chat manager."""
        self._message_thread.clear()
        if self._termination_condition is not None:
            await self._termination_condition.reset()
        self._n_rounds = 0
        self._n_stalls = 0
        self._task = ""
        self._facts = ""
        self._plan = ""

    async def _reenter_outer_loop(self, cancellation_token: CancellationToken) -> None:
        """Re-enter Outer loop of the orchestrator after creating task ledger."""
        # Reset the agents
        for participant_topic_type in self._participant_name_to_topic_type.values():
            await self._runtime.send_message(
                GroupChatReset(),
                recipient=AgentId(type=participant_topic_type, key=self.id.key),
                cancellation_token=cancellation_token,
            )
        # Reset partially the group chat manager
        self._message_thread.clear()

        # Prepare the ledger
        ledger_message = TextMessage(
            content=self._get_task_ledger_full_prompt(self._task, self._team_description, self._facts, self._plan),
            source=self._name,
        )

        # Save my copy
        await self.update_message_thread([ledger_message])

        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=ledger_message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(ledger_message)

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=ledger_message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
        )

        # Restart the inner loop
        await self._orchestrate_step(cancellation_token=cancellation_token)

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        """Implements the inner loop of the orchestrator and selects next speaker."""
        # Check if we reached the maximum number of rounds
        if self._max_turns is not None and self._n_rounds > self._max_turns:
            await self._prepare_final_answer("Max rounds reached.", cancellation_token)
            return
        self._n_rounds += 1

        # Update the progress ledger
        context = self._thread_to_context()

        progress_ledger_prompt = self._get_progress_ledger_prompt(
            self._task, self._team_description, self._participant_names
        )
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        progress_ledger: Dict[str, Any] = {}
        assert self._max_json_retries > 0
        key_error: bool = False
        for _ in range(self._max_json_retries):
            if self._model_client.model_info.get("structured_output", False):
                response = await self._model_client.create(
                    self._get_compatible_context(context), json_output=LedgerEntry
                )
            elif self._model_client.model_info.get("json_output", False):
                response = await self._model_client.create(
                    self._get_compatible_context(context), cancellation_token=cancellation_token, json_output=True
                )
            else:
                response = await self._model_client.create(
                    self._get_compatible_context(context), cancellation_token=cancellation_token
                )
            ledger_str = response.content
            try:
                assert isinstance(ledger_str, str)
                output_json = extract_json_from_str(ledger_str)
                if len(output_json) != 1:
                    raise ValueError(
                        f"Progress ledger should contain a single JSON object, but found: {len(progress_ledger)}"
                    )
                progress_ledger = output_json[0]

                # If the team consists of a single agent, deterministically set the next speaker
                if len(self._participant_names) == 1:
                    progress_ledger["next_speaker"] = {
                        "reason": "The team consists of only one agent.",
                        "answer": self._participant_names[0],
                    }

                # Validate the structure
                required_keys = [
                    "is_request_satisfied",
                    "is_progress_being_made",
                    "is_in_loop",
                    "instruction_or_question",
                    "next_speaker",
                ]

                key_error = False
                for key in required_keys:
                    if (
                        key not in progress_ledger
                        or not isinstance(progress_ledger[key], dict)
                        or "answer" not in progress_ledger[key]
                        or "reason" not in progress_ledger[key]
                    ):
                        key_error = True
                        break

                # Validate the next speaker if the task is not yet complete
                if (
                    not progress_ledger["is_request_satisfied"]["answer"]
                    and progress_ledger["next_speaker"]["answer"] not in self._participant_names
                ):
                    key_error = True
                    break

                if not key_error:
                    break
                await self._log_message(f"Failed to parse ledger information, retrying: {ledger_str}")
            except (json.JSONDecodeError, TypeError):
                key_error = True
                await self._log_message("Invalid ledger format encountered, retrying...")
                continue
        if key_error:
            raise ValueError("Failed to parse ledger information after multiple retries.")
        await self._log_message(f"Progress Ledger: {progress_ledger}")

        # Check for task completion
        if progress_ledger["is_request_satisfied"]["answer"]:
            await self._log_message("Task completed, preparing final answer...")
            await self._prepare_final_answer(progress_ledger["is_request_satisfied"]["reason"], cancellation_token)
            return

        # Check for stalling
        if not progress_ledger["is_progress_being_made"]["answer"]:
            self._n_stalls += 1
        elif progress_ledger["is_in_loop"]["answer"]:
            self._n_stalls += 1
        else:
            self._n_stalls = max(0, self._n_stalls - 1)

        # Too much stalling
        if self._n_stalls >= self._max_stalls:
            await self._log_message("Stall count exceeded, re-planning with the outer loop...")
            await self._update_task_ledger(cancellation_token)
            await self._reenter_outer_loop(cancellation_token)
            return

        # Broadcast the next step
        message = TextMessage(content=progress_ledger["instruction_or_question"]["answer"], source=self._name)
        await self.update_message_thread([message])  # My copy

        await self._log_message(f"Next Speaker: {progress_ledger['next_speaker']['answer']}")
        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(message)

        # Broadcast it
        await self.publish_message(  # Broadcast
            GroupChatAgentResponse(response=Response(chat_message=message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Request that the step be completed
        next_speaker = progress_ledger["next_speaker"]["answer"]
        # Check if the next speaker is valid
        if next_speaker not in self._participant_name_to_topic_type:
            raise ValueError(
                f"Invalid next speaker: {next_speaker} from the ledger, participants are: {self._participant_names}"
            )
        participant_topic_type = self._participant_name_to_topic_type[next_speaker]
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

        # Send the message to the next speaker
        if self._emit_team_events:
            select_msg = SelectSpeakerEvent(content=[next_speaker], source=self._name)
            await self.publish_message(
                GroupChatMessage(message=select_msg),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            await self._output_message_queue.put(select_msg)

    async def _update_task_ledger(self, cancellation_token: CancellationToken) -> None:
        """Update the task ledger (outer loop) with the latest facts and plan."""
        context = self._thread_to_context()

        # Update the facts
        update_facts_prompt = self._get_task_ledger_facts_update_prompt(self._task, self._facts)
        context.append(UserMessage(content=update_facts_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._facts = response.content
        context.append(AssistantMessage(content=self._facts, source=self._name))

        # Update the plan
        update_plan_prompt = self._get_task_ledger_plan_update_prompt(self._team_description)
        context.append(UserMessage(content=update_plan_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._plan = response.content

    async def _prepare_final_answer(self, reason: str, cancellation_token: CancellationToken) -> None:
        """Prepare the final answer for the task."""
        context = self._thread_to_context()

        # Get the final answer
        final_answer_prompt = self._get_final_answer_prompt(self._task)
        context.append(UserMessage(content=final_answer_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )
        assert isinstance(response.content, str)
        message = TextMessage(content=response.content, source=self._name)

        await self.update_message_thread([message])  # My copy

        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(message)

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        if self._termination_condition is not None:
            await self._termination_condition.reset()
        # Signal termination
        await self._signal_termination(StopMessage(content=reason, source=self._name))

    def _thread_to_context(self) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in self._message_thread:
            if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                # Ignore tool call messages.
                continue
            elif isinstance(m, StopMessage | HandoffMessage):
                context.append(UserMessage(content=m.content, source=m.source))
            elif m.source == self._name:
                assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                context.append(AssistantMessage(content=m.content, source=m.source))
            else:
                assert isinstance(m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage))
                context.append(UserMessage(content=m.content, source=m.source))
        return context

    def _get_compatible_context(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if self._model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)

import glob
import pandas

# From benchmarks/process_logs.py
def process_logs(logs_path, single_benchmark=False):
    """
    logs_path: str, path to the logs directory, containing subdirectories for each benchmark subset
    returns: pandas DataFrame with all the logs processed
    """
    # check if logs_path exists
    if not os.path.exists(logs_path):
        raise FileNotFoundError(
            f"Path {logs_path} does not exist, need to download logs, extract them into one common folder"
        )
    if single_benchmark:
        # subset should be a list with single folder which is the last part of the path
        subsets = [logs_path.split("/")[-1]]
        logs_path = "/".join(logs_path.split("/")[:-1])

    else:
        subsets = os.listdir(logs_path)
    results = []
    for subset in subsets:
        # check if folder is not empty
        if not os.listdir(os.path.join(logs_path, subset)) or subset == ".DS_Store" or subset == "__MACOSX":
            continue
        benchmark_name = subset.split("_")[0]
        instances = [
            f
            for f in os.listdir(os.path.join(logs_path, subset))
            if os.path.isdir(os.path.join(logs_path, subset, f))
            and os.path.exists(os.path.join(logs_path, subset, f, "0"))
        ]
        logging.info(f"Processing {subset} with {len(instances)} instances")
        for instance in instances:
            instance_dir_path = os.path.join(logs_path, subset, instance, "0")
            try:
                correct, expected_answer, final_answer = scorer(instance_dir_path, benchmark_name)
            except Exception as e:
                logging.error(f"Error processing {instance_dir_path}: {e}")
                continue
            messages = get_message_logs(instance_dir_path)
            results.append(
                {
                    "benchmark": benchmark_name,
                    "subset_benchmark": subset,
                    "instance": instance,
                    "task_information": get_task_information(instance_dir_path, benchmark_name),
                    "expected_answer": expected_answer,
                    "final_answer": final_answer,
                    "correct": correct,
                    "stalled": did_agent_stall(instance_dir_path),
                    "num_messages": len(messages),
                    "messages": messages,
                    "progress_not_being_made": is_progress_not_being_made(instance_dir_path),
                }
            )
    df_logs = pd.DataFrame(results)
    return df_logs

# From benchmarks/process_logs.py
def normalize_answer(a):
    """
    Taken from custom_tabulate.py in the WebArena benchmark, given an answer, returns the normalized answer.
    Operations: lower case, trim, standardize comma separated values, replace multiple spaces with one space, remove trailing punctuation
    a: str, answer
    returns: str, normalized answer
    """
    norm_answer = ", ".join(a.strip().lower().split(","))
    norm_answer = re.sub(r"[\.\!\?]+$", "", re.sub(r"\s+", " ", norm_answer))
    return norm_answer

# From benchmarks/process_logs.py
def scorer(instance_dir, benchmark_name):
    """
    Returns results based on the benchmark name and the instance directory.

    benchmark_name: str, the name of the benchmark, either "gaia" or "webarena"
    instance_dir: str, path to the instance directory
    returns: tuple, (bool, str, str) or None, depending on the benchmark
    """

    if benchmark_name == "gaia" or benchmark_name == "assistant":
        # Read the expected answer
        expected_answer_file = os.path.join(instance_dir, "expected_answer.txt")
        if not os.path.isfile(expected_answer_file):
            return None

        with open(expected_answer_file, "rt") as fh:
            expected_answer = fh.read().strip()

        # Read the console log
        console_log_file = os.path.join(instance_dir, "console_log.txt")
        if not os.path.isfile(console_log_file):
            return None

        with open(console_log_file, "rt") as fh:
            console_log = fh.read()
            final_answer = None
            m = re.search(r"FINAL ANSWER:(.*?)\n", console_log, re.DOTALL)
            if m:
                final_answer = m.group(1).strip()

            if final_answer is None:
                return None
            not_normalized_final = final_answer

            n_ex = normalize_answer(expected_answer)
            n_final = normalize_answer(final_answer)
            return (n_ex != "" and n_ex == n_final), n_ex, not_normalized_final

    elif benchmark_name == "webarena":
        # Read the console log
        console_log_file = os.path.join(instance_dir, "console_log.txt")
        if not os.path.isfile(console_log_file):
            return None

        with open(console_log_file, "rt") as fh:
            console_log = fh.read()
            final_score = None
            m = re.search(r"FINAL SCORE:(.*?)\n", console_log, re.DOTALL)
            if m:
                final_score = m.group(1).strip()

            if final_score is None:
                return None
            else:
                return float(final_score) > 0, "", ""

    else:
        raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

# From benchmarks/process_logs.py
def get_number_of_chat_messages(chat_messages_dir):
    # Count the number of chat messages in the chat_messages_dir
    result = 0
    for file in glob.glob(f"{chat_messages_dir}/*_messages.json"):
        with open(file, "r") as f:
            content = json.load(f)
            for agent, messages in content.items():
                result += len(messages)
    return result

# From benchmarks/process_logs.py
def did_agent_stall(instance_dir):
    # Check if the agent stalled
    log_file_path = os.path.join(instance_dir, "log.jsonl")
    if not os.path.isfile(log_file_path):
        return None
    # Stalled.... Replanning...
    with open(log_file_path, "r") as f:
        for line in f:
            if "Stalled.... Replanning..." in line:
                return True
    return False

# From benchmarks/process_logs.py
def get_message_logs(instance_dir):
    # Read the log file and return the messages
    log_file_path = os.path.join(instance_dir, "log.jsonl")
    if not os.path.isfile(log_file_path):
        return None
    messages = []
    # for each line, convert to dict, check if it has a message and source key, and append to messages
    with open(log_file_path, "r") as f:
        for line in f:
            line_dict = json.loads(line)
            if "message" in line_dict and "source" in line_dict:
                messages.append(line_dict)
    return messages

# From benchmarks/process_logs.py
def get_task_information(instance_dir, benchmark_name):
    # Read the task information from the log file
    if benchmark_name == "gaia" or benchmark_name == "assistant":
        prompt_file = os.path.join(instance_dir, "prompt.txt")
        if not os.path.isfile(prompt_file):
            return None
        with open(prompt_file, "r") as f:
            return f.read().strip()
    elif benchmark_name == "webarena":
        task_prompt_file = os.path.join(instance_dir, "task_prompt.json")
        if not os.path.isfile(task_prompt_file):
            return None
        with open(task_prompt_file, "r") as f:
            return json.load(f)["intent"]
    else:
        raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

# From benchmarks/process_logs.py
def is_progress_not_being_made(instance_dir):
    # if at any point in the log, progress is not being made, return True
    pattern = r'"is_progress_being_made": \{\s+"reason": ".*?",\s+"answer": false\s+\}'
    log_file_path = os.path.join(instance_dir, "log.jsonl")
    if not os.path.isfile(log_file_path):
        return None
    with open(log_file_path, "r") as f:
        for line in f:
            line_dict = json.loads(line)
            if (
                "source" in line_dict
                and line_dict["source"] == "Orchestrator (thought)"
                and "Updated Ledger:" in line_dict["message"]
                and re.search(pattern, line_dict["message"])
            ):
                return True
    return False

import yaml
import contextvars
import builtins
import shutil
from collections import deque
from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_core import TRACE_LOGGER_NAME
from autogen_core import EVENT_LOGGER_NAME
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.logging import LLMCallEvent
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.messages import AgentEvent
from autogen_agentchat.messages import ChatMessage
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.messages import StopMessage
from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_agentchat.messages import ToolCallRequestEvent
from autogen_agentchat.messages import ToolCallSummaryMessage
from autogen_ext.models.openai._model_info import _MODEL_TOKEN_LIMITS
from autogen_ext.models.openai._model_info import resolve_model
from autogen_agentchat.utils import content_to_str

# From ParallelAgents/scenario.py
class LogHandler(logging.FileHandler):
    def __init__(self, filename: str = "log.jsonl", print_message: bool = True) -> None:
        super().__init__(filename, mode="w")
        self.print_message = print_message

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).isoformat()
            if AGENTCHAT_EVENT_LOGGER_NAME in record.name:
                original_msg = record.msg
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": content_to_str(record.msg.content),
                        "type": record.msg.type,
                    }
                )
                super().emit(record)
                record.msg = original_msg
            elif CORE_EVENT_LOGGER_NAME in record.name:
                if isinstance(record.msg, LLMCallEvent):
                    original_msg = record.msg
                    record.msg = json.dumps(
                        {
                            "timestamp": ts,
                            "prompt_tokens": record.msg.kwargs["prompt_tokens"],
                            "completion_tokens": record.msg.kwargs["completion_tokens"],
                            "type": "LLMCallEvent",
                        }
                    )
                    super().emit(record)
                    record.msg = original_msg
        except Exception:
            print("error in logHandler.emit", flush=True)
            self.handleError(record)

# From ParallelAgents/scenario.py
def tee_print(*args, **kwargs):
    # Get the current log file from the context.
    log_file = current_log_file.get()
    # Call the original print (goes to the console).
    original_print(*args, **kwargs)
    # Also write to the log file if one is set.
    if log_file is not None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(map(str, args)) + end
        log_file.write(message)
        log_file.flush()

# From ParallelAgents/scenario.py
def team_specific_agentchat_event_logger_info(msg, *args, **kwargs):
    team_id = current_team_id.get()
    if team_id is not None:
        # Get a logger with a team-specific name.
        team_logger = logging.getLogger(f"{AGENTCHAT_EVENT_LOGGER_NAME}.team{team_id}")
        team_logger.info(msg, *args, **kwargs)
    else:
        original_agentchat_event_logger_info(msg, *args, **kwargs)

# From ParallelAgents/scenario.py
def team_specific_core_event_logger_info(msg, *args, **kwargs):
    team_id = current_team_id.get()
    if team_id is not None:
        # Get a logger with a team-specific name.
        team_logger = logging.getLogger(f"{CORE_EVENT_LOGGER_NAME}.team{team_id}")
        team_logger.info(msg, *args, **kwargs)
    else:
        original_core_event_logger_info(msg, *args, **kwargs)

# From ParallelAgents/scenario.py
def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).isoformat()
            if AGENTCHAT_EVENT_LOGGER_NAME in record.name:
                original_msg = record.msg
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": content_to_str(record.msg.content),
                        "type": record.msg.type,
                    }
                )
                super().emit(record)
                record.msg = original_msg
            elif CORE_EVENT_LOGGER_NAME in record.name:
                if isinstance(record.msg, LLMCallEvent):
                    original_msg = record.msg
                    record.msg = json.dumps(
                        {
                            "timestamp": ts,
                            "prompt_tokens": record.msg.kwargs["prompt_tokens"],
                            "completion_tokens": record.msg.kwargs["completion_tokens"],
                            "type": "LLMCallEvent",
                        }
                    )
                    super().emit(record)
                    record.msg = original_msg
        except Exception:
            print("error in logHandler.emit", flush=True)
            self.handleError(record)


# From AgentChat/custom_code_executor.py
class CustomCodeExecutorAgent(CodeExecutorAgent):

    def __init__(
        self,
        name: str,
        code_executor: CodeExecutor,
        *,
        description: str = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, code_executor=code_executor, sources=sources)
        self._test_code = ""
        with open("test.txt", "rt") as fh:
            self._test_code = fh.read()


    def _extract_markdown_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        code_blocks = super()._extract_markdown_code_blocks(markdown_text)
        new_blocks: List[CodeBlock] = []
        for block in code_blocks:

            # Handle deepseek
            code_content = block.code
            #m = re.search(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*?)\s*$", code_content, re.DOTALL)
            #if m:
            #    code_content = m.group(2)

            # If python, wrap the extracted code in a unit testing harness
            if block.language and block.language.lower() == "python":
                code_content = self._test_code + """

def run_tests(candidate):
    try:
        check(candidate)
        # We can search for this string in the output
        print("ALL TESTS PASSED !#!#")
        print("TERMINATE")
    except AssertionError:
        print("SOME TESTS FAILED - TRY AGAIN !#!#")

""" + code_content + """

run_tests(__ENTRY_POINT__)
"""
            new_blocks.append(CodeBlock(code=code_content, language=block.language))

        return new_blocks

from autogen_agentchat.agents import ApprovalFuncType
from autogen_agentchat.base import ChatAgent
from autogen_ext.models.openai._openai_client import BaseOpenAIChatCompletionClient
import docker
from docker.errors import DockerException
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

# From teams/magentic_one.py
class MagenticOne(MagenticOneGroupChat):
    """
    MagenticOne is a specialized group chat class that integrates various agents
    such as FileSurfer, WebSurfer, Coder, and Executor to solve complex tasks.
    To read more about the science behind Magentic-One, see the full blog post: `Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks <https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks>`_ and the references below.

    Installation:

    .. code-block:: bash

        pip install "autogen-ext[magentic-one]"


    Args:
        client (ChatCompletionClient): The client used for model interactions.
        hil_mode (bool): Optional; If set to True, adds the UserProxyAgent to the list of agents.
        input_func (InputFuncType | None): Optional; Function to use for user input in human-in-the-loop mode.
        code_executor (CodeExecutor | None): Optional; Code executor to use. If None, will use Docker if available, otherwise local executor.
        approval_func (ApprovalFuncType | None): Optional; Function to approve code execution before running. If None, code will execute without approval.

    .. warning::
        Using Magentic-One involves interacting with a digital world designed for humans, which carries inherent risks. To minimize these risks, consider the following precautions:

        1. **Use Containers**: Run all tasks in docker containers to isolate the agents and prevent direct system attacks.
        2. **Virtual Environment**: Use a virtual environment to run the agents and prevent them from accessing sensitive data.
        3. **Monitor Logs**: Closely monitor logs during and after execution to detect and mitigate risky behavior.
        4. **Human Oversight**: Run the examples with a human in the loop to supervise the agents and prevent unintended consequences.
        5. **Limit Access**: Restrict the agents' access to the internet and other resources to prevent unauthorized actions.
        6. **Safeguard Data**: Ensure that the agents do not have access to sensitive data or resources that could be compromised. Do not share sensitive information with the agents.

        Be aware that agents may occasionally attempt risky actions, such as recruiting humans for help or accepting cookie agreements without human involvement. Always ensure agents are monitored and operate within a controlled environment to prevent unintended consequences. Moreover, be cautious that Magentic-One may be susceptible to prompt injection attacks from webpages.

    Architecture:

    Magentic-One is a generalist multi-agent system for solving open-ended web and file-based tasks across a variety of domains. It represents a significant step towards developing agents that can complete tasks that people encounter in their work and personal lives.

    Magentic-One work is based on a multi-agent architecture where a lead Orchestrator agent is responsible for high-level planning, directing other agents, and tracking task progress. The Orchestrator begins by creating a plan to tackle the task, gathering needed facts and educated guesses in a Task Ledger that is maintained. At each step of its plan, the Orchestrator creates a Progress Ledger where it self-reflects on task progress and checks whether the task is completed. If the task is not yet completed, it assigns one of Magentic-One's other agents a subtask to complete. After the assigned agent completes its subtask, the Orchestrator updates the Progress Ledger and continues in this way until the task is complete. If the Orchestrator finds that progress is not being made for enough steps, it can update the Task Ledger and create a new plan.

    Overall, Magentic-One consists of the following agents:

    - Orchestrator: The lead agent responsible for task decomposition and planning, directing other agents in executing subtasks, tracking overall progress, and taking corrective actions as needed.
    - WebSurfer: An LLM-based agent proficient in commanding and managing the state of a Chromium-based web browser. It performs actions on the browser and reports on the new state of the web page.
    - FileSurfer: An LLM-based agent that commands a markdown-based file preview application to read local files of most types. It can also perform common navigation tasks such as listing the contents of directories and navigating a folder structure.
    - Coder: An LLM-based agent specialized in writing code, analyzing information collected from other agents, or creating new artifacts.
    - ComputerTerminal: Provides the team with access to a console shell where the Coder's programs can be executed, and where new programming libraries can be installed.

    Together, Magentic-One's agents provide the Orchestrator with the tools and capabilities needed to solve a broad variety of open-ended problems, as well as the ability to autonomously adapt to, and act in, dynamic and ever-changing web and file-system environments.

    Examples:

        .. code-block:: python

            # Autonomously complete a coding task:
            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_ext.teams.magentic_one import MagenticOne
            from autogen_agentchat.ui import Console


            async def example_usage():
                client = OpenAIChatCompletionClient(model="gpt-4o")
                m1 = MagenticOne(client=client)  # Uses DockerCommandLineCodeExecutor by default
                task = "Write a Python script to fetch data from an API."
                result = await Console(m1.run_stream(task=task))
                print(result)


            if __name__ == "__main__":
                asyncio.run(example_usage())


        .. code-block:: python

            # Enable human-in-the-loop mode with explicit Docker executor and code approval
            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_ext.teams.magentic_one import MagenticOne
            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
            from autogen_agentchat.ui import Console
            from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse


            def user_input_func(prompt: str) -> str:
                \"\"\"Custom input function for user interaction.\"\"\"
                return input(prompt)


            def approval_func(request: ApprovalRequest) -> ApprovalResponse:
                \"\"\"Simple approval function that requests user input.\"\"\"
                print(f\"Code to execute:\\n{request.code}\")
                user_input = input("Do you approve this code execution? (y/n): ").strip().lower()
                if user_input == 'y':
                    return ApprovalResponse(approved=True, reason=\"User approved the code execution\")
                else:
                    return ApprovalResponse(approved=False, reason=\"User denied the code execution\")


            async def example_usage_hil():
                client = OpenAIChatCompletionClient(model="gpt-4o")
                # Explicitly specify Docker code executor for better security
                async with DockerCommandLineCodeExecutor() as code_executor:
                    m1 = MagenticOne(
                        client=client,
                        hil_mode=True,
                        input_func=user_input_func,
                        code_executor=code_executor,
                        approval_func=approval_func
                    )
                    task = "Write a Python script to fetch data from an API."
                    result = await Console(m1.run_stream(task=task))
                    print(result)


            if __name__ == "__main__":
                asyncio.run(example_usage_hil())


        .. code-block:: python

            # Enable code execution approval without human-in-the-loop mode
            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_ext.teams.magentic_one import MagenticOne
            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
            from autogen_agentchat.ui import Console
            from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse


            def approval_func(request: ApprovalRequest) -> ApprovalResponse:
                \"\"\"Simple approval function that requests user input.\"\"\"
                print(f\"Code to execute:\\n{request.code}\")
                user_input = input("Do you approve this code execution? (y/n): ").strip().lower()
                if user_input == 'y':
                    return ApprovalResponse(approved=True, reason=\"User approved the code execution\")
                else:
                    return ApprovalResponse(approved=False, reason=\"User denied the code execution\")


            async def example_usage_with_approval():
                client = OpenAIChatCompletionClient(model="gpt-4o")
                # Use approval_func for code approval only (hil_mode=False)
                async with DockerCommandLineCodeExecutor() as code_executor:
                    m1 = MagenticOne(
                        client=client,
                        hil_mode=False,  # No human-in-the-loop for general conversation
                        code_executor=code_executor,
                        approval_func=approval_func  # But still ask for code execution approval
                    )
                    task = "Write a Python script to fetch data from an API."
                    result = await Console(m1.run_stream(task=task))
                    print(result)


            if __name__ == "__main__":
                asyncio.run(example_usage_with_approval())

    References:
        .. code-block:: bibtex

            @article{fourney2024magentic,
                title={Magentic-one: A generalist multi-agent system for solving complex tasks},
                author={Fourney, Adam and Bansal, Gagan and Mozannar, Hussein and Tan, Cheng and Salinas, Eduardo and Niedtner, Friederike and Proebsting, Grace and Bassman, Griffin and Gerrits, Jack and Alber, Jacob and others},
                journal={arXiv preprint arXiv:2411.04468},
                year={2024},
                url={https://arxiv.org/abs/2411.04468}
            }


    """

    def __init__(
        self,
        client: ChatCompletionClient,
        hil_mode: bool = False,
        input_func: InputFuncType | None = None,
        code_executor: CodeExecutor | None = None,
        approval_func: ApprovalFuncType | None = None,
    ):
        self.client = client
        self._validate_client_capabilities(client)

        if code_executor is None:
            warnings.warn(
                "Instantiating MagenticOne without a code_executor is deprecated. Provide a code_executor to clear this warning (e.g., code_executor=DockerCommandLineCodeExecutor() ).",
                DeprecationWarning,
                stacklevel=2,
            )
            code_executor = _create_default_code_executor()

        fs = FileSurfer("FileSurfer", model_client=client)
        ws = MultimodalWebSurfer("WebSurfer", model_client=client)
        coder = MagenticOneCoderAgent("Coder", model_client=client)

        executor = CodeExecutorAgent("ComputerTerminal", code_executor=code_executor, approval_func=approval_func)

        agents: List[ChatAgent] = [fs, ws, coder, executor]
        if hil_mode:
            user_proxy = UserProxyAgent("User", input_func=input_func)
            agents.append(user_proxy)
        super().__init__(agents, model_client=client)

    def _validate_client_capabilities(self, client: ChatCompletionClient) -> None:
        capabilities = client.model_info
        required_capabilities = ["function_calling", "json_output"]

        if not all(capabilities.get(cap) for cap in required_capabilities):
            warnings.warn(
                "Client capabilities for MagenticOne must include vision, " "function calling, and json output.",
                stacklevel=2,
            )

        if not isinstance(client, BaseOpenAIChatCompletionClient):
            warnings.warn(
                "MagenticOne performs best with OpenAI GPT-4o model either " "through OpenAI or Azure OpenAI.",
                stacklevel=2,
            )

from typing import AsyncIterator
from autogen_core import TopicId
from autogen_core._agent_id import AgentId
from autogen_core._runtime_impl_helpers import SubscriptionManager
from _constants import GRPC_IMPORT_ERROR_STR
from _utils import subscription_from_proto
from _utils import subscription_to_proto
from protos import agent_worker_pb2
from protos import agent_worker_pb2_grpc
from protos import cloudevent_pb2

# From grpc/_worker_runtime_host_servicer.py
class ChannelConnection(ABC, Generic[SendT, ReceiveT]):
    def __init__(self, request_iterator: AsyncIterator[ReceiveT], client_id: str) -> None:
        self._request_iterator = request_iterator
        self._client_id = client_id
        self._send_queue: asyncio.Queue[SendT] = asyncio.Queue()
        self._receiving_task = asyncio.create_task(self._receive_messages(client_id, request_iterator))

    async def _receive_messages(self, client_id: ClientConnectionId, request_iterator: AsyncIterator[ReceiveT]) -> None:
        # Receive messages from the client and process them.
        async for message in request_iterator:
            logger.info(f"Received message from client {client_id}: {message}")
            await self._handle_message(message)

    def __aiter__(self) -> AsyncIterator[SendT]:
        return self

    async def __anext__(self) -> SendT:
        try:
            return await self._send_queue.get()
        except StopAsyncIteration:
            await self._receiving_task
            raise
        except Exception as e:
            logger.error(f"Failed to get message from send queue: {e}", exc_info=True)
            await self._receiving_task
            raise

    @abstractmethod
    async def _handle_message(self, message: ReceiveT) -> None:
        pass

    async def send(self, message: SendT) -> None:
        await self._send_queue.put(message)

# From grpc/_worker_runtime_host_servicer.py
class CallbackChannelConnection(ChannelConnection[SendT, ReceiveT]):
    def __init__(
        self,
        request_iterator: AsyncIterator[ReceiveT],
        client_id: str,
        handle_callback: Callable[[ReceiveT], Awaitable[None]],
    ) -> None:
        self._handle_callback = handle_callback
        super().__init__(request_iterator, client_id)

    async def _handle_message(self, message: ReceiveT) -> None:
        await self._handle_callback(message)

# From grpc/_worker_runtime_host_servicer.py
class GrpcWorkerAgentRuntimeHostServicer(agent_worker_pb2_grpc.AgentRpcServicer):
    """A gRPC servicer that hosts message delivery service for agents."""

    def __init__(self) -> None:
        self._data_connections: Dict[
            ClientConnectionId, ChannelConnection[agent_worker_pb2.Message, agent_worker_pb2.Message]
        ] = {}
        self._control_connections: Dict[
            ClientConnectionId, ChannelConnection[agent_worker_pb2.ControlMessage, agent_worker_pb2.ControlMessage]
        ] = {}
        self._agent_type_to_client_id_lock = asyncio.Lock()
        self._agent_type_to_client_id: Dict[str, ClientConnectionId] = {}
        self._pending_responses: Dict[ClientConnectionId, Dict[str, Future[Any]]] = {}
        self._background_tasks: Set[Task[Any]] = set()
        self._subscription_manager = SubscriptionManager()
        self._client_id_to_subscription_id_mapping: Dict[ClientConnectionId, set[str]] = {}

    async def OpenChannel(  # type: ignore
        self,
        request_iterator: AsyncIterator[agent_worker_pb2.Message],
        context: grpc.aio.ServicerContext[agent_worker_pb2.Message, agent_worker_pb2.Message],
    ) -> AsyncIterator[agent_worker_pb2.Message]:
        client_id = await get_client_id_or_abort(context)

        async def handle_callback(message: agent_worker_pb2.Message) -> None:
            await self._receive_message(client_id, message)

        connection = CallbackChannelConnection[agent_worker_pb2.Message, agent_worker_pb2.Message](
            request_iterator, client_id, handle_callback=handle_callback
        )
        self._data_connections[client_id] = connection
        logger.info(f"Client {client_id} connected.")

        try:
            async for message in connection:
                yield message
        finally:
            # Clean up the client connection.
            del self._data_connections[client_id]
            # Cancel pending requests sent to this client.
            for future in self._pending_responses.pop(client_id, {}).values():
                future.cancel()
            # Remove the client id from the agent type to client id mapping.
            await self._on_client_disconnect(client_id)

    async def OpenControlChannel(  # type: ignore
        self,
        request_iterator: AsyncIterator[agent_worker_pb2.ControlMessage],
        context: grpc.aio.ServicerContext[agent_worker_pb2.ControlMessage, agent_worker_pb2.ControlMessage],
    ) -> AsyncIterator[agent_worker_pb2.ControlMessage]:
        client_id = await get_client_id_or_abort(context)

        async def handle_callback(message: agent_worker_pb2.ControlMessage) -> None:
            await self._receive_control_message(client_id, message)

        connection = CallbackChannelConnection[agent_worker_pb2.ControlMessage, agent_worker_pb2.ControlMessage](
            request_iterator, client_id, handle_callback=handle_callback
        )
        self._control_connections[client_id] = connection
        logger.info(f"Client {client_id} connected.")

        try:
            async for message in connection:
                yield message
        finally:
            # Clean up the client connection.
            del self._control_connections[client_id]

    async def _on_client_disconnect(self, client_id: ClientConnectionId) -> None:
        async with self._agent_type_to_client_id_lock:
            agent_types = [agent_type for agent_type, id_ in self._agent_type_to_client_id.items() if id_ == client_id]
            for agent_type in agent_types:
                logger.info(f"Removing agent type {agent_type} from agent type to client id mapping")
                del self._agent_type_to_client_id[agent_type]
            for sub_id in self._client_id_to_subscription_id_mapping.get(client_id, set()):
                logger.info(f"Client id {client_id} disconnected. Removing corresponding subscription with id {id}")
                try:
                    await self._subscription_manager.remove_subscription(sub_id)
                # Catch and ignore if the subscription does not exist.
                except ValueError:
                    continue
        logger.info(f"Client {client_id} disconnected successfully")

    def _raise_on_exception(self, task: Task[Any]) -> None:
        exception = task.exception()
        if exception is not None:
            raise exception

    async def _receive_message(self, client_id: ClientConnectionId, message: agent_worker_pb2.Message) -> None:
        logger.info(f"Received message from client {client_id}: {message}")
        oneofcase = message.WhichOneof("message")
        match oneofcase:
            case "request":
                request: agent_worker_pb2.RpcRequest = message.request
                task = asyncio.create_task(self._process_request(request, client_id))
                self._background_tasks.add(task)
                task.add_done_callback(self._raise_on_exception)
                task.add_done_callback(self._background_tasks.discard)
            case "response":
                response: agent_worker_pb2.RpcResponse = message.response
                task = asyncio.create_task(self._process_response(response, client_id))
                self._background_tasks.add(task)
                task.add_done_callback(self._raise_on_exception)
                task.add_done_callback(self._background_tasks.discard)
            case "cloudEvent":
                task = asyncio.create_task(self._process_event(message.cloudEvent))
                self._background_tasks.add(task)
                task.add_done_callback(self._raise_on_exception)
                task.add_done_callback(self._background_tasks.discard)
            case None:
                logger.warning("Received empty message")

    async def _receive_control_message(
        self, client_id: ClientConnectionId, message: agent_worker_pb2.ControlMessage
    ) -> None:
        logger.info(f"Received message from client {client_id}: {message}")
        destination = message.destination
        if destination.startswith("agentid="):
            agent_id = AgentId.from_str(destination[len("agentid=") :])
            target_client_id = self._agent_type_to_client_id.get(agent_id.type)
            if target_client_id is None:
                logger.error(f"Agent client id not found for agent type {agent_id.type}.")
                return
        elif destination.startswith("clientid="):
            target_client_id = destination[len("clientid=") :]
        else:
            logger.error(f"Invalid destination {destination}")
            return

        target_send_queue = self._control_connections.get(target_client_id)
        if target_send_queue is None:
            logger.error(f"Client {target_client_id} not found, failed to deliver message.")
            return
        await target_send_queue.send(message)

    async def _process_request(self, request: agent_worker_pb2.RpcRequest, client_id: ClientConnectionId) -> None:
        # Deliver the message to a client given the target agent type.
        async with self._agent_type_to_client_id_lock:
            target_client_id = self._agent_type_to_client_id.get(request.target.type)
        if target_client_id is None:
            logger.error(f"Agent {request.target.type} not found, failed to deliver message.")
            return
        target_send_queue = self._data_connections.get(target_client_id)
        if target_send_queue is None:
            logger.error(f"Client {target_client_id} not found, failed to deliver message.")
            return
        await target_send_queue.send(agent_worker_pb2.Message(request=request))

        # Create a future to wait for the response from the target.
        future = asyncio.get_event_loop().create_future()
        self._pending_responses.setdefault(target_client_id, {})[request.request_id] = future

        # Create a task to wait for the response and send it back to the client.
        send_response_task = asyncio.create_task(self._wait_and_send_response(future, client_id))
        self._background_tasks.add(send_response_task)
        send_response_task.add_done_callback(self._raise_on_exception)
        send_response_task.add_done_callback(self._background_tasks.discard)

    async def _wait_and_send_response(
        self, future: Future[agent_worker_pb2.RpcResponse], client_id: ClientConnectionId
    ) -> None:
        response = await future
        message = agent_worker_pb2.Message(response=response)
        send_queue = self._data_connections.get(client_id)
        if send_queue is None:
            logger.error(f"Client {client_id} not found, failed to send response message.")
            return
        await send_queue.send(message)

    async def _process_response(self, response: agent_worker_pb2.RpcResponse, client_id: ClientConnectionId) -> None:
        # Setting the result of the future will send the response back to the original sender.
        future = self._pending_responses[client_id].pop(response.request_id)
        future.set_result(response)

    async def _process_event(self, event: cloudevent_pb2.CloudEvent) -> None:
        topic_id = TopicId(type=event.type, source=event.source)
        recipients = await self._subscription_manager.get_subscribed_recipients(topic_id)
        # Get the client ids of the recipients.
        async with self._agent_type_to_client_id_lock:
            client_ids: Set[ClientConnectionId] = set()
            for recipient in recipients:
                client_id = self._agent_type_to_client_id.get(recipient.type)
                if client_id is not None:
                    client_ids.add(client_id)
                else:
                    logger.error(f"Agent {recipient.type} and its client not found for topic {topic_id}.")
        # Deliver the event to clients.
        for client_id in client_ids:
            await self._data_connections[client_id].send(agent_worker_pb2.Message(cloudEvent=event))

    async def RegisterAgent(  # type: ignore
        self,
        request: agent_worker_pb2.RegisterAgentTypeRequest,
        context: grpc.aio.ServicerContext[
            agent_worker_pb2.RegisterAgentTypeRequest, agent_worker_pb2.RegisterAgentTypeResponse
        ],
    ) -> agent_worker_pb2.RegisterAgentTypeResponse:
        client_id = await get_client_id_or_abort(context)

        async with self._agent_type_to_client_id_lock:
            if request.type in self._agent_type_to_client_id:
                existing_client_id = self._agent_type_to_client_id[request.type]
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Agent type {request.type} already registered with client {existing_client_id}.",
                )
            else:
                self._agent_type_to_client_id[request.type] = client_id

        return agent_worker_pb2.RegisterAgentTypeResponse()

    async def AddSubscription(  # type: ignore
        self,
        request: agent_worker_pb2.AddSubscriptionRequest,
        context: grpc.aio.ServicerContext[
            agent_worker_pb2.AddSubscriptionRequest, agent_worker_pb2.AddSubscriptionResponse
        ],
    ) -> agent_worker_pb2.AddSubscriptionResponse:
        client_id = await get_client_id_or_abort(context)

        subscription = subscription_from_proto(request.subscription)
        try:
            await self._subscription_manager.add_subscription(subscription)
            subscription_ids = self._client_id_to_subscription_id_mapping.setdefault(client_id, set())
            subscription_ids.add(subscription.id)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return agent_worker_pb2.AddSubscriptionResponse()

    async def RemoveSubscription(  # type: ignore
        self,
        request: agent_worker_pb2.RemoveSubscriptionRequest,
        context: grpc.aio.ServicerContext[
            agent_worker_pb2.RemoveSubscriptionRequest, agent_worker_pb2.RemoveSubscriptionResponse
        ],
    ) -> agent_worker_pb2.RemoveSubscriptionResponse:
        _client_id = await get_client_id_or_abort(context)
        await self._subscription_manager.remove_subscription(request.id)
        return agent_worker_pb2.RemoveSubscriptionResponse()

    async def GetSubscriptions(  # type: ignore
        self,
        request: agent_worker_pb2.GetSubscriptionsRequest,
        context: grpc.aio.ServicerContext[
            agent_worker_pb2.GetSubscriptionsRequest, agent_worker_pb2.GetSubscriptionsResponse
        ],
    ) -> agent_worker_pb2.GetSubscriptionsResponse:
        _client_id = await get_client_id_or_abort(context)
        subscriptions = self._subscription_manager.subscriptions
        return agent_worker_pb2.GetSubscriptionsResponse(
            subscriptions=[subscription_to_proto(sub) for sub in subscriptions]
        )

# From grpc/_worker_runtime_host_servicer.py
def metadata_to_dict(metadata: Sequence[Tuple[str, str]] | None) -> Dict[str, str]:
    if metadata is None:
        return {}
    return {key: value for key, value in metadata}

import signal
from collections import defaultdict
from typing import AsyncIterable
from autogen_core import JSON_DATA_CONTENT_TYPE
from autogen_core import Agent
from autogen_core import AgentInstantiationContext
from autogen_core import AgentMetadata
from autogen_core import AgentType
from autogen_core import MessageHandlerContext
from autogen_core import MessageSerializer
from autogen_core import Subscription
from autogen_core._runtime_impl_helpers import get_impl
from autogen_core._serialization import SerializationRegistry
from autogen_core._telemetry import MessageRuntimeTracingConfig
from autogen_core._telemetry import TraceHelper
from autogen_core._telemetry import get_telemetry_grpc_metadata
from google.protobuf import any_pb2
from autogen_ext.runtimes.grpc._utils import subscription_to_proto
from  import _constants
from _type_helpers import ChannelArgumentType
import grpc.aio
from protos.agent_worker_pb2_grpc import AgentRpcAsyncStub
from grpc.aio import StreamStreamCall

# From grpc/_worker_runtime.py
class QueueAsyncIterable(AsyncIterator[Any], AsyncIterable[Any]):
    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self._queue = queue

    async def __anext__(self) -> Any:
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

# From grpc/_worker_runtime.py
class HostConnection:
    DEFAULT_GRPC_CONFIG: ClassVar[ChannelArgumentType] = [
        (
            "grpc.service_config",
            json.dumps(
                {
                    "methodConfig": [
                        {
                            "name": [{}],
                            "retryPolicy": {
                                "maxAttempts": 3,
                                "initialBackoff": "0.01s",
                                "maxBackoff": "5s",
                                "backoffMultiplier": 2,
                                "retryableStatusCodes": ["UNAVAILABLE"],
                            },
                        }
                    ],
                }
            ),
        )
    ]

    def __init__(self, channel: grpc.aio.Channel, stub: Any) -> None:  # type: ignore
        self._channel = channel
        self._send_queue = asyncio.Queue[agent_worker_pb2.Message]()
        self._recv_queue = asyncio.Queue[agent_worker_pb2.Message]()
        self._connection_task: Task[None] | None = None
        self._stub: AgentRpcAsyncStub = stub
        self._client_id = str(uuid.uuid4())

    @property
    def stub(self) -> Any:
        return self._stub

    @property
    def metadata(self) -> Sequence[Tuple[str, str]]:
        return [("client-id", self._client_id)]

    @classmethod
    async def from_host_address(
        cls, host_address: str, extra_grpc_config: ChannelArgumentType = DEFAULT_GRPC_CONFIG
    ) -> Self:
        logger.info("Connecting to %s", host_address)
        #  Always use DEFAULT_GRPC_CONFIG and override it with provided grpc_config
        merged_options = [
            (k, v) for k, v in {**dict(HostConnection.DEFAULT_GRPC_CONFIG), **dict(extra_grpc_config)}.items()
        ]

        channel = grpc.aio.insecure_channel(
            host_address,
            options=merged_options,
        )
        stub: AgentRpcAsyncStub = agent_worker_pb2_grpc.AgentRpcStub(channel)  # type: ignore
        instance = cls(channel, stub)

        instance._connection_task = await instance._connect(
            stub, instance._send_queue, instance._recv_queue, instance._client_id
        )

        return instance

    async def close(self) -> None:
        if self._connection_task is None:
            raise RuntimeError("Connection is not open.")
        await self._channel.close()
        await self._connection_task

    @staticmethod
    async def _connect(
        stub: Any,  # AgentRpcAsyncStub
        send_queue: asyncio.Queue[agent_worker_pb2.Message],
        receive_queue: asyncio.Queue[agent_worker_pb2.Message],
        client_id: str,
    ) -> Task[None]:
        from grpc.aio import StreamStreamCall

        # TODO: where do exceptions from reading the iterable go? How do we recover from those?
        stream: StreamStreamCall[agent_worker_pb2.Message, agent_worker_pb2.Message] = stub.OpenChannel(  # type: ignore
            QueueAsyncIterable(send_queue), metadata=[("client-id", client_id)]
        )

        await stream.wait_for_connection()

        async def read_loop() -> None:
            while True:
                logger.info("Waiting for message from host")
                message = cast(agent_worker_pb2.Message, await stream.read())  # type: ignore
                if message == grpc.aio.EOF:  # type: ignore
                    logger.info("EOF")
                    break
                logger.info(f"Received a message from host: {message}")
                await receive_queue.put(message)
                logger.info("Put message in receive queue")

        return asyncio.create_task(read_loop())

    async def send(self, message: agent_worker_pb2.Message) -> None:
        logger.info(f"Send message to host: {message}")
        await self._send_queue.put(message)
        logger.info("Put message in send queue")

    async def recv(self) -> agent_worker_pb2.Message:
        logger.info("Getting message from queue")
        return await self._recv_queue.get()

# From grpc/_worker_runtime.py
class GrpcWorkerAgentRuntime(AgentRuntime):
    """An agent runtime for running remote or cross-language agents.

    Agent messaging uses protobufs from `agent_worker.proto`_ and ``CloudEvent`` from `cloudevent.proto`_.

    Cross-language agents will additionally require all agents use shared protobuf schemas for any message types that are sent between agents.

    .. _agent_worker.proto: https://github.com/microsoft/autogen/blob/main/protos/agent_worker.proto

    .. _cloudevent.proto: https://github.com/microsoft/autogen/blob/main/protos/cloudevent.proto

    """

    # TODO: Needs to handle agent close() call
    def __init__(
        self,
        host_address: str,
        tracer_provider: TracerProvider | None = None,
        extra_grpc_config: ChannelArgumentType | None = None,
        payload_serialization_format: str = JSON_DATA_CONTENT_TYPE,
    ) -> None:
        self._host_address = host_address
        self._trace_helper = TraceHelper(tracer_provider, MessageRuntimeTracingConfig("Worker Runtime"))
        self._per_type_subscribers: DefaultDict[tuple[str, str], Set[AgentId]] = defaultdict(set)
        self._agent_factories: Dict[
            str, Callable[[], Agent | Awaitable[Agent]] | Callable[[AgentRuntime, AgentId], Agent | Awaitable[Agent]]
        ] = {}
        self._instantiated_agents: Dict[AgentId, Agent] = {}
        self._known_namespaces: set[str] = set()
        self._read_task: None | Task[None] = None
        self._running = False
        self._pending_requests: Dict[str, Future[Any]] = {}
        self._pending_requests_lock = asyncio.Lock()
        self._next_request_id = 0
        self._host_connection: HostConnection | None = None
        self._background_tasks: Set[Task[Any]] = set()
        self._subscription_manager = SubscriptionManager()
        self._serialization_registry = SerializationRegistry()
        self._extra_grpc_config = extra_grpc_config or []
        self._agent_instance_types: Dict[str, Type[Agent]] = {}

        if payload_serialization_format not in {JSON_DATA_CONTENT_TYPE, PROTOBUF_DATA_CONTENT_TYPE}:
            raise ValueError(f"Unsupported payload serialization format: {payload_serialization_format}")

        self._payload_serialization_format = payload_serialization_format

    async def start(self) -> None:
        """Start the runtime in a background task."""
        if self._running:
            raise ValueError("Runtime is already running.")
        logger.info(f"Connecting to host: {self._host_address}")
        self._host_connection = await HostConnection.from_host_address(
            self._host_address, extra_grpc_config=self._extra_grpc_config
        )
        logger.info("Connection established")
        if self._read_task is None:
            self._read_task = asyncio.create_task(self._run_read_loop())
        self._running = True

    def _raise_on_exception(self, task: Task[Any]) -> None:
        exception = task.exception()
        if exception is not None:
            raise exception

    async def _run_read_loop(self) -> None:
        logger.info("Starting read loop")
        assert self._host_connection is not None
        # TODO: catch exceptions and reconnect
        while self._running:
            try:
                message = await self._host_connection.recv()
                oneofcase = agent_worker_pb2.Message.WhichOneof(message, "message")
                match oneofcase:
                    case "request":
                        task = asyncio.create_task(self._process_request(message.request))
                        self._background_tasks.add(task)
                        task.add_done_callback(self._raise_on_exception)
                        task.add_done_callback(self._background_tasks.discard)
                    case "response":
                        task = asyncio.create_task(self._process_response(message.response))
                        self._background_tasks.add(task)
                        task.add_done_callback(self._raise_on_exception)
                        task.add_done_callback(self._background_tasks.discard)
                    case "cloudEvent":
                        task = asyncio.create_task(self._process_event(message.cloudEvent))
                        self._background_tasks.add(task)
                        task.add_done_callback(self._raise_on_exception)
                        task.add_done_callback(self._background_tasks.discard)
                    case None:
                        logger.warning("No message")
            except Exception as e:
                logger.error("Error in read loop", exc_info=e)

    async def stop(self) -> None:
        """Stop the runtime immediately."""
        if not self._running:
            raise RuntimeError("Runtime is not running.")
        self._running = False
        # Wait for all background tasks to finish.
        final_tasks_results = await asyncio.gather(*self._background_tasks, return_exceptions=True)
        for task_result in final_tasks_results:
            if isinstance(task_result, Exception):
                logger.error("Error in background task", exc_info=task_result)
        # Close the host connection.
        if self._host_connection is not None:
            try:
                await self._host_connection.close()
            except asyncio.CancelledError:
                pass
        # Cancel the read task.
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

    async def stop_when_signal(self, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)) -> None:
        """Stop the runtime when a signal is received."""
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler() -> None:
            logger.info("Received exit signal, shutting down gracefully...")
            shutdown_event.set()

        for sig in signals:
            loop.add_signal_handler(sig, signal_handler)

        # Wait for the signal to trigger the shutdown event.
        await shutdown_event.wait()

        # Stop the runtime.
        await self.stop()

    @property
    def _known_agent_names(self) -> Set[str]:
        return set(self._agent_factories.keys())

    async def _send_message(
        self,
        runtime_message: agent_worker_pb2.Message,
        send_type: Literal["send", "publish"],
        recipient: AgentId | TopicId,
        telemetry_metadata: Mapping[str, str],
    ) -> None:
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")
        with self._trace_helper.trace_block(send_type, recipient, parent=telemetry_metadata):
            await self._host_connection.send(runtime_message)

    async def send_message(
        self,
        message: Any,
        recipient: AgentId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any:
        # TODO: use message_id
        if not self._running:
            raise ValueError("Runtime must be running when sending message.")
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")
        data_type = self._serialization_registry.type_name(message)
        with self._trace_helper.trace_block(
            "create", recipient, parent=None, extraAttributes={"message_type": data_type}
        ):
            # create a new future for the result
            future = asyncio.get_event_loop().create_future()
            request_id = await self._get_new_request_id()
            self._pending_requests[request_id] = future
            serialized_message = self._serialization_registry.serialize(
                message, type_name=data_type, data_content_type=JSON_DATA_CONTENT_TYPE
            )
            telemetry_metadata = get_telemetry_grpc_metadata()
            runtime_message = agent_worker_pb2.Message(
                request=agent_worker_pb2.RpcRequest(
                    request_id=request_id,
                    target=agent_worker_pb2.AgentId(type=recipient.type, key=recipient.key),
                    source=agent_worker_pb2.AgentId(type=sender.type, key=sender.key) if sender is not None else None,
                    metadata=telemetry_metadata,
                    payload=agent_worker_pb2.Payload(
                        data_type=data_type,
                        data=serialized_message,
                        data_content_type=JSON_DATA_CONTENT_TYPE,
                    ),
                )
            )

            # TODO: Find a way to handle timeouts/errors
            task = asyncio.create_task(self._send_message(runtime_message, "send", recipient, telemetry_metadata))
            self._background_tasks.add(task)
            task.add_done_callback(self._raise_on_exception)
            task.add_done_callback(self._background_tasks.discard)
            return await future

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> None:
        if not self._running:
            raise ValueError("Runtime must be running when publishing message.")
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")
        if message_id is None:
            message_id = str(uuid.uuid4())

        message_type = self._serialization_registry.type_name(message)
        with self._trace_helper.trace_block(
            "create", topic_id, parent=None, extraAttributes={"message_type": message_type}
        ):
            serialized_message = self._serialization_registry.serialize(
                message, type_name=message_type, data_content_type=self._payload_serialization_format
            )

            sender_id = sender or AgentId("unknown", "unknown")
            attributes = {
                _constants.DATA_CONTENT_TYPE_ATTR: cloudevent_pb2.CloudEvent.CloudEventAttributeValue(
                    ce_string=self._payload_serialization_format
                ),
                _constants.DATA_SCHEMA_ATTR: cloudevent_pb2.CloudEvent.CloudEventAttributeValue(ce_string=message_type),
                _constants.AGENT_SENDER_TYPE_ATTR: cloudevent_pb2.CloudEvent.CloudEventAttributeValue(
                    ce_string=sender_id.type
                ),
                _constants.AGENT_SENDER_KEY_ATTR: cloudevent_pb2.CloudEvent.CloudEventAttributeValue(
                    ce_string=sender_id.key
                ),
                _constants.MESSAGE_KIND_ATTR: cloudevent_pb2.CloudEvent.CloudEventAttributeValue(
                    ce_string=_constants.MESSAGE_KIND_VALUE_PUBLISH
                ),
            }

            # If sending JSON we fill text_data with the serialized message
            # If sending Protobuf we fill proto_data with the serialized message
            # TODO: add an encoding field for serializer

            if self._payload_serialization_format == JSON_DATA_CONTENT_TYPE:
                runtime_message = agent_worker_pb2.Message(
                    cloudEvent=cloudevent_pb2.CloudEvent(
                        id=message_id,
                        spec_version="1.0",
                        type=topic_id.type,
                        source=topic_id.source,
                        attributes=attributes,
                        # TODO: use text, or proto fields appropriately
                        binary_data=serialized_message,
                    )
                )
            else:
                # We need to unpack the serialized proto back into an Any
                # TODO: find a way to prevent the roundtrip serialization
                any_proto = any_pb2.Any()
                any_proto.ParseFromString(serialized_message)
                runtime_message = agent_worker_pb2.Message(
                    cloudEvent=cloudevent_pb2.CloudEvent(
                        id=message_id,
                        spec_version="1.0",
                        type=topic_id.type,
                        source=topic_id.source,
                        attributes=attributes,
                        proto_data=any_proto,
                    )
                )

            telemetry_metadata = get_telemetry_grpc_metadata()
            task = asyncio.create_task(self._send_message(runtime_message, "publish", topic_id, telemetry_metadata))
            self._background_tasks.add(task)
            task.add_done_callback(self._raise_on_exception)
            task.add_done_callback(self._background_tasks.discard)

    async def save_state(self) -> Mapping[str, Any]:
        raise NotImplementedError("Saving state is not yet implemented.")

    async def load_state(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError("Loading state is not yet implemented.")

    async def agent_metadata(self, agent: AgentId) -> AgentMetadata:
        raise NotImplementedError("Agent metadata is not yet implemented.")

    async def agent_save_state(self, agent: AgentId) -> Mapping[str, Any]:
        raise NotImplementedError("Agent save_state is not yet implemented.")

    async def agent_load_state(self, agent: AgentId, state: Mapping[str, Any]) -> None:
        raise NotImplementedError("Agent load_state is not yet implemented.")

    async def _get_new_request_id(self) -> str:
        async with self._pending_requests_lock:
            self._next_request_id += 1
            return str(self._next_request_id)

    async def _process_request(self, request: agent_worker_pb2.RpcRequest) -> None:
        assert self._host_connection is not None
        recipient = AgentId(request.target.type, request.target.key)
        sender: AgentId | None = None
        if request.HasField("source"):
            sender = AgentId(request.source.type, request.source.key)
            logging.info(f"Processing request from {sender} to {recipient}")
        else:
            logging.info(f"Processing request from unknown source to {recipient}")

        # Deserialize the message.
        message = self._serialization_registry.deserialize(
            request.payload.data,
            type_name=request.payload.data_type,
            data_content_type=request.payload.data_content_type,
        )

        # Get the receiving agent and prepare the message context.
        rec_agent = await self._get_agent(recipient)
        message_context = MessageContext(
            sender=sender,
            topic_id=None,
            is_rpc=True,
            cancellation_token=CancellationToken(),
            message_id=request.request_id,
        )

        # Call the receiving agent.
        try:
            with MessageHandlerContext.populate_context(rec_agent.id):
                with self._trace_helper.trace_block(
                    "process",
                    rec_agent.id,
                    parent=request.metadata,
                    attributes={"request_id": request.request_id},
                    extraAttributes={"message_type": request.payload.data_type},
                ):
                    result = await rec_agent.on_message(message, ctx=message_context)
        except BaseException as e:
            response_message = agent_worker_pb2.Message(
                response=agent_worker_pb2.RpcResponse(
                    request_id=request.request_id,
                    error=str(e),
                    metadata=get_telemetry_grpc_metadata(),
                ),
            )
            # Send the error response.
            await self._host_connection.send(response_message)
            return

        # Serialize the result.
        result_type = self._serialization_registry.type_name(result)
        serialized_result = self._serialization_registry.serialize(
            result, type_name=result_type, data_content_type=JSON_DATA_CONTENT_TYPE
        )

        # Create the response message.
        response_message = agent_worker_pb2.Message(
            response=agent_worker_pb2.RpcResponse(
                request_id=request.request_id,
                payload=agent_worker_pb2.Payload(
                    data_type=result_type,
                    data=serialized_result,
                    data_content_type=JSON_DATA_CONTENT_TYPE,
                ),
                metadata=get_telemetry_grpc_metadata(),
            )
        )

        # Send the response.
        await self._host_connection.send(response_message)

    async def _process_response(self, response: agent_worker_pb2.RpcResponse) -> None:
        with self._trace_helper.trace_block(
            "ack",
            None,
            parent=response.metadata,
            attributes={"request_id": response.request_id},
            extraAttributes={"message_type": response.payload.data_type},
        ):
            # Deserialize the result.
            result = self._serialization_registry.deserialize(
                response.payload.data,
                type_name=response.payload.data_type,
                data_content_type=response.payload.data_content_type,
            )
            # Get the future and set the result.
            future = self._pending_requests.pop(response.request_id)
            if len(response.error) > 0:
                future.set_exception(Exception(response.error))
            else:
                future.set_result(result)

    async def _process_event(self, event: cloudevent_pb2.CloudEvent) -> None:
        event_attributes = event.attributes
        sender: AgentId | None = None
        if (
            _constants.AGENT_SENDER_TYPE_ATTR in event_attributes
            and _constants.AGENT_SENDER_KEY_ATTR in event_attributes
        ):
            sender = AgentId(
                event_attributes[_constants.AGENT_SENDER_TYPE_ATTR].ce_string,
                event_attributes[_constants.AGENT_SENDER_KEY_ATTR].ce_string,
            )
        topic_id = TopicId(event.type, event.source)
        # Get the recipients for the topic.
        recipients = await self._subscription_manager.get_subscribed_recipients(topic_id)

        message_content_type = event_attributes[_constants.DATA_CONTENT_TYPE_ATTR].ce_string
        message_type = event_attributes[_constants.DATA_SCHEMA_ATTR].ce_string

        if message_content_type == JSON_DATA_CONTENT_TYPE:
            message = self._serialization_registry.deserialize(
                event.binary_data, type_name=message_type, data_content_type=message_content_type
            )
        elif message_content_type == PROTOBUF_DATA_CONTENT_TYPE:
            # TODO: find a way to prevent the roundtrip serialization
            proto_binary_data = event.proto_data.SerializeToString()
            message = self._serialization_registry.deserialize(
                proto_binary_data, type_name=message_type, data_content_type=message_content_type
            )
        else:
            raise ValueError(f"Unsupported message content type: {message_content_type}")

        # TODO: dont read these values in the runtime
        topic_type_suffix = topic_id.type.split(":", maxsplit=1)[1] if ":" in topic_id.type else ""
        is_rpc = topic_type_suffix == _constants.MESSAGE_KIND_VALUE_RPC_REQUEST
        is_marked_rpc_type = (
            _constants.MESSAGE_KIND_ATTR in event_attributes
            and event_attributes[_constants.MESSAGE_KIND_ATTR].ce_string == _constants.MESSAGE_KIND_VALUE_RPC_REQUEST
        )
        if is_rpc and not is_marked_rpc_type:
            warnings.warn("Received RPC request with topic type suffix but not marked as RPC request.", stacklevel=2)

        # Send the message to each recipient.
        responses: List[Awaitable[Any]] = []
        for agent_id in recipients:
            if agent_id == sender:
                continue
            message_context = MessageContext(
                sender=sender,
                topic_id=topic_id,
                is_rpc=is_rpc,
                cancellation_token=CancellationToken(),
                message_id=event.id,
            )
            agent = await self._get_agent(agent_id)
            with MessageHandlerContext.populate_context(agent.id):

                def stringify_attributes(
                    attributes: Mapping[str, cloudevent_pb2.CloudEvent.CloudEventAttributeValue],
                ) -> Mapping[str, str]:
                    result: Dict[str, str] = {}
                    for key, value in attributes.items():
                        item = None
                        match value.WhichOneof("attr"):
                            case "ce_boolean":
                                item = str(value.ce_boolean)
                            case "ce_integer":
                                item = str(value.ce_integer)
                            case "ce_string":
                                item = value.ce_string
                            case "ce_bytes":
                                item = str(value.ce_bytes)
                            case "ce_uri":
                                item = value.ce_uri
                            case "ce_uri_ref":
                                item = value.ce_uri_ref
                            case "ce_timestamp":
                                item = str(value.ce_timestamp)
                            case _:
                                raise ValueError("Unknown attribute kind")
                        result[key] = item

                    return result

                async def send_message(agent: Agent, message_context: MessageContext) -> Any:
                    with self._trace_helper.trace_block(
                        "process",
                        agent.id,
                        parent=stringify_attributes(event.attributes),
                        extraAttributes={"message_type": message_type},
                    ):
                        await agent.on_message(message, ctx=message_context)

                future = send_message(agent, message_context)
            responses.append(future)
        # Wait for all responses.
        try:
            await asyncio.gather(*responses)
        except BaseException as e:
            logger.error("Error handling event", exc_info=e)

    async def _register_agent_type(self, agent_type: str) -> None:
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")
        message = agent_worker_pb2.RegisterAgentTypeRequest(type=agent_type)
        _response: agent_worker_pb2.RegisterAgentTypeResponse = await self._host_connection.stub.RegisterAgent(
            message, metadata=self._host_connection.metadata
        )

    async def register_factory(
        self,
        type: str | AgentType,
        agent_factory: Callable[[], T | Awaitable[T]],
        *,
        expected_class: type[T] | None = None,
    ) -> AgentType:
        if isinstance(type, str):
            type = AgentType(type)

        if type.type in self._agent_factories:
            raise ValueError(f"Agent with type {type} already exists.")
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")

        async def factory_wrapper() -> T:
            maybe_agent_instance = agent_factory()
            if inspect.isawaitable(maybe_agent_instance):
                agent_instance = await maybe_agent_instance
            else:
                agent_instance = maybe_agent_instance

            if expected_class is not None and type_func_alias(agent_instance) != expected_class:
                raise ValueError("Factory registered using the wrong type.")

            return agent_instance

        self._agent_factories[type.type] = factory_wrapper
        # Send the registration request message to the host.
        await self._register_agent_type(type.type)

        return type

    async def register_agent_instance(
        self,
        agent_instance: Agent,
        agent_id: AgentId,
    ) -> AgentId:
        def agent_factory() -> Agent:
            raise RuntimeError(
                "Agent factory was invoked for an agent instance that was not registered. This is likely due to the agent type being incorrectly subscribed to a topic. If this exception occurs when publishing a message to the DefaultTopicId, then it is likely that `skip_class_subscriptions` needs to be turned off when registering the agent."
            )

        if agent_id in self._instantiated_agents:
            raise ValueError(f"Agent with id {agent_id} already exists.")

        if agent_id.type not in self._agent_factories:
            self._agent_factories[agent_id.type] = agent_factory
            await self._register_agent_type(agent_id.type)
            self._agent_instance_types[agent_id.type] = type_func_alias(agent_instance)
        else:
            if self._agent_factories[agent_id.type].__code__ != agent_factory.__code__:
                raise ValueError("Agent factories and agent instances cannot be registered to the same type.")
            if self._agent_instance_types[agent_id.type] != type_func_alias(agent_instance):
                raise ValueError("Agent instances must be the same object type.")

        await agent_instance.bind_id_and_runtime(id=agent_id, runtime=self)
        self._instantiated_agents[agent_id] = agent_instance
        return agent_id

    async def _invoke_agent_factory(
        self,
        agent_factory: Callable[[], T | Awaitable[T]] | Callable[[AgentRuntime, AgentId], T | Awaitable[T]],
        agent_id: AgentId,
    ) -> T:
        with AgentInstantiationContext.populate_context((self, agent_id)):
            if len(inspect.signature(agent_factory).parameters) == 0:
                factory_one = cast(Callable[[], T], agent_factory)
                agent = factory_one()
            elif len(inspect.signature(agent_factory).parameters) == 2:
                warnings.warn(
                    "Agent factories that take two arguments are deprecated. Use AgentInstantiationContext instead. Two arg factories will be removed in a future version.",
                    stacklevel=2,
                )
                factory_two = cast(Callable[[AgentRuntime, AgentId], T], agent_factory)
                agent = factory_two(self, agent_id)
            else:
                raise ValueError("Agent factory must take 0 or 2 arguments.")

            if inspect.isawaitable(agent):
                agent = cast(T, await agent)

        return agent

    async def _get_agent(self, agent_id: AgentId) -> Agent:
        if agent_id in self._instantiated_agents:
            return self._instantiated_agents[agent_id]

        if agent_id.type not in self._agent_factories:
            raise ValueError(f"Agent with name {agent_id.type} not found.")

        agent_factory = self._agent_factories[agent_id.type]
        agent = await self._invoke_agent_factory(agent_factory, agent_id)
        self._instantiated_agents[agent_id] = agent
        return agent

    # TODO: uncomment out the following type ignore when this is fixed in mypy: https://github.com/python/mypy/issues/3737
    async def try_get_underlying_agent_instance(self, id: AgentId, type: Type[T] = Agent) -> T:  # type: ignore[assignment]
        if id.type not in self._agent_factories:
            raise LookupError(f"Agent with name {id.type} not found.")

        # TODO: check if remote
        agent_instance = await self._get_agent(id)

        if not isinstance(agent_instance, type):
            raise TypeError(f"Agent with name {id.type} is not of type {type.__name__}")

        return agent_instance

    async def add_subscription(self, subscription: Subscription) -> None:
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")

        message = agent_worker_pb2.AddSubscriptionRequest(subscription=subscription_to_proto(subscription))
        _response: agent_worker_pb2.AddSubscriptionResponse = await self._host_connection.stub.AddSubscription(
            message, metadata=self._host_connection.metadata
        )

        # Add to local subscription manager.
        await self._subscription_manager.add_subscription(subscription)

    async def remove_subscription(self, id: str) -> None:
        if self._host_connection is None:
            raise RuntimeError("Host connection is not set.")

        message = agent_worker_pb2.RemoveSubscriptionRequest(id=id)
        _response: agent_worker_pb2.RemoveSubscriptionResponse = await self._host_connection.stub.RemoveSubscription(
            message, metadata=self._host_connection.metadata
        )

        await self._subscription_manager.remove_subscription(id)

    async def get(
        self, id_or_type: AgentId | AgentType | str, /, key: str = "default", *, lazy: bool = True
    ) -> AgentId:
        return await get_impl(
            id_or_type=id_or_type,
            key=key,
            lazy=lazy,
            instance_getter=self._get_agent,
        )

    def add_message_serializer(self, serializer: MessageSerializer[Any] | Sequence[MessageSerializer[Any]]) -> None:
        self._serialization_registry.add_serializer(serializer)

# From grpc/_worker_runtime.py
def stub(self) -> Any:
        return self._stub

# From grpc/_worker_runtime.py
def signal_handler() -> None:
            logger.info("Received exit signal, shutting down gracefully...")
            shutdown_event.set()

# From grpc/_worker_runtime.py
def stringify_attributes(
                    attributes: Mapping[str, cloudevent_pb2.CloudEvent.CloudEventAttributeValue],
                ) -> Mapping[str, str]:
                    result: Dict[str, str] = {}
                    for key, value in attributes.items():
                        item = None
                        match value.WhichOneof("attr"):
                            case "ce_boolean":
                                item = str(value.ce_boolean)
                            case "ce_integer":
                                item = str(value.ce_integer)
                            case "ce_string":
                                item = value.ce_string
                            case "ce_bytes":
                                item = str(value.ce_bytes)
                            case "ce_uri":
                                item = value.ce_uri
                            case "ce_uri_ref":
                                item = value.ce_uri_ref
                            case "ce_timestamp":
                                item = str(value.ce_timestamp)
                            case _:
                                raise ValueError("Unknown attribute kind")
                        result[key] = item

                    return result

from _worker_runtime_host_servicer import GrpcWorkerAgentRuntimeHostServicer

# From grpc/_worker_runtime_host.py
class GrpcWorkerAgentRuntimeHost:
    def __init__(self, address: str, extra_grpc_config: Optional[ChannelArgumentType] = None) -> None:
        self._server = grpc.aio.server(options=extra_grpc_config)
        self._servicer = GrpcWorkerAgentRuntimeHostServicer()
        agent_worker_pb2_grpc.add_AgentRpcServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(address)
        self._address = address
        self._serve_task: asyncio.Task[None] | None = None

    async def _serve(self) -> None:
        await self._server.start()
        logger.info(f"Server started at {self._address}.")
        await self._server.wait_for_termination()

    def start(self) -> None:
        """Start the server in a background task."""
        if self._serve_task is not None:
            raise RuntimeError("Host runtime is already started.")
        self._serve_task = asyncio.create_task(self._serve())

    async def stop(self, grace: int = 5) -> None:
        """Stop the server."""
        if self._serve_task is None:
            raise RuntimeError("Host runtime is not started.")
        await self._server.stop(grace=grace)
        self._serve_task.cancel()
        try:
            await self._serve_task
        except asyncio.CancelledError:
            pass
        logger.info("Server stopped.")
        self._serve_task = None

    async def stop_when_signal(
        self, grace: int = 5, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)
    ) -> None:
        """Stop the server when a signal is received."""
        if self._serve_task is None:
            raise RuntimeError("Host runtime is not started.")
        # Set up signal handling for graceful shutdown.
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler() -> None:
            logger.info("Received exit signal, shutting down gracefully...")
            shutdown_event.set()

        for sig in signals:
            loop.add_signal_handler(sig, signal_handler)

        # Wait for the signal to trigger the shutdown event.
        await shutdown_event.wait()

        # Shutdown the server.
        await self.stop(grace=grace)

from  import agent_worker_pb2

# From protos/agent_worker_pb2_grpc.py
class AgentRpcStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.OpenChannel = channel.stream_stream(
                '/agents.AgentRpc/OpenChannel',
                request_serializer=agent__worker__pb2.Message.SerializeToString,
                response_deserializer=agent__worker__pb2.Message.FromString,
                _registered_method=True)
        self.OpenControlChannel = channel.stream_stream(
                '/agents.AgentRpc/OpenControlChannel',
                request_serializer=agent__worker__pb2.ControlMessage.SerializeToString,
                response_deserializer=agent__worker__pb2.ControlMessage.FromString,
                _registered_method=True)
        self.RegisterAgent = channel.unary_unary(
                '/agents.AgentRpc/RegisterAgent',
                request_serializer=agent__worker__pb2.RegisterAgentTypeRequest.SerializeToString,
                response_deserializer=agent__worker__pb2.RegisterAgentTypeResponse.FromString,
                _registered_method=True)
        self.AddSubscription = channel.unary_unary(
                '/agents.AgentRpc/AddSubscription',
                request_serializer=agent__worker__pb2.AddSubscriptionRequest.SerializeToString,
                response_deserializer=agent__worker__pb2.AddSubscriptionResponse.FromString,
                _registered_method=True)
        self.RemoveSubscription = channel.unary_unary(
                '/agents.AgentRpc/RemoveSubscription',
                request_serializer=agent__worker__pb2.RemoveSubscriptionRequest.SerializeToString,
                response_deserializer=agent__worker__pb2.RemoveSubscriptionResponse.FromString,
                _registered_method=True)
        self.GetSubscriptions = channel.unary_unary(
                '/agents.AgentRpc/GetSubscriptions',
                request_serializer=agent__worker__pb2.GetSubscriptionsRequest.SerializeToString,
                response_deserializer=agent__worker__pb2.GetSubscriptionsResponse.FromString,
                _registered_method=True)

# From protos/agent_worker_pb2_grpc.py
class AgentRpcServicer(object):
    """Missing associated documentation comment in .proto file."""

    def OpenChannel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def OpenControlChannel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterAgent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AddSubscription(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoveSubscription(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSubscriptions(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

# From protos/agent_worker_pb2_grpc.py
class AgentRpc(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def OpenChannel(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/agents.AgentRpc/OpenChannel',
            agent__worker__pb2.Message.SerializeToString,
            agent__worker__pb2.Message.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def OpenControlChannel(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/agents.AgentRpc/OpenControlChannel',
            agent__worker__pb2.ControlMessage.SerializeToString,
            agent__worker__pb2.ControlMessage.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RegisterAgent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/agents.AgentRpc/RegisterAgent',
            agent__worker__pb2.RegisterAgentTypeRequest.SerializeToString,
            agent__worker__pb2.RegisterAgentTypeResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def AddSubscription(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/agents.AgentRpc/AddSubscription',
            agent__worker__pb2.AddSubscriptionRequest.SerializeToString,
            agent__worker__pb2.AddSubscriptionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RemoveSubscription(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/agents.AgentRpc/RemoveSubscription',
            agent__worker__pb2.RemoveSubscriptionRequest.SerializeToString,
            agent__worker__pb2.RemoveSubscriptionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetSubscriptions(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/agents.AgentRpc/GetSubscriptions',
            agent__worker__pb2.GetSubscriptionsRequest.SerializeToString,
            agent__worker__pb2.GetSubscriptionsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

# From protos/agent_worker_pb2_grpc.py
def add_AgentRpcServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'OpenChannel': grpc.stream_stream_rpc_method_handler(
                    servicer.OpenChannel,
                    request_deserializer=agent__worker__pb2.Message.FromString,
                    response_serializer=agent__worker__pb2.Message.SerializeToString,
            ),
            'OpenControlChannel': grpc.stream_stream_rpc_method_handler(
                    servicer.OpenControlChannel,
                    request_deserializer=agent__worker__pb2.ControlMessage.FromString,
                    response_serializer=agent__worker__pb2.ControlMessage.SerializeToString,
            ),
            'RegisterAgent': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterAgent,
                    request_deserializer=agent__worker__pb2.RegisterAgentTypeRequest.FromString,
                    response_serializer=agent__worker__pb2.RegisterAgentTypeResponse.SerializeToString,
            ),
            'AddSubscription': grpc.unary_unary_rpc_method_handler(
                    servicer.AddSubscription,
                    request_deserializer=agent__worker__pb2.AddSubscriptionRequest.FromString,
                    response_serializer=agent__worker__pb2.AddSubscriptionResponse.SerializeToString,
            ),
            'RemoveSubscription': grpc.unary_unary_rpc_method_handler(
                    servicer.RemoveSubscription,
                    request_deserializer=agent__worker__pb2.RemoveSubscriptionRequest.FromString,
                    response_serializer=agent__worker__pb2.RemoveSubscriptionResponse.SerializeToString,
            ),
            'GetSubscriptions': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSubscriptions,
                    request_deserializer=agent__worker__pb2.GetSubscriptionsRequest.FromString,
                    response_serializer=agent__worker__pb2.GetSubscriptionsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'agents.AgentRpc', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('agents.AgentRpc', rpc_method_handlers)

# From protos/agent_worker_pb2_grpc.py
def OpenChannel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

# From protos/agent_worker_pb2_grpc.py
def OpenControlChannel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

# From protos/agent_worker_pb2_grpc.py
def RegisterAgent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

# From protos/agent_worker_pb2_grpc.py
def AddSubscription(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

# From protos/agent_worker_pb2_grpc.py
def RemoveSubscription(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

# From protos/agent_worker_pb2_grpc.py
def GetSubscriptions(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

from  import cloudevent_pb2

from typing import Iterable
import aiofiles
from autogen_core.tools import Tool
from openai import NOT_GIVEN
from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI
from openai import NotGiven
from openai.pagination import AsyncCursorPage
from openai.resources.beta.threads import AsyncMessages
from openai.resources.beta.threads import AsyncRuns
from openai.resources.beta.threads import AsyncThreads
from openai.types import FileObject
from openai.types.beta import thread_update_params
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_response_format_option_param import AssistantResponseFormatOptionParam
from openai.types.beta.assistant_tool_param import AssistantToolParam
from openai.types.beta.code_interpreter_tool_param import CodeInterpreterToolParam
from openai.types.beta.file_search_tool_param import FileSearchToolParam
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.thread import Thread
from openai.types.beta.thread import ToolResources
from openai.types.beta.thread import ToolResourcesCodeInterpreter
from openai.types.beta.threads import Message
from openai.types.beta.threads import MessageDeleted
from openai.types.beta.threads import Run
from openai.types.beta.threads.image_url_content_block_param import ImageURLContentBlockParam
from openai.types.beta.threads.image_url_param import ImageURLParam
from openai.types.beta.threads.message_content_part_param import MessageContentPartParam
from openai.types.beta.threads.text_content_block_param import TextContentBlockParam
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.vector_store import VectorStore

# From openai/_openai_assistant_agent.py
class OpenAIAssistantAgentState(BaseModel):
    type: str = Field(default="OpenAIAssistantAgentState")
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    initial_message_ids: List[str] = Field(default_factory=list)
    vector_store_id: Optional[str] = None
    uploaded_file_ids: List[str] = Field(default_factory=list)

# From openai/_openai_assistant_agent.py
class OpenAIAssistantAgent(BaseChatAgent):
    """An agent implementation that uses the Assistant API to generate responses.

    Installation:

    .. code-block:: bash

        pip install "autogen-ext[openai]"  # For OpenAI Assistant
        # pip install "autogen-ext[openai,azure]"  # For Azure OpenAI Assistant


    This agent leverages the Assistant API to create AI assistants with capabilities like:

    * Code interpretation and execution
    * File handling and search
    * Custom function calling
    * Multi-turn conversations

    The agent maintains a thread of conversation and can use various tools including

    * Code interpreter: For executing code and working with files
    * File search: For searching through uploaded documents
    * Custom functions: For extending capabilities with user-defined tools

    Key Features:

    * Supports multiple file formats including code, documents, images
    * Can handle up to 128 tools per assistant
    * Maintains conversation context in threads
    * Supports file uploads for code interpreter and search
    * Vector store integration for efficient file search
    * Automatic file parsing and embedding

    You can use an existing thread or assistant by providing the `thread_id` or `assistant_id` parameters.

    Examples:

        Use the assistant to analyze data in a CSV file:

        .. code-block:: python

            from openai import AsyncOpenAI
            from autogen_core import CancellationToken
            import asyncio
            from autogen_ext.agents.openai import OpenAIAssistantAgent
            from autogen_agentchat.messages import TextMessage


            async def example():
                cancellation_token = CancellationToken()

                # Create an OpenAI client
                client = AsyncOpenAI(api_key="your-api-key", base_url="your-base-url")

                # Create an assistant with code interpreter
                assistant = OpenAIAssistantAgent(
                    name="PythonHelper",
                    description="Helps with Python programming",
                    client=client,
                    model="gpt-4",
                    instructions="You are a helpful Python programming assistant.",
                    tools=["code_interpreter"],
                )

                # Upload files for the assistant to use
                await assistant.on_upload_for_code_interpreter("data.csv", cancellation_token)

                # Get response from the assistant
                response = await assistant.on_messages(
                    [TextMessage(source="user", content="Analyze the data in data.csv")], cancellation_token
                )

                print(response)

                # Clean up resources
                await assistant.delete_uploaded_files(cancellation_token)
                await assistant.delete_assistant(cancellation_token)


            asyncio.run(example())

        Use Azure OpenAI Assistant with AAD authentication:

        .. code-block:: python

            from openai import AsyncAzureOpenAI
            import asyncio
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            from autogen_core import CancellationToken
            from autogen_ext.agents.openai import OpenAIAssistantAgent
            from autogen_agentchat.messages import TextMessage


            async def example():
                cancellation_token = CancellationToken()

                # Create an Azure OpenAI client
                token_provider = get_bearer_token_provider(DefaultAzureCredential())
                client = AsyncAzureOpenAI(
                    azure_deployment="YOUR_AZURE_DEPLOYMENT",
                    api_version="YOUR_API_VERSION",
                    azure_endpoint="YOUR_AZURE_ENDPOINT",
                    azure_ad_token_provider=token_provider,
                )

                # Create an assistant with code interpreter
                assistant = OpenAIAssistantAgent(
                    name="PythonHelper",
                    description="Helps with Python programming",
                    client=client,
                    model="gpt-4o",
                    instructions="You are a helpful Python programming assistant.",
                    tools=["code_interpreter"],
                )

                # Get response from the assistant
                response = await assistant.on_messages([TextMessage(source="user", content="Hello.")], cancellation_token)

                print(response)

                # Clean up resources
                await assistant.delete_assistant(cancellation_token)


            asyncio.run(example())

    Args:
        name (str): Name of the assistant
        description (str): Description of the assistant's purpose
        client (AsyncOpenAI | AsyncAzureOpenAI): OpenAI client or Azure OpenAI client instance
        model (str): Model to use (e.g. "gpt-4")
        instructions (str): System instructions for the assistant
        tools (Optional[Iterable[Union[Literal["code_interpreter", "file_search"], Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]]]]): Tools the assistant can use
        assistant_id (Optional[str]): ID of existing assistant to use
        thread_id (Optional[str]): ID of existing thread to use
        metadata (Optional[Dict[str, str]]): Additional metadata for the assistant.
        response_format (Optional[AssistantResponseFormatOptionParam]): Response format settings
        temperature (Optional[float]): Temperature for response generation
        tool_resources (Optional[ToolResources]): Additional tool configuration
        top_p (Optional[float]): Top p sampling parameter
    """

    def __init__(
        self,
        name: str,
        description: str,
        client: AsyncOpenAI | AsyncAzureOpenAI,
        model: str,
        instructions: str,
        tools: Optional[
            Iterable[
                Union[
                    Literal["code_interpreter", "file_search"],
                    Tool | Callable[..., Any] | Callable[..., Awaitable[Any]],
                ]
            ]
        ] = None,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        response_format: Optional["AssistantResponseFormatOptionParam"] = None,
        temperature: Optional[float] = None,
        tool_resources: Optional["ToolResources"] = None,
        top_p: Optional[float] = None,
    ) -> None:
        if isinstance(client, ChatCompletionClient):
            raise ValueError(
                "Incorrect client passed to OpenAIAssistantAgent. Please use an OpenAI AsyncClient instance instead of an AutoGen ChatCompletionClient instance."
            )

        super().__init__(name, description)
        if tools is None:
            tools = []

        # Store original tools and converted tools separately
        self._original_tools: List[Tool] = []
        converted_tools: List["AssistantToolParam"] = []
        for tool in tools:
            if isinstance(tool, str):
                if tool == "code_interpreter":
                    converted_tools.append(CodeInterpreterToolParam(type="code_interpreter"))
                elif tool == "file_search":
                    converted_tools.append(FileSearchToolParam(type="file_search"))
            elif isinstance(tool, Tool):
                self._original_tools.append(tool)
                converted_tools.append(_convert_tool_to_function_param(tool))
            elif callable(tool):
                if hasattr(tool, "__doc__") and tool.__doc__ is not None:
                    description = tool.__doc__
                else:
                    description = ""
                function_tool = FunctionTool(tool, description=description)
                self._original_tools.append(function_tool)
                converted_tools.append(_convert_tool_to_function_param(function_tool))
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        self._client = client
        self._assistant: Optional["Assistant"] = None
        self._thread: Optional["Thread"] = None
        self._init_thread_id = thread_id
        self._model = model
        self._instructions = instructions
        self._api_tools = converted_tools
        self._assistant_id = assistant_id
        self._metadata = metadata
        self._response_format = response_format
        self._temperature = temperature
        self._tool_resources = tool_resources
        self._top_p = top_p
        self._vector_store_id: Optional[str] = None
        self._uploaded_file_ids: List[str] = []

        # Variables to track initial state
        self._initial_message_ids: Set[str] = set()
        self._initial_state_retrieved: bool = False

    async def _ensure_initialized(self) -> None:
        """Ensure assistant and thread are created."""
        if self._assistant is None:
            if self._assistant_id:
                self._assistant = await self._client.beta.assistants.retrieve(assistant_id=self._assistant_id)  # type: ignore[reportDeprecated]
            else:
                self._assistant = await self._client.beta.assistants.create(  # type: ignore[reportDeprecated]
                    model=self._model,
                    description=self.description,
                    instructions=self._instructions,
                    tools=self._api_tools,
                    metadata=self._metadata,
                    response_format=self._response_format if self._response_format else NOT_GIVEN,  # type: ignore
                    temperature=self._temperature,
                    tool_resources=self._tool_resources if self._tool_resources else NOT_GIVEN,  # type: ignore
                    top_p=self._top_p,
                )

        if self._thread is None:
            if self._init_thread_id:
                self._thread = await self._client.beta.threads.retrieve(thread_id=self._init_thread_id)  # type: ignore[reportDeprecated]
            else:
                self._thread = await self._client.beta.threads.create()  # type: ignore[reportDeprecated]

        # Retrieve initial state only once
        if not self._initial_state_retrieved:
            await self._retrieve_initial_state()
            self._initial_state_retrieved = True

    async def _retrieve_initial_state(self) -> None:
        """Retrieve and store the initial state of messages and runs."""
        # Retrieve all initial message IDs
        initial_message_ids: Set[str] = set()
        after: str | NotGiven = NOT_GIVEN
        while True:
            msgs: AsyncCursorPage[Message] = await self._client.beta.threads.messages.list(  # type: ignore[reportDeprecated]
                self._thread_id, after=after, order="asc", limit=100
            )
            for msg in msgs.data:
                initial_message_ids.add(msg.id)
            if not msgs.has_next_page():
                break
            after = msgs.data[-1].id
        self._initial_message_ids = initial_message_ids

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """The types of messages that the assistant agent produces."""
        return (TextMessage,)

    @property
    def threads(self) -> AsyncThreads:
        return self._client.beta.threads

    @property
    def runs(self) -> AsyncRuns:
        return self._client.beta.threads.runs

    @property
    def messages(self) -> AsyncMessages:
        return self._client.beta.threads.messages

    @property
    def _get_assistant_id(self) -> str:
        if self._assistant is None:
            raise ValueError("Assistant not initialized")
        return self._assistant.id

    @property
    def _thread_id(self) -> str:
        if self._thread is None:
            raise ValueError("Thread not initialized")
        return self._thread.id

    async def _execute_tool_call(self, tool_call: FunctionCall, cancellation_token: CancellationToken) -> str:
        """Execute a tool call and return the result."""
        if not self._original_tools:
            raise ValueError("No tools are available.")
        tool = next((t for t in self._original_tools if t.name == tool_call.name), None)
        if tool is None:
            raise ValueError(f"The tool '{tool_call.name}' is not available.")
        arguments = json.loads(tool_call.arguments)
        result = await tool.run_json(arguments, cancellation_token, call_id=tool_call.id)
        return tool.return_value_as_string(result)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Handle incoming messages and return a response."""

        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Handle incoming messages and return a response."""
        await self._ensure_initialized()

        # Process all messages in sequence
        for message in messages:
            await self.handle_incoming_message(message, cancellation_token)

        # Inner messages for tool calls
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []

        # Create and start a run
        run: Run = await cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.runs.create(  # type: ignore[reportDeprecated]
                    thread_id=self._thread_id,
                    assistant_id=self._get_assistant_id,
                )
            )
        )

        # Wait for run completion by polling
        while True:
            run = await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.runs.retrieve(  # type: ignore[reportDeprecated]
                        thread_id=self._thread_id,
                        run_id=run.id,
                    )
                )
            )

            if run.status == "failed":
                raise ValueError(f"Run failed: {run.last_error}")

            # If the run requires action (function calls), execute tools and continue
            if run.status == "requires_action" and run.required_action is not None:
                tool_calls: List[FunctionCall] = []
                for required_tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    if required_tool_call.type == "function":
                        tool_calls.append(
                            FunctionCall(
                                id=required_tool_call.id,
                                name=required_tool_call.function.name,
                                arguments=required_tool_call.function.arguments,
                            )
                        )

                # Add tool call message to inner messages
                tool_call_msg = ToolCallRequestEvent(source=self.name, content=tool_calls)
                inner_messages.append(tool_call_msg)
                event_logger.debug(tool_call_msg)
                yield tool_call_msg

                # Execute tool calls and get results
                tool_outputs: List[FunctionExecutionResult] = []
                for tool_call in tool_calls:
                    try:
                        result = await self._execute_tool_call(tool_call, cancellation_token)
                        is_error = False
                    except Exception as e:
                        result = f"Error: {e}"
                        is_error = True
                    tool_outputs.append(
                        FunctionExecutionResult(
                            content=result, call_id=tool_call.id, is_error=is_error, name=tool_call.name
                        )
                    )

                # Add tool result message to inner messages
                tool_result_msg = ToolCallExecutionEvent(source=self.name, content=tool_outputs)
                inner_messages.append(tool_result_msg)
                event_logger.debug(tool_result_msg)
                yield tool_result_msg

                # Submit tool outputs back to the run
                run = await cancellation_token.link_future(
                    asyncio.ensure_future(
                        self._client.beta.threads.runs.submit_tool_outputs(  # type: ignore[reportDeprecated]
                            thread_id=self._thread_id,
                            run_id=run.id,
                            tool_outputs=[{"tool_call_id": t.call_id, "output": t.content} for t in tool_outputs],
                        )
                    )
                )
                continue

            if run.status == "completed":
                break

            await asyncio.sleep(0.5)

        # Get messages after run completion
        assistant_messages: AsyncCursorPage[Message] = await cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.messages.list(thread_id=self._thread_id, order="desc", limit=1)  # type: ignore[reportDeprecated]
            )
        )

        if not assistant_messages.data:
            raise ValueError("No messages received from assistant")

        # Get the last message's content
        last_message = assistant_messages.data[0]
        if not last_message.content:
            raise ValueError(f"No content in the last message: {last_message}")

        # Extract text content
        text_content = [content for content in last_message.content if content.type == "text"]
        if not text_content:
            raise ValueError(f"Expected text content in the last message: {last_message.content}")

        # Return the assistant's response as a Response with inner messages
        chat_message = TextMessage(source=self.name, content=text_content[0].text.value)
        yield Response(chat_message=chat_message, inner_messages=inner_messages)

    async def handle_incoming_message(self, message: BaseChatMessage, cancellation_token: CancellationToken) -> None:
        """Handle regular text messages by adding them to the thread."""
        content: str | List[MessageContentPartParam] | None = None
        llm_message = message.to_model_message()
        if isinstance(llm_message.content, str):
            content = llm_message.content
        else:
            content = []
            for c in llm_message.content:
                if isinstance(c, str):
                    content.append(TextContentBlockParam(text=c, type="text"))
                elif isinstance(c, Image):
                    content.append(ImageURLContentBlockParam(image_url=ImageURLParam(url=c.data_uri), type="image_url"))
                else:
                    raise ValueError(f"Unsupported content type: {type(c)} in {message}")
        await cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.messages.create(  # type: ignore[reportDeprecated]
                    thread_id=self._thread_id,
                    content=content,
                    role="user",
                )
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Handle reset command by deleting new messages and runs since initialization."""
        await self._ensure_initialized()

        # Retrieve all message IDs in the thread
        new_message_ids: List[str] = []
        after: str | NotGiven = NOT_GIVEN
        while True:
            msgs: AsyncCursorPage[Message] = await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.messages.list(self._thread_id, after=after, order="asc", limit=100)  # type: ignore[reportDeprecated]
                )
            )
            for msg in msgs.data:
                if msg.id not in self._initial_message_ids:
                    new_message_ids.append(msg.id)
            if not msgs.has_next_page():
                break
            after = msgs.data[-1].id

        # Delete new messages
        for msg_id in new_message_ids:
            status: MessageDeleted = await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.messages.delete(message_id=msg_id, thread_id=self._thread_id)  # type: ignore[reportDeprecated]
                )
            )
            assert status.deleted is True

    async def _upload_files(self, file_paths: str | Iterable[str], cancellation_token: CancellationToken) -> List[str]:
        """Upload files and return their IDs."""
        await self._ensure_initialized()

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        file_ids: List[str] = []
        for file_path in file_paths:
            async with aiofiles.open(file_path, mode="rb") as f:
                file_content = await cancellation_token.link_future(asyncio.ensure_future(f.read()))
            file_name = os.path.basename(file_path)

            file: FileObject = await cancellation_token.link_future(
                asyncio.ensure_future(self._client.files.create(file=(file_name, file_content), purpose="assistants"))
            )
            file_ids.append(file.id)
            self._uploaded_file_ids.append(file.id)

        return file_ids

    async def on_upload_for_code_interpreter(
        self, file_paths: str | Iterable[str], cancellation_token: CancellationToken
    ) -> None:
        """Handle file uploads for the code interpreter."""
        await self._ensure_initialized()

        file_ids = await self._upload_files(file_paths, cancellation_token)

        # Update thread with the new files
        thread = await cancellation_token.link_future(
            asyncio.ensure_future(self._client.beta.threads.retrieve(thread_id=self._thread_id))  # type: ignore[reportDeprecated]
        )
        tool_resources: ToolResources = thread.tool_resources or ToolResources()
        code_interpreter: ToolResourcesCodeInterpreter = (
            tool_resources.code_interpreter or ToolResourcesCodeInterpreter()
        )
        existing_file_ids: List[str] = code_interpreter.file_ids or []
        existing_file_ids.extend(file_ids)
        tool_resources.code_interpreter = ToolResourcesCodeInterpreter(file_ids=existing_file_ids)

        await cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.update(  # type: ignore[reportDeprecated]
                    thread_id=self._thread_id,
                    tool_resources=cast(thread_update_params.ToolResources, tool_resources.model_dump()),
                )
            )
        )

    async def on_upload_for_file_search(
        self, file_paths: str | Iterable[str], cancellation_token: CancellationToken
    ) -> None:
        """Handle file uploads for file search."""
        await self._ensure_initialized()

        # Check if file_search is enabled in tools
        if not any(tool.get("type") == "file_search" for tool in self._api_tools):
            raise ValueError(
                "File search is not enabled for this assistant. Add a file_search tool when creating the assistant."
            )

        # Create vector store if not already created
        if self._vector_store_id is None:
            vector_store: VectorStore = await cancellation_token.link_future(
                asyncio.ensure_future(self._client.vector_stores.create())
            )
            self._vector_store_id = vector_store.id

            # Update assistant with vector store ID
            await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.assistants.update(
                        assistant_id=self._get_assistant_id,
                        tool_resources={"file_search": {"vector_store_ids": [self._vector_store_id]}},
                    )
                )
            )

        file_ids = await self._upload_files(file_paths, cancellation_token)

        # Create file batch with the file IDs
        await cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.vector_stores.file_batches.create_and_poll(
                    vector_store_id=self._vector_store_id, file_ids=file_ids
                )
            )
        )

    async def delete_uploaded_files(self, cancellation_token: CancellationToken) -> None:
        """Delete all files that were uploaded by this agent instance."""
        await self._ensure_initialized()
        for file_id in self._uploaded_file_ids:
            try:
                await cancellation_token.link_future(asyncio.ensure_future(self._client.files.delete(file_id=file_id)))
            except Exception as e:
                event_logger.error(f"Failed to delete file {file_id}: {str(e)}")
        self._uploaded_file_ids = []

    async def delete_assistant(self, cancellation_token: CancellationToken) -> None:
        """Delete the assistant if it was created by this instance."""
        await self._ensure_initialized()
        if self._assistant is not None and not self._assistant_id:
            try:
                await cancellation_token.link_future(
                    asyncio.ensure_future(self._client.beta.assistants.delete(assistant_id=self._get_assistant_id))  # type: ignore[reportDeprecated]
                )
                self._assistant = None
            except Exception as e:
                event_logger.error(f"Failed to delete assistant: {str(e)}")

    async def delete_vector_store(self, cancellation_token: CancellationToken) -> None:
        """Delete the vector store if it was created by this instance."""
        await self._ensure_initialized()
        if self._vector_store_id is not None:
            try:
                await cancellation_token.link_future(
                    asyncio.ensure_future(self._client.vector_stores.delete(vector_store_id=self._vector_store_id))
                )
                self._vector_store_id = None
            except Exception as e:
                event_logger.error(f"Failed to delete vector store: {str(e)}")

    async def save_state(self) -> Mapping[str, Any]:
        state = OpenAIAssistantAgentState(
            assistant_id=self._assistant.id if self._assistant else self._assistant_id,
            thread_id=self._thread.id if self._thread else self._init_thread_id,
            initial_message_ids=list(self._initial_message_ids),
            vector_store_id=self._vector_store_id,
            uploaded_file_ids=self._uploaded_file_ids,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        agent_state = OpenAIAssistantAgentState.model_validate(state)
        self._assistant_id = agent_state.assistant_id
        self._init_thread_id = agent_state.thread_id
        self._initial_message_ids = set(agent_state.initial_message_ids)
        self._vector_store_id = agent_state.vector_store_id
        self._uploaded_file_ids = agent_state.uploaded_file_ids

# From openai/_openai_assistant_agent.py
def threads(self) -> AsyncThreads:
        return self._client.beta.threads

# From openai/_openai_assistant_agent.py
def runs(self) -> AsyncRuns:
        return self._client.beta.threads.runs

# From openai/_openai_assistant_agent.py
def messages(self) -> AsyncMessages:
        return self._client.beta.threads.messages

from typing_extensions import NotRequired
from typing_extensions import TypedDict

# From openai/_openai_agent.py
class FileSearchToolConfig(TypedDict):
    """Configuration for file_search tool."""

    type: Literal["file_search"]
    vector_store_ids: List[str]  # required - The IDs of the vector stores to search
    max_num_results: NotRequired[int]  # optional
    ranking_options: NotRequired[Dict[str, Any]]  # optional
    filters: NotRequired[Dict[str, Any]]

# From openai/_openai_agent.py
class WebSearchToolConfig(TypedDict):
    """Configuration for web_search_preview tool."""

    type: Literal["web_search_preview"]
    search_context_size: NotRequired[int]  # optional
    user_location: NotRequired[Union[str, Dict[str, Any]]]

# From openai/_openai_agent.py
class ComputerUseToolConfig(TypedDict):
    """Configuration for computer_use_preview tool."""

    type: Literal["computer_use_preview"]
    display_height: int  # required - Display height in pixels
    display_width: int  # required - Display width in pixels
    environment: str

# From openai/_openai_agent.py
class MCPToolConfig(TypedDict):
    """Configuration for mcp tool."""

    type: Literal["mcp"]
    server_label: str  # required - Label for the MCP server
    server_url: str  # required - URL of the MCP server
    allowed_tools: NotRequired[List[str]]  # optional - List of allowed tools
    headers: NotRequired[Dict[str, str]]  # optional - HTTP headers for requests
    require_approval: NotRequired[bool]

# From openai/_openai_agent.py
class CodeInterpreterToolConfig(TypedDict):
    """Configuration for code_interpreter tool."""

    type: Literal["code_interpreter"]
    container: str

# From openai/_openai_agent.py
class ImageGenerationToolConfig(TypedDict):
    """Configuration for image_generation tool."""

    type: Literal["image_generation"]
    background: NotRequired[str]  # optional - Background color or image
    input_image_mask: NotRequired[str]

# From openai/_openai_agent.py
class LocalShellToolConfig(TypedDict):
    """Configuration for local_shell tool.

    WARNING: This tool is only supported with the 'codex-mini-latest' model
    and is available exclusively through the Responses API.
    """

    type: Literal["local_shell"]

# From openai/_openai_agent.py
class ImageMessage(BaseChatMessage):
    """A message containing an image."""

    content: str  # URL or base64 string

    def to_model_message(self) -> UserMessage:
        return UserMessage(content=self.content, source=self.source)

    def to_model_text(self) -> str:
        return "[image]"

    def to_text(self) -> str:
        # Truncate long image content (especially base64) for better readability
        # While still showing enough of the URL or content to be identifiable
        if len(self.content) > IMAGE_CONTENT_PREVIEW_LENGTH:
            return f"[Image: {self.content[:IMAGE_CONTENT_PREVIEW_LENGTH]}...]"
        return f"[Image: {self.content}]"

# From openai/_openai_agent.py
class OpenAIMessageContent(TypedDict):
    type: str
    text: str

# From openai/_openai_agent.py
class OpenAIImageUrlContent(TypedDict):
    url: str

# From openai/_openai_agent.py
class OpenAIImageContent(TypedDict):
    type: str
    image_url: OpenAIImageUrlContent

# From openai/_openai_agent.py
class OpenAIMessage(TypedDict):
    role: str
    content: Union[str, List[Union[OpenAIMessageContent, OpenAIImageContent]]]

# From openai/_openai_agent.py
class OpenAIAgentState(BaseModel):
    type: str = Field(default="OpenAIAgentState")
    response_id: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

# From openai/_openai_agent.py
class OpenAIAgentConfig(BaseModel):
    """Configuration model for OpenAI agent that supports both custom tools and built-in tools.

    .. versionchanged:: v0.7.0
       Added support for built-in tools in JSON configuration via _to_config and _from_config methods.
       The tools field now accepts ComponentModel (for custom tools), built-in tool configurations
       (dict format), and built-in tool names (string format).
    """

    name: str
    description: str
    model: str
    instructions: str
    tools: List[ToolConfigUnion] | None = None
    temperature: Optional[float] = 1
    max_output_tokens: Optional[int] = None
    json_mode: bool = False
    store: bool = True
    truncation: str = "disabled"

# From openai/_openai_agent.py
class FunctionExecutionResult(BaseModel):
    """Result of a function execution."""

    content: str
    call_id: str
    name: str
    is_error: bool = False

# From openai/_openai_agent.py
class OpenAIAgent(BaseChatAgent, Component[OpenAIAgentConfig]):
    """
    An agent implementation that uses the OpenAI Responses API to generate responses.

    Installation:

    .. code-block:: bash

        pip install "autogen-ext[openai]"
        # pip install "autogen-ext[openai,azure]"  # For Azure OpenAI Assistant

    This agent leverages the Responses API to generate responses with capabilities like:

    * Custom function calling
    * Multi-turn conversations
    * Built-in tool support (file_search, code_interpreter, web_search_preview, etc.)

    .. versionchanged:: v0.7.0

       Added support for built-in tool types like file_search, web_search_preview,
       code_interpreter, computer_use_preview, image_generation, and mcp.
       Added support for tool configurations with required and optional parameters.

       Built-in tools are split into two categories:

       **Tools that can use string format** (no required parameters):

       - web_search_preview: Can be used as "web_search_preview" or with optional config
         (user_location, search_context_size)
       - image_generation: Can be used as "image_generation" or with optional config (background, input_image_mask)
       - local_shell: Can be used as "local_shell" (WARNING: Only works with codex-mini-latest model)

       **Tools that REQUIRE dict configuration** (have required parameters):

       - file_search: MUST use dict with vector_store_ids (List[str])
       - computer_use_preview: MUST use dict with display_height (int), display_width (int), environment (str)
       - code_interpreter: MUST use dict with container (str)
       - mcp: MUST use dict with server_label (str), server_url (str)

       Using required-parameter tools in string format will raise a ValueError with helpful error messages.
       The tools parameter type annotation only accepts string values for tools that don't require parameters.


    Args:
        name (str): Name of the agent
        description (str): Description of the agent's purpose
        client (Union[AsyncOpenAI, AsyncAzureOpenAI]): OpenAI client instance
        model (str): Model to use (e.g. "gpt-4.1")
        instructions (str): System instructions for the agent
        tools (Optional[Iterable[Union[str, BuiltinToolConfig, Tool]]]): Tools the agent can use.
            Supported string values (no required parameters): "web_search_preview", "image_generation", "local_shell".
            Dict values can provide configuration for built-in tools with parameters.
            Required parameters for built-in tools:
            - file_search: vector_store_ids (List[str])
            - computer_use_preview: display_height (int), display_width (int), environment (str)
            - code_interpreter: container (str)
            - mcp: server_label (str), server_url (str)
            Optional parameters for built-in tools:
            - file_search: max_num_results (int), ranking_options (dict), filters (dict)
            - web_search_preview: user_location (str or dict), search_context_size (int)
            - image_generation: background (str), input_image_mask (str)
            - mcp: allowed_tools (List[str]), headers (dict), require_approval (bool)
            Special tools with model restrictions:
            - local_shell: Only works with "codex-mini-latest" model (WARNING: Very limited support)
            Also accepts custom Tool objects for function calling.
        temperature (Optional[float]): Temperature for response generation (default: 1)
        max_output_tokens (Optional[int]): Maximum output tokens
        json_mode (bool): Whether to use JSON mode (default: False)
        store (bool): Whether to store conversations (default: True)
        truncation (str): Truncation strategy (default: "disabled")

    Example:

        Basic usage with built-in tools:

        .. code-block:: python

            from openai import AsyncOpenAI
            from autogen_core import CancellationToken
            from autogen_ext.agents.openai import OpenAIAgent
            from autogen_agentchat.messages import TextMessage
            import logging


            async def example():
                cancellation_token = CancellationToken()
                client = AsyncOpenAI()
                agent = OpenAIAgent(
                    name="Simple Agent",
                    description="A simple OpenAI agent using the Responses API",
                    client=client,
                    model="gpt-4o",
                    instructions="You are a helpful assistant.",
                    tools=["web_search_preview", "image_generation"],  # Only tools without required params
                )
                response = await agent.on_messages(
                    [TextMessage(source="user", content="Search for recent AI developments")], cancellation_token
                )
                logging.info(response)

        Usage with configured built-in tools:

        .. code-block:: python

            from openai import AsyncOpenAI
            from autogen_core import CancellationToken
            from autogen_ext.agents.openai import OpenAIAgent
            from autogen_agentchat.messages import TextMessage
            import logging


            async def example_with_configs():
                cancellation_token = CancellationToken()
                client = AsyncOpenAI()

                # Configure tools with required and optional parameters
                tools = [
                    {
                        "type": "file_search",
                        "vector_store_ids": ["vs_abc123"],  # required
                        "max_num_results": 10,  # optional
                    },
                    {
                        "type": "computer_use_preview",
                        "display_height": 1024,  # required
                        "display_width": 1280,  # required
                        "environment": "desktop",  # required
                    },
                    {
                        "type": "code_interpreter",
                        "container": "python-3.11",  # required
                    },
                    {
                        "type": "mcp",
                        "server_label": "my-mcp-server",  # required
                        "server_url": "http://localhost:3000",  # required
                    },
                    {
                        "type": "web_search_preview",
                        "user_location": {  # optional - structured location
                            "type": "approximate",  # required: "approximate" or "exact"
                            "country": "US",  # optional
                            "region": "CA",  # optional
                            "city": "San Francisco",  # optional
                        },
                        "search_context_size": 5,  # optional
                    },
                    "image_generation",  # Simple tools can still use string format
                ]

                agent = OpenAIAgent(
                    name="Configured Agent",
                    description="An agent with configured tools",
                    client=client,
                    model="gpt-4o",
                    instructions="You are a helpful assistant with specialized tools.",
                    tools=tools,  # type: ignore
                )
                response = await agent.on_messages(
                    [TextMessage(source="user", content="Search for recent AI developments")], cancellation_token
                )
                logging.info(response)

        Mixed usage with custom function tools:

        .. code-block:: python

            import asyncio
            import logging
            from openai import AsyncOpenAI
            from autogen_core import CancellationToken
            from autogen_ext.agents.openai import OpenAIAgent
            from autogen_agentchat.messages import TextMessage
            from autogen_core.tools import FunctionTool


            # Define a simple calculator function
            async def calculate(a: int, b: int) -> int:
                '''Simple function to add two numbers.'''
                return a + b


            # Wrap the calculate function as a tool
            calculator = FunctionTool(calculate, description="A simple calculator tool")


            async def example_mixed_tools():
                cancellation_token = CancellationToken()
                client = AsyncOpenAI()
                # Use the FunctionTool instance defined above

                agent = OpenAIAgent(
                    name="Mixed Tools Agent",
                    description="An agent with both built-in and custom tools",
                    client=client,
                    model="gpt-4o",
                    instructions="You are a helpful assistant with calculation and web search capabilities.",
                    tools=[
                        "web_search_preview",
                        calculator,
                        {"type": "mcp", "server_label": "tools", "server_url": "http://localhost:3000"},
                    ],
                )
                response = await agent.on_messages(
                    [TextMessage(source="user", content="What's 2+2 and what's the weather like?")],
                    cancellation_token,
                )
                logging.info(response)


            asyncio.run(example_mixed_tools())


    """

    component_config_schema = OpenAIAgentConfig
    component_provider_override = "autogen_ext.agents.openai.OpenAIAgent"

    def __init__(
        self: "OpenAIAgent",
        name: str,
        description: str,
        client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        model: str,
        instructions: str,
        tools: Optional[
            Iterable[
                Union[
                    Literal["web_search_preview", "image_generation", "local_shell"],
                    BuiltinToolConfig,
                    Tool,
                ]
            ]
        ] = None,
        temperature: Optional[float] = 1,
        max_output_tokens: Optional[int] = None,
        json_mode: bool = False,
        store: bool = True,
        truncation: str = "disabled",
    ) -> None:
        super().__init__(name, description)
        self._client: Union[AsyncOpenAI, AsyncAzureOpenAI] = client
        self._model: str = model
        self._instructions: str = instructions
        self._temperature: Optional[float] = temperature
        self._max_output_tokens: Optional[int] = max_output_tokens
        self._json_mode: bool = json_mode
        self._store: bool = store
        self._truncation: str = truncation
        self._last_response_id: Optional[str] = None
        self._message_history: List[Dict[str, Any]] = []
        self._tools: List[Dict[str, Any]] = []
        self._tool_map: Dict[str, Tool] = {}
        if tools is not None:
            for tool in tools:
                if isinstance(tool, str):
                    # Handle built-in tool types
                    self._add_builtin_tool(tool)
                elif isinstance(tool, dict) and "type" in tool:
                    # Handle configured built-in tools
                    self._add_configured_tool(tool)
                elif isinstance(tool, Tool):
                    # Handle custom function tools
                    function_schema: Dict[str, Any] = {
                        "type": "function",
                        "function": _convert_tool_to_function_schema(tool),
                    }
                    self._tools.append(function_schema)
                    self._tool_map[tool.name] = tool
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")

    def _add_builtin_tool(self, tool_name: str) -> None:
        """Add a built-in tool by name."""
        # Skip if an identical tool has already been registered (idempotent behaviour)
        if any(td.get("type") == tool_name for td in self._tools):
            return  # Duplicate – ignore rather than raise to stay backward-compatible
        # Only allow string format for tools that don't require parameters
        if tool_name == "web_search_preview":
            self._tools.append({"type": "web_search_preview"})
        elif tool_name == "image_generation":
            self._tools.append({"type": "image_generation"})
        elif tool_name == "local_shell":
            # Special handling for local_shell - very limited model support
            if self._model != "codex-mini-latest":
                raise ValueError(
                    f"Tool 'local_shell' is only supported with model 'codex-mini-latest', "
                    f"but current model is '{self._model}'. "
                    f"This tool is available exclusively through the Responses API and has severe limitations. "
                    f"Consider using autogen_ext.tools.code_execution.PythonCodeExecutionTool with "
                    f"autogen_ext.code_executors.local.LocalCommandLineCodeExecutor for shell execution instead."
                )
            self._tools.append({"type": "local_shell"})
        elif tool_name in ["file_search", "code_interpreter", "computer_use_preview", "mcp"]:
            # These tools require specific parameters and must use dict configuration
            raise ValueError(
                f"Tool '{tool_name}' requires specific parameters and cannot be added using string format. "
                f"Use dict configuration instead. Required parameters for {tool_name}: "
                f"{self._get_required_params_help(tool_name)}"
            )
        else:
            raise ValueError(f"Unsupported built-in tool type: {tool_name}")

    def _get_required_params_help(self, tool_name: str) -> str:
        """Get help text for required parameters of a tool."""
        help_text = {
            "file_search": "vector_store_ids (List[str])",
            "code_interpreter": "container (str)",
            "computer_use_preview": "display_height (int), display_width (int), environment (str)",
            "mcp": "server_label (str), server_url (str)",
        }
        return help_text.get(tool_name, "unknown parameters")

    def _add_configured_tool(self, tool_config: BuiltinToolConfig) -> None:
        """Add a configured built-in tool with parameters."""
        tool_type = tool_config.get("type")
        if not tool_type:
            raise ValueError("Tool configuration must include 'type' field")

        # If an identical configuration is already present we simply ignore the new one (keeps API payload minimal)
        if cast(Dict[str, Any], tool_config) in self._tools:
            return

        # Initialize tool definition
        tool_def: Dict[str, Any] = {}

        # Special validation for model-restricted tools
        if tool_type == "local_shell":
            if self._model != "codex-mini-latest":
                raise ValueError(
                    f"Tool 'local_shell' is only supported with model 'codex-mini-latest', "
                    f"but current model is '{self._model}'. "
                    f"This tool is available exclusively through the Responses API and has severe limitations. "
                    f"Consider using autogen_ext.tools.code_execution.PythonCodeExecutionTool with "
                    f"autogen_ext.code_executors.local.LocalCommandLineCodeExecutor for shell execution instead."
                )
            tool_def = {"type": "local_shell"}

        # For Responses API, built-in tools are defined directly without nesting
        elif tool_type == "file_search":
            # file_search requires vector_store_ids
            fs_config = cast(FileSearchToolConfig, tool_config)
            if "vector_store_ids" not in fs_config:
                raise ValueError("file_search tool requires 'vector_store_ids' parameter")

            vector_store_ids = fs_config["vector_store_ids"]
            if not isinstance(vector_store_ids, list) or not vector_store_ids:
                raise ValueError("file_search 'vector_store_ids' must be a non-empty list of strings")
            if not all(isinstance(vid, str) and vid.strip() for vid in vector_store_ids):
                raise ValueError("file_search 'vector_store_ids' must contain non-empty strings")

            tool_def = {"type": "file_search", "vector_store_ids": vector_store_ids}
            # Optional parameters
            if "max_num_results" in fs_config:
                max_results = fs_config["max_num_results"]
                if not isinstance(max_results, int) or max_results <= 0:
                    raise ValueError("file_search 'max_num_results' must be a positive integer")
                tool_def["max_num_results"] = max_results
            if "ranking_options" in fs_config:
                tool_def["ranking_options"] = fs_config["ranking_options"]
            if "filters" in fs_config:
                tool_def["filters"] = fs_config["filters"]

        elif tool_type == "web_search_preview":
            # web_search_preview can have optional parameters
            ws_config = cast(WebSearchToolConfig, tool_config)
            tool_def = {"type": "web_search_preview"}
            if "search_context_size" in ws_config:
                context_size = ws_config["search_context_size"]
                if not isinstance(context_size, int) or context_size <= 0:
                    raise ValueError("web_search_preview 'search_context_size' must be a positive integer")
                tool_def["search_context_size"] = context_size
            if "user_location" in ws_config:
                user_location = ws_config["user_location"]
                if isinstance(user_location, str):
                    if not user_location.strip():
                        raise ValueError(
                            "web_search_preview 'user_location' must be a non-empty string when using string format"
                        )
                elif isinstance(user_location, dict):
                    if "type" not in user_location:
                        raise ValueError("web_search_preview 'user_location' dictionary must include 'type' field")
                    location_type = user_location["type"]
                    if location_type not in ["approximate", "exact"]:
                        raise ValueError("web_search_preview 'user_location' type must be 'approximate' or 'exact'")
                    # Optional fields: country, region, city can be validated if present
                    for optional_field in ["country", "region", "city"]:
                        if optional_field in user_location:
                            if (
                                not isinstance(user_location[optional_field], str)
                                or not user_location[optional_field].strip()
                            ):
                                raise ValueError(
                                    f"web_search_preview 'user_location' {optional_field} must be a non-empty string"
                                )
                else:
                    raise ValueError("web_search_preview 'user_location' must be a string or dictionary")
                tool_def["user_location"] = user_location

        elif tool_type == "computer_use_preview":
            # computer_use_preview requires display dimensions and environment
            cu_config = cast(ComputerUseToolConfig, tool_config)
            required_params = ["display_height", "display_width", "environment"]
            for param in required_params:
                if param not in cu_config:
                    raise ValueError(f"computer_use_preview tool requires '{param}' parameter")

            # Validate display dimensions
            height = cu_config["display_height"]
            width = cu_config["display_width"]
            if not isinstance(height, int) or height <= 0:
                raise ValueError("computer_use_preview 'display_height' must be a positive integer")
            if not isinstance(width, int) or width <= 0:
                raise ValueError("computer_use_preview 'display_width' must be a positive integer")

            # Validate environment
            environment = cu_config["environment"]
            if not isinstance(environment, str) or not environment.strip():
                raise ValueError("computer_use_preview 'environment' must be a non-empty string")

            tool_def = {
                "type": "computer_use_preview",
                "display_height": height,
                "display_width": width,
                "environment": environment,
            }

        elif tool_type == "mcp":
            # MCP requires server_label and server_url
            mcp_config = cast(MCPToolConfig, tool_config)
            required_params = ["server_label", "server_url"]
            for param in required_params:
                if param not in mcp_config:
                    raise ValueError(f"mcp tool requires '{param}' parameter")

            # Validate required parameters
            server_label = mcp_config["server_label"]
            server_url = mcp_config["server_url"]
            if not isinstance(server_label, str) or not server_label.strip():
                raise ValueError("mcp 'server_label' must be a non-empty string")
            if not isinstance(server_url, str) or not server_url.strip():
                raise ValueError("mcp 'server_url' must be a non-empty string")

            tool_def = {"type": "mcp", "server_label": server_label, "server_url": server_url}
            # Optional parameters
            if "allowed_tools" in mcp_config:
                allowed_tools = mcp_config["allowed_tools"]
                if not isinstance(allowed_tools, list):
                    raise ValueError("mcp 'allowed_tools' must be a list of strings")
                if not all(isinstance(tool, str) for tool in allowed_tools):
                    raise ValueError("mcp 'allowed_tools' must contain only strings")
                tool_def["allowed_tools"] = allowed_tools
            if "headers" in mcp_config:
                headers = mcp_config["headers"]
                if not isinstance(headers, dict):
                    raise ValueError("mcp 'headers' must be a dictionary")
                tool_def["headers"] = headers
            if "require_approval" in mcp_config:
                require_approval = mcp_config["require_approval"]
                if not isinstance(require_approval, bool):
                    raise ValueError("mcp 'require_approval' must be a boolean")
                tool_def["require_approval"] = require_approval

        elif tool_type == "code_interpreter":
            # code_interpreter requires container
            ci_config = cast(CodeInterpreterToolConfig, tool_config)
            if "container" not in ci_config:
                raise ValueError("code_interpreter tool requires 'container' parameter")

            container = ci_config["container"]
            if not isinstance(container, str) or not container.strip():
                raise ValueError("code_interpreter 'container' must be a non-empty string")

            tool_def = {"type": "code_interpreter", "container": container}

        elif tool_type == "image_generation":
            # image_generation can have optional parameters
            ig_config = cast(ImageGenerationToolConfig, tool_config)
            tool_def = {"type": "image_generation"}
            if "background" in ig_config:
                background = ig_config["background"]
                if not isinstance(background, str) or not background.strip():
                    raise ValueError("image_generation 'background' must be a non-empty string")
                tool_def["background"] = background
            if "input_image_mask" in ig_config:
                input_image_mask = ig_config["input_image_mask"]
                if not isinstance(input_image_mask, str) or not input_image_mask.strip():
                    raise ValueError("image_generation 'input_image_mask' must be a non-empty string")
                tool_def["input_image_mask"] = input_image_mask

        else:
            raise ValueError(f"Unsupported built-in tool type: {tool_type}")

        self._tools.append(tool_def)

    def _convert_message_to_dict(self, message: OpenAIMessage) -> Dict[str, Any]:
        """Convert an OpenAIMessage to a Dict[str, Any]."""
        return dict(message)

    @property
    def produced_message_types(
        self: "OpenAIAgent",
    ) -> Sequence[
        Union[
            Type[TextMessage],
            Type[MultiModalMessage],
            Type[StopMessage],
            Type[ToolCallSummaryMessage],
            Type[HandoffMessage],
        ]
    ]:
        """Return the types of messages that this agent can produce."""
        return [TextMessage, MultiModalMessage, StopMessage, ToolCallSummaryMessage, HandoffMessage]

    async def _execute_tool_call(
        self: "OpenAIAgent", tool_call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        tool_name = tool_call.name
        if tool_name not in self._tool_map:
            return FunctionExecutionResult(
                content=f"Error: Tool '{tool_name}' is not available",
                call_id=tool_call.id,
                name=tool_name,
                is_error=True,
            )

        tool = self._tool_map[tool_name]
        try:
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError as json_err:
                return FunctionExecutionResult(
                    content=f"Error: Invalid JSON in tool arguments - {str(json_err)}",
                    call_id=tool_call.id,
                    name=tool_name,
                    is_error=True,
                )

            result = await tool.run_json(arguments, cancellation_token, call_id=tool_call.id)
            return FunctionExecutionResult(
                content=tool.return_value_as_string(result), call_id=tool_call.id, name=tool_name, is_error=False
            )
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            event_logger.warning(f"Tool execution error in {tool_name}: {error_msg}")
            return FunctionExecutionResult(content=error_msg, call_id=tool_call.id, name=tool_name, is_error=True)

    def _build_api_parameters(self: "OpenAIAgent", messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        has_system_message = any(msg.get("role") == "system" for msg in messages)
        if self._instructions and not has_system_message:
            messages = [{"role": "system", "content": self._instructions}] + messages
        api_params: Dict[str, Any] = {
            "model": self._model,
            "input": messages,  # Responses API expects 'input'
        }
        if self._temperature is not None:
            api_params["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            api_params["max_output_tokens"] = self._max_output_tokens
        if self._tools:
            api_params["tools"] = self._tools
        if self._json_mode:
            api_params["text"] = {"type": "json_object"}
        api_params["store"] = self._store
        api_params["truncation"] = self._truncation
        if self._last_response_id:
            api_params["previous_response_id"] = self._last_response_id
        return api_params

    async def on_messages(
        self: "OpenAIAgent", messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        response = None
        inner_messages: List[
            Union[AgentEvent, TextMessage, MultiModalMessage, StopMessage, ToolCallSummaryMessage, HandoffMessage]
        ] = []

        async for msg in self.on_messages_stream(messages, cancellation_token):
            if isinstance(msg, Response):
                response = msg
            # ModelClientStreamingChunkEvent does not exist in this version, so skip this check
            else:
                inner_messages.append(msg)

        if response is None:
            raise ValueError("No response was generated")

        if response.inner_messages is None:
            response.inner_messages = []

        for msg in inner_messages:
            if msg not in response.inner_messages:
                response.inner_messages = list(response.inner_messages) + [msg]

        return response

    async def on_messages_stream(
        self: "OpenAIAgent", messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[
        Union[
            AgentEvent, TextMessage, MultiModalMessage, StopMessage, ToolCallSummaryMessage, HandoffMessage, Response
        ],
        None,
    ]:
        input_messages: List[Dict[str, Any]] = []

        if self._message_history:
            input_messages.extend(self._message_history)

        for message in messages:
            if isinstance(
                message, (TextMessage, MultiModalMessage, StopMessage, ToolCallSummaryMessage, HandoffMessage)
            ):
                openai_message = _convert_message_to_openai_message(message)
                dict_message = self._convert_message_to_dict(openai_message)
                input_messages.append(dict_message)
                self._message_history.append(dict_message)
            else:
                msg_content = str(cast(Any, message).content) if hasattr(message, "content") else str(message)
                dict_message = {"role": "user", "content": msg_content}
                input_messages.append(dict_message)
                self._message_history.append(dict_message)

        inner_messages: List[AgentEvent | ChatMessage] = []

        api_params = self._build_api_parameters(input_messages)

        try:
            client = cast(Any, self._client)
            response_obj = await cancellation_token.link_future(
                asyncio.ensure_future(client.responses.create(**api_params))
            )
            content = getattr(response_obj, "output_text", None)
            response_id = getattr(response_obj, "id", None)
            self._last_response_id = response_id
            # Use a readable placeholder when the API returns no content to aid debugging
            content_str: str = str(content) if content is not None else "[no content returned]"
            self._message_history.append({"role": "assistant", "content": content_str})
            final_message = TextMessage(source=self.name, content=content_str)
            response = Response(chat_message=final_message, inner_messages=inner_messages)
            yield response
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            event_logger.error(f"API error: {error_message}", exc_info=True)
            error_response = TextMessage(source=self.name, content=error_message)
            yield Response(chat_message=error_response, inner_messages=inner_messages)

    async def on_reset(self: "OpenAIAgent", cancellation_token: CancellationToken) -> None:
        self._last_response_id = None
        self._message_history = []

    async def save_state(self: "OpenAIAgent") -> Mapping[str, Any]:
        state = OpenAIAgentState(
            response_id=self._last_response_id,
            history=self._message_history,
        )
        return state.model_dump()

    async def load_state(self: "OpenAIAgent", state: Mapping[str, Any]) -> None:
        agent_state = OpenAIAgentState.model_validate(state)
        self._last_response_id = agent_state.response_id
        self._message_history = agent_state.history

    def _to_config(self: "OpenAIAgent") -> OpenAIAgentConfig:
        """Convert the OpenAI agent to a declarative config.

        Serializes both custom Tool objects and built-in tools to their appropriate
        configuration formats for JSON serialization.

        .. versionchanged:: v0.6.2
           Added support for serializing built-in tools alongside custom tools.

        Returns:
            OpenAIAgentConfig: The configuration that can recreate this agent.
        """
        # Serialize tools in the **original order** they were registered.  We iterate over the
        # internal ``self._tools`` list which contains both built-in tool definitions **and** the
        # synthetic "function" records for custom :class:`Tool` objects.  For the latter we
        # convert the synthetic record back to a :class:`ComponentModel` by looking up the actual
        # tool instance in ``self._tool_map``.  This approach keeps ordering stable while still
        # supporting full round-trip serialisation.
        tool_configs: List[ToolConfigUnion] = []

        for tool_def in self._tools:
            # 1. Custom function tools are stored internally as ``{"type": "function", "function": {...}}``.
            if tool_def.get("type") == "function":
                fn_schema = cast(Dict[str, Any], tool_def.get("function", {}))
                tool_name = fn_schema.get("name")  # type: ignore[arg-type]
                if tool_name and tool_name in self._tool_map:
                    tool_obj = self._tool_map[tool_name]
                    try:
                        if hasattr(tool_obj, "dump_component"):
                            component_model = cast(Any, tool_obj).dump_component()
                            tool_configs.append(component_model)
                        else:
                            component_model = ComponentModel(
                                provider="autogen_core.tools.FunctionTool",
                                component_type=None,
                                config={
                                    "name": tool_obj.name,
                                    "description": getattr(tool_obj, "description", ""),
                                },
                            )
                            tool_configs.append(component_model)
                    except Exception as e:  # pragma: no cover – extremely unlikely
                        warnings.warn(
                            f"Error serializing tool '{tool_name}': {e}",
                            stacklevel=2,
                        )
                        component_model = ComponentModel(
                            provider="autogen_core.tools.FunctionTool",
                            component_type=None,
                            config={
                                "name": tool_name or "unknown_tool",
                                "description": getattr(tool_obj, "description", ""),
                            },
                        )
                        tool_configs.append(component_model)
            # 2. Built-in tools are already in their correct dict form – append verbatim.
            elif "type" in tool_def:  # built-in tool
                tool_configs.append(cast(BuiltinToolConfig, tool_def))
            else:  # pragma: no cover – should never happen
                warnings.warn(
                    f"Encountered unexpected tool definition during serialisation: {tool_def}",
                    stacklevel=2,
                )

        return OpenAIAgentConfig(
            name=self.name,
            description=self.description,
            model=self._model,
            instructions=self._instructions,
            tools=tool_configs if tool_configs else None,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            json_mode=self._json_mode,
            store=self._store,
            truncation=self._truncation,
        )

    @classmethod
    def _from_config(cls: Type["OpenAIAgent"], config: OpenAIAgentConfig) -> "OpenAIAgent":
        """Create an OpenAI agent from a declarative config.

        Handles both custom Tool objects (from ComponentModel) and built-in tools
        (from string or dict configurations).

        .. versionchanged:: v0.6.2
           Added support for loading built-in tools alongside custom tools.

        Args:
            config: The configuration to load the agent from.

        Returns:
            OpenAIAgent: The reconstructed agent.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        tools: Optional[List[Union[str, BuiltinToolConfig, Tool]]] = None
        if config.tools:
            tools_list: List[Union[str, BuiltinToolConfig, Tool]] = []
            for tool_config in config.tools:
                # Handle ComponentModel (custom Tool objects)
                if isinstance(tool_config, ComponentModel):
                    try:
                        provider = tool_config.provider
                        module_name, class_name = provider.rsplit(".", 1)
                        module = __import__(module_name, fromlist=[class_name])
                        tool_cls = getattr(module, class_name)
                        tool = tool_cls(**tool_config.config)
                        tools_list.append(cast(Tool, tool))
                    except Exception as e:
                        warnings.warn(f"Error loading custom tool: {e}", stacklevel=2)
                        from autogen_core.tools import FunctionTool

                        async def dummy_func(*args: Any, **kwargs: Any) -> str:
                            return "Tool not fully restored"

                        tool = FunctionTool(
                            name=tool_config.config.get("name", "unknown_tool"),
                            description=tool_config.config.get("description", ""),
                            func=dummy_func,
                        )
                        tools_list.append(tool)

                # Handle string format built-in tools
                elif isinstance(tool_config, str):
                    tools_list.append(tool_config)

                # Handle dict format built-in tools
                elif isinstance(tool_config, dict) and "type" in tool_config:
                    tools_list.append(tool_config)  # type: ignore[arg-type]

                else:
                    warnings.warn(f"Unknown tool configuration format: {type(tool_config)}", stacklevel=2)

            tools = tools_list if tools_list else None

        return cls(
            name=config.name,
            description=config.description,
            client=client,
            model=config.model,
            instructions=config.instructions,
            tools=cast(
                Optional[
                    Iterable[
                        Union[
                            BuiltinToolConfig,
                            Tool,
                            Literal["web_search_preview", "image_generation", "local_shell"],
                        ]
                    ]
                ],
                tools,
            ),
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            json_mode=config.json_mode,
            store=config.store,
            truncation=config.truncation,
        )

    # Add public API wrappers for configuration and tools
    def to_config(self) -> OpenAIAgentConfig:
        """Public wrapper for the private _to_config method."""
        return self._to_config()

    @classmethod
    def from_config(cls, config: OpenAIAgentConfig) -> "OpenAIAgent":
        """Public wrapper for the private _from_config classmethod."""
        return cls._from_config(config)

    @property
    def tools(self) -> list[Any]:
        """Public access to the agent's tools."""
        return self._tools

    @property
    def model(self) -> str:
        """Public access to the agent's model."""
        return self._model

# From openai/_openai_agent.py
def to_config(self) -> OpenAIAgentConfig:
        """Public wrapper for the private _to_config method."""
        return self._to_config()

# From openai/_openai_agent.py
def from_config(cls, config: OpenAIAgentConfig) -> "OpenAIAgent":
        """Public wrapper for the private _from_config classmethod."""
        return cls._from_config(config)

# From openai/_openai_agent.py
def model(self) -> str:
        """Public access to the agent's model."""
        return self._model

from autogen_core.models._types import FunctionExecutionResult
from azure.ai.agents.models import Agent
from azure.ai.agents.models import AgentsResponseFormat
from azure.ai.agents.models import AgentThread
from azure.ai.agents.models import AzureAISearchToolDefinition
from azure.ai.agents.models import AzureFunctionToolDefinition
from azure.ai.agents.models import BingGroundingToolDefinition
from azure.ai.agents.models import CodeInterpreterToolDefinition
from azure.ai.agents.models import CodeInterpreterToolResource
from azure.ai.agents.models import FileInfo
from azure.ai.agents.models import FilePurpose
from azure.ai.agents.models import FileSearchToolDefinition
from azure.ai.agents.models import FileSearchToolResource
from azure.ai.agents.models import FileState
from azure.ai.agents.models import FunctionDefinition
from azure.ai.agents.models import FunctionToolDefinition
from azure.ai.agents.models import ListSortOrder
from azure.ai.agents.models import MessageRole
from azure.ai.agents.models import MessageTextUrlCitationAnnotation
from azure.ai.agents.models import RunStatus
from azure.ai.agents.models import ThreadRun
from azure.ai.agents.models import ToolDefinition
from azure.ai.agents.models import ToolOutput
from azure.ai.agents.models import ToolResources
from azure.ai.agents.models import VectorStore
from azure.ai.agents.models import VectorStoreChunkingStrategyRequest
from azure.ai.agents.models import VectorStoreDataSource
from azure.ai.agents.models import VectorStoreExpirationPolicy
from azure.ai.agents.models._patch import ThreadMessage
from azure.ai.projects.aio import AIProjectClient
from _types import AzureAIAgentState
from _types import ListToolType

# From azure/_azure_ai_agent.py
class AzureAIAgent(BaseChatAgent):
    """
    Azure AI Assistant agent for AutoGen.

    Installation:

    .. code-block:: bash

        pip install "autogen-ext[azure]"  # For Azure AI Foundry Agent Service

    This agent leverages the Azure AI Assistant API to create AI assistants with capabilities like:

    * Code interpretation and execution
    * Grounding with Bing search
    * File handling and search
    * Custom function calling
    * Multi-turn conversations

    The agent integrates with AutoGen's messaging system, providing a seamless way to use Azure AI
    capabilities within the AutoGen framework. It supports tools like code interpreter,
    file search, and various grounding mechanisms.

    Agent name must be a valid Python identifier:
        1. It must start with a letter (A-Z, a-z) or an underscore (_).
        2. It can only contain letters, digits (0-9), or underscores.
        3. It cannot be a Python keyword.
        4. It cannot contain spaces or special characters.
        5. It cannot start with a digit.


    Check here on how to create a new secured agent with user-managed identity:
    https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/virtual-networks

    Examples:

        Use the AzureAIAgent to create an agent grounded with Bing:

        .. code-block:: python

            import asyncio
            import os

            from autogen_agentchat.messages import TextMessage
            from autogen_core import CancellationToken
            from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent
            from azure.ai.projects.aio import AIProjectClient
            from azure.identity.aio import DefaultAzureCredential
            from azure.ai.agents.models import BingGroundingTool
            import dotenv


            async def bing_example():
                async with DefaultAzureCredential() as credential:
                    async with AIProjectClient(  # type: ignore
                        credential=credential, endpoint=os.getenv("AZURE_PROJECT_ENDPOINT", "")
                    ) as project_client:
                        conn = await project_client.connections.get(name=os.getenv("BING_CONNECTION_NAME", ""))

                        bing_tool = BingGroundingTool(conn.id)
                        agent_with_bing_grounding = AzureAIAgent(
                            name="bing_agent",
                            description="An AI assistant with Bing grounding",
                            project_client=project_client,
                            deployment_name="gpt-4o",
                            instructions="You are a helpful assistant.",
                            tools=bing_tool.definitions,
                            metadata={"source": "AzureAIAgent"},
                        )

                        # For the bing grounding tool to return the citations, the message must contain an instruction for the model to do return them.
                        # For example: "Please provide citations for the answers"

                        result = await agent_with_bing_grounding.on_messages(
                            messages=[
                                TextMessage(
                                    content="What is Microsoft\\'s annual leave policy? Provide citations for your answers.",
                                    source="user",
                                )
                            ],
                            cancellation_token=CancellationToken(),
                            message_limit=5,
                        )
                        print(result)


            if __name__ == "__main__":
                dotenv.load_dotenv()
                asyncio.run(bing_example())

        Use the AzureAIAgent to create an agent with file search capability:

        .. code-block:: python

            import asyncio
            import os
            import tempfile
            import urllib.request

            import dotenv
            from autogen_agentchat.messages import TextMessage
            from autogen_core import CancellationToken
            from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent
            from azure.ai.projects.aio import AIProjectClient
            from azure.identity.aio import DefaultAzureCredential


            async def file_search_example():
                # Download README.md from GitHub
                readme_url = "https://raw.githubusercontent.com/microsoft/autogen/refs/heads/main/README.md"
                temp_file = None

                try:
                    # Create a temporary file to store the downloaded README
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
                    urllib.request.urlretrieve(readme_url, temp_file.name)
                    print(f"Downloaded README.md to {temp_file.name}")

                    async with DefaultAzureCredential() as credential:
                        async with AIProjectClient(  # type: ignore
                            credential=credential, endpoint=os.getenv("AZURE_PROJECT_ENDPOINT", "")
                        ) as project_client:
                            agent_with_file_search = AzureAIAgent(
                                name="file_search_agent",
                                description="An AI assistant with file search capabilities",
                                project_client=project_client,
                                deployment_name="gpt-4.1-mini",
                                instructions="You are a helpful assistant.",
                                tools=["file_search"],
                                metadata={"source": "AzureAIAgent"},
                            )

                            ct: CancellationToken = CancellationToken()
                            # Use the downloaded README file for file search
                            await agent_with_file_search.on_upload_for_file_search(
                                file_paths=[temp_file.name],
                                vector_store_name="file_upload_index",
                                vector_store_metadata={"source": "AzureAIAgent"},
                                cancellation_token=ct,
                                vector_store_polling_interval=60,
                            )
                            result = await agent_with_file_search.on_messages(
                                messages=[
                                    TextMessage(
                                        content="Hello, what is AutoGen and what capabilities does it have?", source="user"
                                    )
                                ],
                                cancellation_token=ct,
                                message_limit=5,
                            )
                            print(result)
                finally:
                    # Clean up the temporary file
                    if temp_file and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        print(f"Removed temporary file {temp_file.name}")


            if __name__ == "__main__":
                dotenv.load_dotenv()
                asyncio.run(file_search_example())

        Use the AzureAIAgent to create an agent with code interpreter capability:

        .. code-block:: python

            import asyncio
            import os

            import dotenv
            from autogen_agentchat.messages import TextMessage
            from autogen_core import CancellationToken
            from autogen_ext.agents.azure._azure_ai_agent import AzureAIAgent
            from azure.ai.projects.aio import AIProjectClient
            from azure.identity.aio import DefaultAzureCredential


            async def code_interpreter_example():
                async with DefaultAzureCredential() as credential:
                    async with AIProjectClient(  # type: ignore
                        credential=credential, endpoint=os.getenv("AZURE_PROJECT_ENDPOINT", "")
                    ) as project_client:
                        agent_with_code_interpreter = AzureAIAgent(
                            name="code_interpreter_agent",
                            description="An AI assistant with code interpreter capabilities",
                            project_client=project_client,
                            deployment_name="gpt-4.1-mini",
                            instructions="You are a helpful assistant.",
                            tools=["code_interpreter"],
                            metadata={"source": "AzureAIAgent"},
                        )

                        await agent_with_code_interpreter.on_upload_for_code_interpreter(
                            file_paths="/workspaces/autogen/python/packages/autogen-core/docs/src/user-guide/core-user-guide/cookbook/data/nifty_500_quarterly_results.csv",
                            cancellation_token=CancellationToken(),
                            polling_interval=5,
                        )

                        result = await agent_with_code_interpreter.on_messages(
                            messages=[
                                TextMessage(
                                    content="Aggregate the number of stocks per industry and give me a markdown table as a result?",
                                    source="user",
                                )
                            ],
                            cancellation_token=CancellationToken(),
                        )

                        print(result)


            if __name__ == "__main__":
                dotenv.load_dotenv()
                asyncio.run(code_interpreter_example())
    """

    def __init__(
        self,
        name: str,
        description: str,
        project_client: AIProjectClient,
        deployment_name: str,
        instructions: str,
        tools: Optional[ListToolType] = None,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        response_format: Optional[AgentsResponseFormat] = None,
        temperature: Optional[float] = None,
        tool_resources: Optional[ToolResources] = None,
        top_p: Optional[float] = None,
    ) -> None:
        """
        Initialize the Azure AI Agent.

        Args:
            name (str): The name of the agent. Must be a valid Python identifier.
            description (str): A brief description of the agent's purpose.
            project_client (AIProjectClient): The Azure AI Project client for API interactions.
            deployment_name (str): The model deployment name to use for the agent (e.g., "gpt-4").
            instructions (str): Detailed instructions for the agent's behavior.
            tools (Optional[Iterable[Union[str, ToolDefinition, Tool, Callable]]]): A list of tools the agent can use.
                Supported string values: "file_search", "code_interpreter", "bing_grounding",
                "azure_ai_search", "azure_function", "sharepoint_grounding".
            agent_id (Optional[str]): Existing agent ID to use instead of creating a new one.
            thread_id (Optional[str]): Existing thread ID to continue a conversation.
            metadata (Optional[Dict[str, str]]): Additional metadata for the agent.
            response_format (Optional[_types.AgentsApiResponseFormatOption]): Format options for the agent's responses.
            temperature (Optional[float]): Sampling temperature, controls randomness of output.
            tool_resources (Optional[models.ToolResources]): Resources configuration for agent tools.
            top_p (Optional[float]): An alternative to temperature, nucleus sampling parameter.

        Raises:
            ValueError: If an unsupported tool type is provided.
        """
        super().__init__(name, description)

        if tools is None:
            tools = []

        self._original_tools: list[Tool] = []

        converted_tools: List[ToolDefinition] = []
        self._add_tools(tools, converted_tools)

        self._project_client = project_client
        self._agent: Optional[Agent] = None
        self._thread: Optional[AgentThread] = None
        self._init_thread_id = thread_id
        self._deployment_name = deployment_name
        self._instructions = instructions
        self._api_tools = converted_tools
        self._agent_id = agent_id
        self._metadata = metadata
        self._response_format = response_format
        self._temperature = temperature
        self._tool_resources = tool_resources
        self._top_p = top_p
        self._vector_store_id: Optional[str] = None
        self._uploaded_file_ids: List[str] = []

        self._initial_message_ids: Set[str] = set()
        self._initial_state_retrieved: bool = False

    # Properties
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        """The types of messages that the assistant agent produces."""
        return (TextMessage,)

    @property
    def thread_id(self) -> str:
        if self._thread is None:
            raise ValueError("Thread not initialized")
        return self._thread.id

    @property
    def _get_agent_id(self) -> str:
        if self._agent is None:
            raise ValueError("Agent not initialized")
        return self._agent.id

    @property
    def description(self) -> str:
        if not self._description:
            raise ValueError("Description not initialized")
        return self._description

    @property
    def agent_id(self) -> str:
        if not self._agent_id:
            raise ValueError("Agent not initialized")
        return self._agent_id

    @property
    def deployment_name(self) -> str:
        if not self._deployment_name:
            raise ValueError("Deployment name not initialized")
        return self._deployment_name

    @property
    def instructions(self) -> str:
        if not self._instructions:
            raise ValueError("Instructions not initialized")
        return self._instructions

    @property
    def tools(self) -> List[ToolDefinition]:
        """
        Get the list of tools available to the agent.

        Returns:
            List[ToolDefinition]: The list of tool definitions.
        """
        return self._api_tools

    def _add_tools(self, tools: Optional[ListToolType], converted_tools: List[ToolDefinition]) -> None:
        """
        Convert various tool formats to Azure AI Agent tool definitions.

        Args:
            tools: List of tools in various formats (string identifiers, ToolDefinition objects, Tool objects, or callables)
            converted_tools: List to which converted tool definitions will be added

        Raises:
            ValueError: If an unsupported tool type is provided
        """
        if tools is None:
            return

        for tool in tools:
            if isinstance(tool, str):
                if tool == "file_search":
                    converted_tools.append(FileSearchToolDefinition())
                elif tool == "code_interpreter":
                    converted_tools.append(CodeInterpreterToolDefinition())
                elif tool == "bing_grounding":
                    converted_tools.append(BingGroundingToolDefinition())  # type: ignore
                elif tool == "azure_ai_search":
                    converted_tools.append(AzureAISearchToolDefinition())
                elif tool == "azure_function":
                    converted_tools.append(AzureFunctionToolDefinition())  # type: ignore
                # elif tool == "sharepoint_grounding":
                #     converted_tools.append(SharepointToolDefinition())  # type: ignore
                else:
                    raise ValueError(f"Unsupported tool string: {tool}")
            elif isinstance(tool, ToolDefinition):
                converted_tools.append(tool)
            elif isinstance(tool, Tool):
                self._original_tools.append(tool)
                converted_tools.append(self._convert_tool_to_function_tool_definition(tool))
            elif callable(tool):
                if hasattr(tool, "__doc__") and tool.__doc__ is not None:
                    description = tool.__doc__
                else:
                    description = ""
                function_tool = FunctionTool(tool, description=description)
                self._original_tools.append(function_tool)
                converted_tools.append(self._convert_tool_to_function_tool_definition(function_tool))
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

    def _convert_tool_to_function_tool_definition(self, tool: Tool) -> FunctionToolDefinition:
        """
        Convert an autogen Tool to an Azure AI Agent function tool definition.

        Args:
            tool (Tool): The AutoGen tool to convert

        Returns:
            models.FunctionToolDefinition: A function tool definition compatible with Azure AI Agent API
        """

        schema = tool.schema
        parameters: Dict[str, object] = {}

        if "parameters" in schema:
            parameters = {
                "type": schema["parameters"]["type"],
                "properties": schema["parameters"]["properties"],
            }
            if "required" in schema["parameters"]:
                parameters["required"] = schema["parameters"]["required"]

        func_definition = FunctionDefinition(name=tool.name, description=tool.description, parameters=parameters)

        return FunctionToolDefinition(
            function=func_definition,
        )

    async def _ensure_initialized(self, create_new_thread: bool = False, create_new_agent: bool = False) -> None:
        """
        Ensure agent and thread are properly initialized before operations.

        This method ensures that both the Azure AI Agent and thread are created or retrieved
        from existing IDs. It also handles retrieving the initial state of an existing thread
        when needed.

        Args:
            create_new_thread (bool): When True, creates a new thread even if thread_id is provided
            create_new_agent (bool): When True, creates a new agent even if agent_id is provided

        Raises:
            ValueError: If agent or thread creation fails
        """
        if self._agent is None or create_new_agent:
            if self._agent_id and create_new_agent is False:
                self._agent = await self._project_client.agents.get_agent(agent_id=self._agent_id)
            else:
                self._agent = await self._project_client.agents.create_agent(
                    name=self.name,
                    model=self._deployment_name,
                    description=self.description,
                    instructions=self._instructions,
                    tools=self._api_tools,
                    metadata=self._metadata,
                    response_format=self._response_format if self._response_format else None,  # type: ignore
                    temperature=self._temperature,
                    tool_resources=self._tool_resources if self._tool_resources else None,  # type: ignore
                    top_p=self._top_p,
                )

        if self._thread is None or create_new_thread:
            if self._init_thread_id and create_new_thread is False:
                self._thread = await self._project_client.agents.threads.get(thread_id=self._init_thread_id)
                # Retrieve initial state only once
                if not self._initial_state_retrieved:
                    await self._retrieve_initial_state()
                    self._initial_state_retrieved = True
            else:
                self._thread = await self._project_client.agents.threads.create()

    async def _retrieve_initial_state(self) -> None:
        """
        Retrieve and store the initial state of messages in the thread.

        This method retrieves all message IDs from an existing thread to track which
        messages were present before this agent instance started interacting with the thread.
        It handles pagination to ensure all messages are captured.
        """
        # Retrieve all initial message IDs
        initial_message_ids: Set[str] = set()
        async for msg in self._project_client.agents.messages.list(
            thread_id=self.thread_id,
            order=ListSortOrder.ASCENDING,
            limit=100,
        ):
            initial_message_ids.add(msg.id)
        self._initial_message_ids = initial_message_ids

    async def _execute_tool_call(self, tool_call: FunctionCall, cancellation_token: CancellationToken) -> str:
        """
        Execute a tool call requested by the Azure AI agent.

        Args:
            tool_call (FunctionCall): The function call information including name and arguments
            cancellation_token (CancellationToken): Token for cancellation handling

        Returns:
            str: The string representation of the tool call result

        Raises:
            ValueError: If the requested tool is not available or no tools are registered
        """
        if not self._original_tools:
            raise ValueError("No tools are available.")
        tool = next((t for t in self._original_tools if t.name == tool_call.name), None)
        if tool is None:
            raise ValueError(f"The tool '{tool_call.name}' is not available.")
        arguments = json.loads(tool_call.arguments)
        result = await tool.run_json(arguments, cancellation_token, call_id=tool_call.id)
        return tool.return_value_as_string(result)

    async def _upload_files(
        self,
        file_paths: str | Iterable[str],
        purpose: str = "assistant",
        polling_interval: float = 0.5,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> List[str]:
        """
        Upload files to the Azure AI Assistant API.

        This method handles uploading one or more files to be used by the agent
        and tracks their IDs in the agent's state.

        Args:
            file_paths (str | Iterable[str]): Path(s) to file(s) to upload
            purpose (str): The purpose of the file, defaults to "assistant"
            polling_interval (float): Time to sleep between polling for file status
            cancellation_token (Optional[CancellationToken]): Token for cancellation handling

        Returns:
            List[str]: List of file IDs for the uploaded files

        Raises:
            ValueError: If file upload fails
        """
        if cancellation_token is None:
            cancellation_token = CancellationToken()

        await self._ensure_initialized()

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        file_ids: List[str] = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)

            file: FileInfo = await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._project_client.agents.files.upload_and_poll(
                        file_path=file_path, purpose=purpose, polling_interval=polling_interval
                    )
                )
            )

            if file.status != FileState.PROCESSED:
                raise ValueError(f"File upload failed with status {file.status}")

            trace_logger.debug(f"File uploaded successfully: {file.id}, {file_name}")

            file_ids.append(file.id)
            self._uploaded_file_ids.append(file.id)

        return file_ids

    # Public Methods
    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: Optional[CancellationToken] = None,
        message_limit: int = 1,
    ) -> Response:
        """
        Process incoming messages and return a response from the Azure AI agent.

        This method is the primary entry point for interaction with the agent.
        It delegates to on_messages_stream and returns the final response.

        Args:
            messages (Sequence[BaseChatMessage]): The messages to process
            cancellation_token (CancellationToken): Token for cancellation handling
            message_limit (int, optional): Maximum number of messages to retrieve from the thread

        Returns:
            Response: The agent's response, including the chat message and any inner events

        Raises:
            AssertionError: If the stream doesn't return a final result
        """
        async for message in self.on_messages_stream(
            messages=messages, cancellation_token=cancellation_token, message_limit=message_limit
        ):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: Optional[CancellationToken] = None,
        message_limit: int = 1,
        polling_interval: float = 0.5,
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        """
        Process incoming messages and yield streaming responses from the Azure AI agent.

        This method handles the complete interaction flow with the Azure AI agent:
        1. Processing input messages
        2. Creating and monitoring a run
        3. Handling tool calls and their results
        4. Retrieving and returning the agent's final response

        The method yields events during processing (like tool calls) and finally yields
        the complete Response with the agent's message.

        Args:
            messages (Sequence[BaseChatMessage]): The messages to process
            cancellation_token (CancellationToken): Token for cancellation handling
            message_limit (int, optional): Maximum number of messages to retrieve from the thread
            polling_interval (float, optional): Time to sleep between polling for run status

        Yields:
            AgentEvent | ChatMessage | Response: Events during processing and the final response

        Raises:
            ValueError: If the run fails or no message is received from the assistant
        """
        if cancellation_token is None:
            cancellation_token = CancellationToken()

        await self._ensure_initialized()

        # Process all messages in sequence
        for message in messages:
            if isinstance(message, (TextMessage, MultiModalMessage)):
                await self.handle_text_message(str(message.content), cancellation_token)
            elif isinstance(message, (StopMessage, HandoffMessage)):
                await self.handle_text_message(message.content, cancellation_token)

        # Inner messages for tool calls
        inner_messages: List[AgentEvent | ChatMessage] = []

        # Create and start a run
        run: ThreadRun = await cancellation_token.link_future(
            asyncio.ensure_future(
                self._project_client.agents.runs.create(
                    thread_id=self.thread_id,
                    agent_id=self._get_agent_id,
                )
            )
        )

        # Wait for run completion by polling
        while True:
            run = await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._project_client.agents.runs.get(
                        thread_id=self.thread_id,
                        run_id=run.id,
                    )
                )
            )

            if run.status == RunStatus.FAILED:
                raise ValueError(f"Run failed: {run.last_error}")

            # If the run requires action (function calls), execute tools and continue
            if run.status == RunStatus.REQUIRES_ACTION and run.required_action is not None:
                tool_calls: List[FunctionCall] = []
                submit_tool_outputs = getattr(run.required_action, "submit_tool_outputs", None)
                if submit_tool_outputs and hasattr(submit_tool_outputs, "tool_calls"):
                    for required_tool_call in submit_tool_outputs.tool_calls:
                        if required_tool_call.type == "function":
                            tool_calls.append(
                                FunctionCall(
                                    id=required_tool_call.id,
                                    name=required_tool_call.function.name,
                                    arguments=required_tool_call.function.arguments,
                                )
                            )

                # Add tool call message to inner messages
                tool_call_msg = ToolCallRequestEvent(source=self.name, content=tool_calls)
                inner_messages.append(tool_call_msg)
                trace_logger.debug(tool_call_msg)
                yield tool_call_msg

                # Execute tool calls and get results
                tool_outputs: List[FunctionExecutionResult] = []

                # TODO: Support parallel execution of tool calls

                for tool_call in tool_calls:
                    try:
                        result = await self._execute_tool_call(tool_call, cancellation_token)
                        is_error = False
                    except Exception as e:
                        result = f"Error: {e}"
                        is_error = True
                    tool_outputs.append(
                        FunctionExecutionResult(
                            content=result, call_id=tool_call.id, is_error=is_error, name=tool_call.name
                        )
                    )

                # Add tool result message to inner messages
                tool_result_msg = ToolCallExecutionEvent(source=self.name, content=tool_outputs)
                inner_messages.append(tool_result_msg)
                trace_logger.debug(tool_result_msg)
                yield tool_result_msg

                # Submit tool outputs back to the run
                run = await cancellation_token.link_future(
                    asyncio.ensure_future(
                        self._project_client.agents.runs.submit_tool_outputs(
                            thread_id=self.thread_id,
                            run_id=run.id,
                            tool_outputs=[ToolOutput(tool_call_id=t.call_id, output=t.content) for t in tool_outputs],
                        )
                    )
                )
                continue

            if run.status == RunStatus.COMPLETED:
                break

            # TODO support for parameter to control polling interval
            await asyncio.sleep(polling_interval)

        # After run is completed, get the messages
        trace_logger.debug("Retrieving messages from thread")
        # Collect up to message_limit messages in DESCENDING order, support cancellation
        agent_messages: List[ThreadMessage] = []
        async for msg in self._project_client.agents.messages.list(
            thread_id=self.thread_id,
            order=ListSortOrder.DESCENDING,
            limit=message_limit,
        ):
            if cancellation_token.is_cancelled():
                trace_logger.debug("Message retrieval cancelled by token.")
                break
            agent_messages.append(msg)
            if len(agent_messages) >= message_limit:
                break
        if not agent_messages:
            raise ValueError("No messages received from assistant")

        # Get the last message from the agent (role=AGENT)
        last_message: Optional[ThreadMessage] = next(
            (m for m in agent_messages if getattr(m, "role", None) == "agent"), None
        )
        if not last_message:
            trace_logger.debug("No message with AGENT role found, falling back to first message")
            last_message = agent_messages[0]  # Fallback to first message
        if not getattr(last_message, "content", None):
            raise ValueError("No content in the last message")

        # Extract text content
        message_text = ""
        for text_message in last_message.text_messages:
            message_text += text_message.text.value

        # Extract citations
        citations: list[Any] = []

        # Try accessing annotations directly

        annotations = getattr(last_message, "annotations", [])

        if isinstance(annotations, list) and annotations:
            annotations = cast(List[MessageTextUrlCitationAnnotation], annotations)

            trace_logger.debug(f"Found {len(annotations)} annotations")
            for annotation in annotations:
                if hasattr(annotation, "url_citation"):  # type: ignore
                    trace_logger.debug(f"Citation found: {annotation.url_citation.url}")
                    citations.append(
                        {"url": annotation.url_citation.url, "title": annotation.url_citation.title, "text": None}  # type: ignore
                    )
        # For backwards compatibility
        elif hasattr(last_message, "url_citation_annotations") and last_message.url_citation_annotations:
            url_annotations = cast(List[Any], last_message.url_citation_annotations)

            trace_logger.debug(f"Found {len(url_annotations)} URL citations")

            for annotation in url_annotations:
                citations.append(
                    {"url": annotation.url_citation.url, "title": annotation.url_citation.title, "text": None}  # type: ignore
                )

        elif hasattr(last_message, "file_citation_annotations") and last_message.file_citation_annotations:
            file_annotations = cast(List[Any], last_message.file_citation_annotations)

            trace_logger.debug(f"Found {len(file_annotations)} URL citations")

            for annotation in file_annotations:
                citations.append(
                    {"file_id": annotation.file_citation.file_id, "title": None, "text": annotation.file_citation.quote}  # type: ignore
                )

        trace_logger.debug(f"Total citations extracted: {len(citations)}")

        # Create the response message with citations as JSON string
        chat_message = TextMessage(
            source=self.name, content=message_text, metadata={"citations": json.dumps(citations)} if citations else {}
        )

        # Return the assistant's response as a Response with inner messages
        yield Response(chat_message=chat_message, inner_messages=inner_messages)

    async def handle_text_message(self, content: str, cancellation_token: Optional[CancellationToken] = None) -> None:
        """
        Handle a text message by adding it to the conversation thread.

        Args:
            content (str): The text content of the message
            cancellation_token (CancellationToken): Token for cancellation handling

        Returns:
            None
        """

        if cancellation_token is None:
            cancellation_token = CancellationToken()

        await cancellation_token.link_future(
            asyncio.ensure_future(
                self._project_client.agents.messages.create(
                    thread_id=self.thread_id,
                    role=MessageRole.USER,
                    content=content,
                )
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """
        Reset the agent's conversation by creating a new thread.

        This method allows for resetting a conversation without losing the agent
        definition or capabilities. It creates a new thread for fresh conversations.

        Note: Currently the Azure AI Agent API has no support for deleting messages,
        so a new thread is created instead.

        Args:
            cancellation_token (CancellationToken): Token for cancellation handling
        """
        # This will enforce the creation of a new thread
        await self._ensure_initialized(create_new_thread=True)

    async def save_state(self) -> Mapping[str, Any]:
        """
        Save the current state of the agent for future restoration.

        This method serializes the agent's state including IDs for the agent, thread,
        messages, and associated resources like vector stores and uploaded files.

        Returns:
            Mapping[str, Any]: A dictionary containing the serialized state data
        """
        state = AzureAIAgentState(
            agent_id=self._agent.id if self._agent else self._agent_id,
            thread_id=self._thread.id if self._thread else self._init_thread_id,
            initial_message_ids=list(self._initial_message_ids),
            vector_store_id=self._vector_store_id,
            uploaded_file_ids=self._uploaded_file_ids,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """
        Load a previously saved state into this agent.

        This method deserializes and restores a previously saved agent state,
        setting up the agent to continue a previous conversation or session.

        Args:
            state (Mapping[str, Any]): The previously saved state dictionary
        """
        agent_state = AzureAIAgentState.model_validate(state)
        self._agent_id = agent_state.agent_id
        self._init_thread_id = agent_state.thread_id
        self._initial_message_ids = set(agent_state.initial_message_ids)
        self._vector_store_id = agent_state.vector_store_id
        self._uploaded_file_ids = agent_state.uploaded_file_ids

    async def on_upload_for_code_interpreter(
        self,
        file_paths: str | Iterable[str],
        cancellation_token: Optional[CancellationToken] = None,
        polling_interval: float = 0.5,
    ) -> None:
        """
        Upload files to be used with the code interpreter tool.

        This method uploads files for the agent's code interpreter tool and
        updates the thread's tool resources to include these files.

        Args:
            file_paths (str | Iterable[str]): Path(s) to file(s) to upload
            cancellation_token (Optional[CancellationToken]): Token for cancellation handling
            polling_interval (float): Time to sleep between polling for file status

        Raises:
            ValueError: If file upload fails or the agent doesn't have code interpreter capability
        """
        if cancellation_token is None:
            cancellation_token = CancellationToken()

        await self._ensure_initialized()

        file_ids = await self._upload_files(
            file_paths=file_paths,
            cancellation_token=cancellation_token,
            polling_interval=polling_interval,
            purpose=FilePurpose.AGENTS,
        )

        # Update thread with the new files
        thread: AgentThread = await cancellation_token.link_future(
            asyncio.ensure_future(self._project_client.agents.threads.get(thread_id=self.thread_id))
        )

        tool_resources: ToolResources = thread.tool_resources or ToolResources()
        code_interpreter_resource = tool_resources.code_interpreter or CodeInterpreterToolResource()
        existing_file_ids: List[str] = code_interpreter_resource.file_ids or []
        existing_file_ids.extend(file_ids)

        await cancellation_token.link_future(
            asyncio.ensure_future(
                self._project_client.agents.threads.update(
                    thread_id=self.thread_id,
                    tool_resources=ToolResources(
                        code_interpreter=CodeInterpreterToolResource(file_ids=existing_file_ids)
                    ),
                )
            )
        )

    async def on_upload_for_file_search(
        self,
        file_paths: str | Iterable[str],
        cancellation_token: CancellationToken,
        vector_store_name: Optional[str] = None,
        data_sources: Optional[List[VectorStoreDataSource]] = None,
        expires_after: Optional[VectorStoreExpirationPolicy] = None,
        chunking_strategy: Optional[VectorStoreChunkingStrategyRequest] = None,
        vector_store_metadata: Optional[Dict[str, str]] = None,
        vector_store_polling_interval: float = 1,
    ) -> None:
        """
        Upload files to be used with the file search tool.

        This method handles uploading files for the file search capability, creating a vector
        store if necessary, and updating the agent's configuration to use the vector store.

        Args:
            file_paths (str | Iterable[str]): Path(s) to file(s) to upload
            cancellation_token (CancellationToken): Token for cancellation handling
            vector_store_name (Optional[str]): Name to assign to the vector store if creating a new one
            data_sources (Optional[List[VectorStoreDataSource]]): Additional data sources for the vector store
            expires_after (Optional[VectorStoreExpirationPolicy]): Expiration policy for vector store content
            chunking_strategy (Optional[VectorStoreChunkingStrategyRequest]): Strategy for chunking file content
            vector_store_metadata (Optional[Dict[str, str]]): Additional metadata for the vector store
            vector_store_polling_interval (float): Time to sleep between polling for vector store status

        Raises:
            ValueError: If file search is not enabled for this agent or file upload fails
        """
        await self._ensure_initialized()

        # Check if file_search is enabled in tools
        if not any(isinstance(tool, FileSearchToolDefinition) for tool in self._api_tools):
            raise ValueError(
                "File search is not enabled for this assistant. Add a file_search tool when creating the assistant."
            )

        # Create vector store if not already created
        if self._vector_store_id is None:
            vector_store: VectorStore = await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._project_client.agents.vector_stores.create_and_poll(
                        file_ids=[],
                        name=vector_store_name,
                        data_sources=data_sources,
                        expires_after=expires_after,
                        chunking_strategy=chunking_strategy,
                        metadata=vector_store_metadata,
                        polling_interval=vector_store_polling_interval,
                    )
                )
            )
            self._vector_store_id = vector_store.id

            # Update assistant with vector store ID
            await cancellation_token.link_future(
                asyncio.ensure_future(
                    self._project_client.agents.update_agent(
                        agent_id=self._get_agent_id,
                        tools=self._api_tools,
                        tool_resources=ToolResources(
                            file_search=FileSearchToolResource(vector_store_ids=[self._vector_store_id])
                        ),
                    )
                )
            )

        file_ids = await self._upload_files(
            file_paths=file_paths, cancellation_token=cancellation_token, purpose=FilePurpose.AGENTS
        )

        # Create file batch with the file IDs
        await cancellation_token.link_future(
            asyncio.ensure_future(
                self._project_client.agents.vector_store_file_batches.create_and_poll(
                    vector_store_id=self._vector_store_id,
                    file_ids=file_ids,
                    polling_interval=vector_store_polling_interval,
                )
            )
        )

    async def close(self) -> None:
        """
        Close the Azure AI agent and release any resources.
        """
        await self._project_client.close()

# From azure/_azure_ai_agent.py
def thread_id(self) -> str:
        if self._thread is None:
            raise ValueError("Thread not initialized")
        return self._thread.id

# From azure/_azure_ai_agent.py
def deployment_name(self) -> str:
        if not self._deployment_name:
            raise ValueError("Deployment name not initialized")
        return self._deployment_name

# From azure/_azure_ai_agent.py
def instructions(self) -> str:
        if not self._instructions:
            raise ValueError("Instructions not initialized")
        return self._instructions

from typing import TypeGuard

# From azure/_types.py
class AzureAIAgentState(BaseModel):
    """
    Represents the state of an AzureAIAgent that can be saved and loaded.

    This state model keeps track of persistent information about an agent session
    including agent and thread identifiers, message history, and associated resources.

    Attributes:
        type (str): The type identifier for the state object, always "AzureAIAgentState"
        agent_id (Optional[str]): The ID of the Azure AI agent
        thread_id (Optional[str]): The ID of the conversation thread
        initial_message_ids (List[str]): List of message IDs from the initial state
        vector_store_id (Optional[str]): The ID of the associated vector store for file search
        uploaded_file_ids (List[str]): List of IDs for files uploaded to the agent
    """

    type: str = Field(default="AzureAIAgentState")
    agent_id: Optional[str] = None
    thread_id: Optional[str] = None
    initial_message_ids: List[str] = Field(default_factory=list)
    vector_store_id: Optional[str] = None
    uploaded_file_ids: List[str] = Field(default_factory=list)

# From azure/_types.py
def has_annotations(obj: Any) -> TypeGuard[list[MessageTextUrlCitationAnnotation]]:
    return obj is not None and isinstance(obj, list)


# From magentic_one/_magentic_one_coder_agent.py
class MagenticOneCoderAgent(AssistantAgent):
    """An agent, used by MagenticOne that provides coding assistance using an LLM model client.

    The prompts and description are sealed, to replicate the original MagenticOne configuration. See AssistantAgent if you wish to modify these values.
    """

    component_provider_override = "autogen_ext.agents.magentic_one.MagenticOneCoderAgent"

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        **kwargs: Any,
    ):
        super().__init__(
            name,
            model_client,
            description=MAGENTIC_ONE_CODER_DESCRIPTION,
            system_message=MAGENTIC_ONE_CODER_SYSTEM_MESSAGE,
        )

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from _semantic_router_components import FinalResult
from _semantic_router_components import TerminationMessage
from _semantic_router_components import UserProxyMessage
from _semantic_router_components import WorkerAgentMessage
from autogen_core import message_handler

# From core_semantic_router/_agents.py
class WorkerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__("A Worker Agent")
        self._name = name

    @message_handler
    async def my_message_handler(self, message: UserProxyMessage, ctx: MessageContext) -> None:
        assert ctx.topic_id is not None
        logger.debug(f"Received message from {message.source}: {message.content}")
        if "END" in message.content:
            await self.publish_message(
                TerminationMessage(reason="user terminated conversation", content=message.content, source=self.type),
                topic_id=DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
            )
        else:
            content = f"Hello from {self._name}! You said: {message.content}"
            logger.debug(f"Returning message: {content}")
            await self.publish_message(
                WorkerAgentMessage(content=content, source=ctx.topic_id.type),
                topic_id=DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
            )

from _semantic_router_components import AgentRegistryBase
from _semantic_router_components import IntentClassifierBase
from autogen_core import default_subscription

# From core_semantic_router/_semantic_router_agent.py
class SemanticRouterAgent(RoutedAgent):
    def __init__(self, name: str, agent_registry: AgentRegistryBase, intent_classifier: IntentClassifierBase) -> None:
        super().__init__("Semantic Router Agent")
        self._name = name
        self._registry = agent_registry
        self._classifier = intent_classifier

    # The User has sent a message that needs to be routed
    @message_handler
    async def route_to_agent(self, message: UserProxyMessage, ctx: MessageContext) -> None:
        assert ctx.topic_id is not None
        logger.debug(f"Received message from {message.source}: {message.content}")
        session_id = ctx.topic_id.source
        intent = await self._identify_intent(message)
        agent = await self._find_agent(intent)
        await self.contact_agent(agent, message, session_id)

    ## Identify the intent of the user message
    async def _identify_intent(self, message: UserProxyMessage) -> str:
        return await self._classifier.classify_intent(message.content)

    ## Use a lookup, search, or LLM to identify the most relevant agent for the intent
    async def _find_agent(self, intent: str) -> str:
        logger.debug(f"Identified intent: {intent}")
        try:
            agent = await self._registry.get_agent(intent)
            return agent
        except KeyError:
            logger.debug("No relevant agent found for intent: " + intent)
            return "termination"

    ## Forward user message to the appropriate agent, or end the thread.
    async def contact_agent(self, agent: str, message: UserProxyMessage, session_id: str) -> None:
        if agent == "termination":
            logger.debug("No relevant agent found")
            await self.publish_message(
                TerminationMessage(reason="No relevant agent found", content=message.content, source=self.type),
                DefaultTopicId(type="user_proxy", source=session_id),
            )
        else:
            logger.debug("Routing to agent: " + agent)
            await self.publish_message(
                UserProxyMessage(content=message.content, source=message.source),
                DefaultTopicId(type=agent, source=session_id),
            )


# From core_semantic_router/_semantic_router_components.py
class IntentClassifierBase(ABC):
    @abstractmethod
    async def classify_intent(self, message: str) -> str:
        pass

# From core_semantic_router/_semantic_router_components.py
class AgentRegistryBase(ABC):
    @abstractmethod
    async def get_agent(self, intent: str) -> str:
        pass

# From core_semantic_router/_semantic_router_components.py
class UserProxyMessage(TextMessage):
    """A message that is sent from the user to the system, and needs to be routed to the appropriate agent."""

    pass

# From core_semantic_router/_semantic_router_components.py
class TerminationMessage(TextMessage):
    """A message that is sent from the system to the user, indicating that the conversation has ended."""

    reason: str

# From core_semantic_router/_semantic_router_components.py
class WorkerAgentMessage(TextMessage):
    """A message that is sent from a worker agent to the user."""

    pass

# From core_semantic_router/_semantic_router_components.py
class FinalResult(TextMessage):
    """A message sent from the agent to the user, indicating the end of a conversation"""

    pass

import platform
from _agents import UserProxyAgent
from _agents import WorkerAgent
from _semantic_router_agent import SemanticRouterAgent
from autogen_core import ClosureAgent
from autogen_core import ClosureContext

# From core_semantic_router/run_semantic_router.py
class MockIntentClassifier(IntentClassifierBase):
    def __init__(self):
        self.intents = {
            "finance_intent": ["finance", "money", "budget"],
            "hr_intent": ["hr", "human resources", "employee"],
        }

    async def classify_intent(self, message: str) -> str:
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in message:
                    return intent
        return "general"

# From core_semantic_router/run_semantic_router.py
class MockAgentRegistry(AgentRegistryBase):
    def __init__(self):
        self.agents = {"finance_intent": "finance", "hr_intent": "hr"}

    async def get_agent(self, intent: str) -> str:
        return self.agents[intent]

import argparse
from typing import Annotated
from autogen_core import SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.tool_agent import ToolAgent
from autogen_core.tool_agent import tool_agent_caller_loop
from autogen_core.tools import ToolSchema
from chess import BLACK
from chess import SQUARE_NAMES
from chess import WHITE
from chess import Board
from chess import Move
from chess import piece_name

# From core_chess_game/main.py
class PlayerAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        instructions: str,
        model_client: ChatCompletionClient,
        model_context: ChatCompletionContext,
        tool_schema: List[ToolSchema],
        tool_agent_type: str,
    ) -> None:
        super().__init__(description=description)
        self._system_messages: List[LLMMessage] = [SystemMessage(content=instructions)]
        self._model_client = model_client
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)
        self._model_context = model_context

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> None:
        # Add the user message to the model context.
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))
        # Run the caller loop to handle tool calls.
        messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=self._system_messages + (await self._model_context.get_messages()),
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        # Add the assistant message to the model context.
        for msg in messages:
            await self._model_context.add_message(msg)
        # Publish the final response.
        assert isinstance(messages[-1].content, str)
        await self.publish_message(TextMessage(content=messages[-1].content, source=self.id.type), DefaultTopicId())

# From core_chess_game/main.py
def validate_turn(board: Board, player: Literal["white", "black"]) -> None:
    """Validate that it is the player's turn to move."""
    last_move = board.peek() if board.move_stack else None
    if last_move is not None:
        if player == "white" and board.color_at(last_move.to_square) == WHITE:
            raise ValueError("It is not your turn to move. Wait for black to move.")
        if player == "black" and board.color_at(last_move.to_square) == BLACK:
            raise ValueError("It is not your turn to move. Wait for white to move.")
    elif last_move is None and player != "white":
        raise ValueError("It is not your turn to move. Wait for white to move first.")

# From core_chess_game/main.py
def get_legal_moves(
    board: Board, player: Literal["white", "black"]
) -> Annotated[str, "A list of legal moves in UCI format."]:
    """Get legal moves for the given player."""
    validate_turn(board, player)
    legal_moves = list(board.legal_moves)
    if player == "black":
        legal_moves = [move for move in legal_moves if board.color_at(move.from_square) == BLACK]
    elif player == "white":
        legal_moves = [move for move in legal_moves if board.color_at(move.from_square) == WHITE]
    else:
        raise ValueError("Invalid player, must be either 'black' or 'white'.")
    if not legal_moves:
        return "No legal moves. The game is over."

    return "Possible moves are: " + ", ".join([move.uci() for move in legal_moves])

# From core_chess_game/main.py
def get_board(board: Board) -> str:
    """Get the current board state."""
    return str(board)

# From core_chess_game/main.py
def make_move(
    board: Board,
    player: Literal["white", "black"],
    thinking: Annotated[str, "Thinking for the move."],
    move: Annotated[str, "A move in UCI format."],
) -> Annotated[str, "Result of the move."]:
    """Make a move on the board."""
    validate_turn(board, player)
    new_move = Move.from_uci(move)
    board.push(new_move)

    # Print the move.
    print("-" * 50)
    print("Player:", player)
    print("Move:", new_move.uci())
    print("Thinking:", thinking)
    print("Board:")
    print(board.unicode(borders=True))

    # Get the piece name.
    piece = board.piece_at(new_move.to_square)
    assert piece is not None
    piece_symbol = piece.unicode_symbol()
    piece_name = get_piece_name(piece.piece_type)
    if piece_symbol.isupper():
        piece_name = piece_name.capitalize()
    return f"Moved {piece_name} ({piece_symbol}) from {SQUARE_NAMES[new_move.from_square]} to {SQUARE_NAMES[new_move.to_square]}."

# From core_chess_game/main.py
def get_legal_moves_black() -> str:
        return get_legal_moves(board, "black")

# From core_chess_game/main.py
def get_legal_moves_white() -> str:
        return get_legal_moves(board, "white")

# From core_chess_game/main.py
def make_move_black(
        thinking: Annotated[str, "Thinking for the move"],
        move: Annotated[str, "A move in UCI format"],
    ) -> str:
        return make_move(board, "black", thinking, move)

# From core_chess_game/main.py
def make_move_white(
        thinking: Annotated[str, "Thinking for the move"],
        move: Annotated[str, "A move in UCI format"],
    ) -> str:
        return make_move(board, "white", thinking, move)

# From core_chess_game/main.py
def get_board_text() -> Annotated[str, "The current board state"]:
        return get_board(board)

from models import UserTask
from models import AgentResponse

# From core_streaming_handoffs_fastapi/agent_base.py
class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str,
        response_queue : asyncio.Queue[str | object]
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._delegate_tools = dict([(tool.name, tool) for tool in delegate_tools])
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type
        self._response_queue = response_queue

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        # Start streaming LLM responses
        llm_stream = self._model_client.create_stream(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token
        )
        final_response = None
        async for chunk in llm_stream:
            if isinstance(chunk, str):
                await self._response_queue.put({'type': "string", 'message': chunk})
            else:
                final_response = chunk
        assert final_response is not None, "No response from model"
        print(f"{'-'*80}\n{self.id.type}:\n{final_response.content}", flush=True)
        # Process the LLM result.
        while isinstance(final_response.content, list) and all(isinstance(m, FunctionCall) for m in final_response.content):
            tool_call_results: List[FunctionExecutionResult] = []
            delegate_targets: List[Tuple[str, UserTask]] = []
            # Process each function call.
            for call in final_response.content:
                arguments = json.loads(call.arguments)
                await self._response_queue.put({"type":"function","message":f"Executing {call.name}"})
                if call.name in self._tools:
                    # Execute the tool directly.
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token, call_id=call.id)
                    result_as_str = self._tools[call.name].return_value_as_string(result)
                    tool_call_results.append(
                        FunctionExecutionResult(call_id=call.id, content=result_as_str, is_error=False, name=call.name)
                    )
                elif call.name in self._delegate_tools:
                    # Execute the tool to get the delegate agent's topic type.
                    result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token, call_id=call.id)
                    topic_type = self._delegate_tools[call.name].return_value_as_string(result)
                    # Create the context for the delegate agent, including the function call and the result.
                    delegate_messages = list(message.context) + [
                        AssistantMessage(content=[call], source=self.id.type),
                        FunctionExecutionResultMessage(
                            content=[
                                FunctionExecutionResult(
                                    call_id=call.id,
                                    content=f"Transferred to {topic_type}. Adopt persona immediately.",
                                    is_error=False,
                                    name=call.name,
                                )
                            ]
                        ),
                    ]
                    delegate_targets.append((topic_type, UserTask(context=delegate_messages)))
                else:
                    raise ValueError(f"Unknown tool: {call.name}")
            if len(delegate_targets) > 0:
                # Delegate the task to other agents by publishing messages to the corresponding topics.
                for topic_type, task in delegate_targets:
                    print(f"{'-'*80}\n{self.id.type}:\nDelegating to {topic_type}", flush=True)
                    await self._response_queue.put({"type":"function","message":f"You are now talking to {topic_type}"})
                    await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))
            if len(tool_call_results) > 0:
                print(f"{'-'*80}\n{self.id.type}:\n{tool_call_results}", flush=True)
                # Make another LLM call with the results.
                message.context.extend([
                    AssistantMessage(content=final_response.content, source=self.id.type),
                    FunctionExecutionResultMessage(content=tool_call_results),
                ])
                llm_stream = self._model_client.create_stream(
                    messages=[self._system_message] + message.context,
                    tools=self._tool_schema + self._delegate_tool_schema,
                    cancellation_token=ctx.cancellation_token
                )
                final_response = None
                async for chunk in llm_stream:
                    if isinstance(chunk, str):
                        await self._response_queue.put({'type': 'string', 'message': chunk})
                    else:
                        final_response = chunk
                assert final_response is not None, "No response from model"
                print(f"{'-'*80}\n{self.id.type}:\n{final_response.content}", flush=True)
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(final_response.content, str)
        message.context.append(AssistantMessage(content=final_response.content, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )


# From core_streaming_handoffs_fastapi/agent_user.py
class UserAgent(RoutedAgent):
    def __init__(self, 
                 description: str, 
                 user_topic_type: str, 
                 agent_topic_type: str, 
                 response_queue : asyncio.Queue[str | object], 
                 stream_done : object) -> None:
        super().__init__(description)
        self._user_topic_type = user_topic_type
        self._agent_topic_type = agent_topic_type
        self._response_queue = response_queue
        self._STREAM_DONE = stream_done

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
        #Save chat history
        context = BufferedChatCompletionContext(buffer_size=10,initial_messages=message.context)
        save_context = await context.save_state()
        # Save context to JSON file
        chat_history_dir = "chat_history"
        if ctx.topic_id is None:
            raise ValueError("MessageContext.topic_id is None, cannot save chat history")
        file_path = os.path.join(chat_history_dir, f"history-{ctx.topic_id.source}.json")
        with open(file_path, 'w') as f:
            json.dump(save_context, f, indent=4)
        
        #End stream
        await self._response_queue.put(self._STREAM_DONE)


# From core_streaming_handoffs_fastapi/models.py
class UserLogin(BaseModel):
    pass

# From core_streaming_handoffs_fastapi/models.py
class UserTask(BaseModel):
    context: List[LLMMessage]

# From core_streaming_handoffs_fastapi/models.py
class AgentResponse(BaseModel):
    reply_to_topic_type: str
    context: List[LLMMessage]



# From core_chainlit/SimpleAssistantAgent.py
class Message:
    content: str

# From core_chainlit/SimpleAssistantAgent.py
class StreamResult(BaseModel):
    content: str | CreateResult | AssistantMessage
    source: str

# From core_chainlit/SimpleAssistantAgent.py
class RequestToSpeak(BaseModel):
    pass

# From core_chainlit/SimpleAssistantAgent.py
class SimpleAssistantAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        system_message: str,
        #context: MessageContext,
        model_client: ChatCompletionClient,
        tool_schema: List[Tool] = [],
        model_client_stream: bool = False,
        reflect_on_tool_use: bool | None = None,
        group_chat_topic_type: str = "Default",
    ) -> None:
        super().__init__(name)
        self._system_message = SystemMessage(content=system_message)
        self._model_client = model_client
        self._tools = tool_schema
        #self._model_context = context 
        self._model_client_stream = model_client_stream
        self._reflect_on_tool_use = reflect_on_tool_use
        self._group_chat_topic_type = group_chat_topic_type
        self._chat_history: List[LLMMessage] = []

    async def _call_model_client(
        self, cancellation_token: CancellationToken
    ) -> AsyncGenerator[str | CreateResult, None]:
        # Call the LLM model to process the message 
        model_result = None
        async for chunk in self._model_client.create_stream(
            messages=[self._system_message] + self._chat_history,
            tools=self._tools,
            cancellation_token=cancellation_token,
        ):
            if isinstance(chunk, CreateResult):
                model_result = chunk
            elif isinstance(chunk, str):
                yield chunk
            else:
                raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
        
        if model_result is None:    # No final result in model client respons
            raise RuntimeError("No final model result in streaming mode.")

        yield model_result
        return

    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token, call_id=call.id)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)

    @message_handler
    async def handle_user_message(self, message: UserMessage, ctx: MessageContext) -> Message:

        # Append the message to chat history.
        self._chat_history.append(
           message 
        )

        # Add message to model context.
        # await self._model_context.add_message(UserMessage(content=message.content, source="User"))
        model_result: Optional[CreateResult] = None

        async for chunk in self._call_model_client(
            cancellation_token=ctx.cancellation_token,
        ):
            if isinstance(chunk, CreateResult):
                model_result = chunk
            elif isinstance(chunk, str):
                # foward the stream tokent to the Queue
                await self.runtime.publish_message(StreamResult(content=chunk, source=self.id.type), topic_id=task_results_topic_id)
            else:
                raise RuntimeError(f"Invalid chunk type: {type(chunk)}")

        if model_result is None:    # No final result in model client respons
            raise RuntimeError("No final model result in streaming mode.")

        # Add the first model create result to the session.
        self._chat_history.append(AssistantMessage(content=model_result.content, source=self.id.type))

        if isinstance(model_result.content, str):    # No tools, return the result
            await self.runtime.publish_message(StreamResult(content=model_result, source=self.id.type), topic_id=task_results_topic_id)
            return (Message(content= model_result.content))

        # Execute the tool calls.
        assert isinstance(model_result.content, list) and all(
            isinstance(call, FunctionCall) for call in model_result.content
        )
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in model_result.content]
        )

        # Add the function execution results to the session.
        self._chat_history.append(FunctionExecutionResultMessage(content=results))

        #if (not self._reflect_on_tool_use):
        #    return Message(content=model_result.content)
        
        # Run the chat completion client again to reflect on the history and function execution results.
        #model_result = None
        model_result2: Optional[CreateResult] = None
        async for chunk in self._call_model_client(
            cancellation_token=ctx.cancellation_token,
        ):
            if isinstance(chunk, CreateResult):
                model_result2 = chunk
            elif isinstance(chunk, str):
                # foward the stream tokent to the Queue
                await self.runtime.publish_message(StreamResult(content=chunk, source=self.id.type), topic_id=task_results_topic_id)
            else:
                raise RuntimeError(f"Invalid chunk type: {type(chunk)}")

        if model_result2 is None:
            raise RuntimeError("No final model result in streaming mode.")
        assert model_result2.content is not None 
        assert isinstance(model_result2.content, str)

        await self.runtime.publish_message(StreamResult(content=model_result2, source=self.id.type), topic_id=task_results_topic_id)

        return Message(content=model_result2.content)

    # Message handler for Group chat message. It just add the message to the agent message history.
    # The message will be processed when the agent receives the RequestToSpeak. 
    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),
                message.body,
            ]
        )

    # Message handler for request to speaker message.
    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        #print(f"### {self.id.type}: ")
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )

        # Run the chat completion client again to reflect on the history and function execution results.
        model_result: Optional[CreateResult] = None
        async for chunk in self._call_model_client(
            cancellation_token=ctx.cancellation_token,
        ):
            if isinstance(chunk, CreateResult):
                model_result = chunk
                await self.runtime.publish_message(StreamResult(content=model_result, source=self.id.type), topic_id=task_results_topic_id)
            elif isinstance(chunk, str):
                # foward the stream tokent to the Queue
                await self.runtime.publish_message(StreamResult(content=chunk, source=self.id.type), topic_id=task_results_topic_id)
            else:
                raise RuntimeError(f"Invalid chunk type: {type(chunk)}")

        if model_result is None:
            raise RuntimeError("No final model result in streaming mode.")

        assert isinstance(model_result.content, str)
        assert model_result.content is not None

        self._chat_history.append(AssistantMessage(content=model_result.content, source=self.id.type))
        #print(model_result.content, flush=True)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=model_result.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )

import chainlit
from SimpleAssistantAgent import SimpleAssistantAgent
from SimpleAssistantAgent import StreamResult

from autogen_ext.experimental.task_centric_memory import MemoryController
from autogen_ext.experimental.task_centric_memory.utils import Teachability

import datetime
from concurrent.futures import ThreadPoolExecutor
from autogen_core import DefaultInterventionHandler
from autogen_core import type_subscription

# From core_async_human_in_the_loop/main.py
class UserTextMessage(TextMessage):
    pass

# From core_async_human_in_the_loop/main.py
class AssistantTextMessage(TextMessage):
    pass

# From core_async_human_in_the_loop/main.py
class GetSlowUserMessage:
    content: str

# From core_async_human_in_the_loop/main.py
class TerminateMessage:
    content: str

# From core_async_human_in_the_loop/main.py
class MockPersistence:
    def __init__(self):
        self._content: Mapping[str, Any] = {}

    def load_content(self) -> Mapping[str, Any]:
        return self._content

    def save_content(self, content: Mapping[str, Any]) -> None:
        self._content = content

# From core_async_human_in_the_loop/main.py
class SlowUserProxyAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        description: str,
    ) -> None:
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(buffer_size=5)
        self._name = name

    @message_handler
    async def handle_message(self, message: AssistantTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(AssistantMessage(content=message.content, source=message.source))
        await self.publish_message(
            GetSlowUserMessage(content=message.content), topic_id=DefaultTopicId("scheduling_assistant_conversation")
        )

    async def save_state(self) -> Mapping[str, Any]:
        state_to_save = {
            "memory": await self._model_context.save_state(),
        }
        return state_to_save

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])

# From core_async_human_in_the_loop/main.py
class ScheduleMeetingInput(BaseModel):
    recipient: str = Field(description="Name of recipient")
    date: str = Field(description="Date of meeting")
    time: str = Field(description="Time of meeting")

# From core_async_human_in_the_loop/main.py
class ScheduleMeetingOutput(BaseModel):
    pass

# From core_async_human_in_the_loop/main.py
class ScheduleMeetingTool(BaseTool[ScheduleMeetingInput, ScheduleMeetingOutput]):
    def __init__(self):
        super().__init__(
            ScheduleMeetingInput,
            ScheduleMeetingOutput,
            "schedule_meeting",
            "Schedule a meeting with a recipient at a specific date and time",
        )

    async def run(self, args: ScheduleMeetingInput, cancellation_token: CancellationToken) -> ScheduleMeetingOutput:
        print(f"Meeting scheduled with {args.recipient} on {args.date} at {args.time}")
        return ScheduleMeetingOutput()

# From core_async_human_in_the_loop/main.py
class SchedulingAssistantAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        description: str,
        model_client: ChatCompletionClient,
        initial_message: AssistantTextMessage | None = None,
    ) -> None:
        super().__init__(description)
        self._model_context = BufferedChatCompletionContext(
            buffer_size=5,
            initial_messages=[UserMessage(content=initial_message.content, source=initial_message.source)]
            if initial_message
            else None,
        )
        self._name = name
        self._model_client = model_client
        self._system_messages = [
            SystemMessage(
                content=f"""
I am a helpful AI assistant that helps schedule meetings.
If there are missing parameters, I will ask for them.

Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
            )
        ]

    @message_handler
    async def handle_message(self, message: UserTextMessage, ctx: MessageContext) -> None:
        await self._model_context.add_message(UserMessage(content=message.content, source=message.source))

        tools = [ScheduleMeetingTool()]
        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()), tools=tools
        )

        if isinstance(response.content, list) and all(isinstance(item, FunctionCall) for item in response.content):
            for call in response.content:
                tool = next((tool for tool in tools if tool.name == call.name), None)
                if tool is None:
                    raise ValueError(f"Tool not found: {call.name}")
                arguments = json.loads(call.arguments)
                await tool.run_json(arguments, ctx.cancellation_token, call_id=call.id)
            await self.publish_message(
                TerminateMessage(content="Meeting scheduled"),
                topic_id=DefaultTopicId("scheduling_assistant_conversation"),
            )
            return

        assert isinstance(response.content, str)
        speech = AssistantTextMessage(content=response.content, source=self.metadata["type"])
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))

        await self.publish_message(speech, topic_id=DefaultTopicId("scheduling_assistant_conversation"))

    async def save_state(self) -> Mapping[str, Any]:
        return {
            "memory": await self._model_context.save_state(),
        }

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._model_context.load_state(state["memory"])

# From core_async_human_in_the_loop/main.py
class NeedsUserInputHandler(DefaultInterventionHandler):
    def __init__(self):
        self.question_for_user: GetSlowUserMessage | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, GetSlowUserMessage):
            self.question_for_user = message
        return message

    @property
    def needs_user_input(self) -> bool:
        return self.question_for_user is not None

    @property
    def user_input_content(self) -> str | None:
        if self.question_for_user is None:
            return None
        return self.question_for_user.content

# From core_async_human_in_the_loop/main.py
class TerminationHandler(DefaultInterventionHandler):
    def __init__(self):
        self.terminateMessage: TerminateMessage | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, TerminateMessage):
            self.terminateMessage = message
        return message

    @property
    def is_terminated(self) -> bool:
        return self.terminateMessage is not None

    @property
    def termination_msg(self) -> str | None:
        if self.terminateMessage is None:
            return None
        return self.terminateMessage.content

# From core_async_human_in_the_loop/main.py
def load_content(self) -> Mapping[str, Any]:
        return self._content

# From core_async_human_in_the_loop/main.py
def save_content(self, content: Mapping[str, Any]) -> None:
        self._content = content

# From core_async_human_in_the_loop/main.py
def needs_user_input(self) -> bool:
        return self.question_for_user is not None

# From core_async_human_in_the_loop/main.py
def user_input_content(self) -> str | None:
        if self.question_for_user is None:
            return None
        return self.question_for_user.content

# From core_async_human_in_the_loop/main.py
def is_terminated(self) -> bool:
        return self.terminateMessage is not None

# From core_async_human_in_the_loop/main.py
def termination_msg(self) -> str | None:
        if self.terminateMessage is None:
            return None
        return self.terminateMessage.content

# From core_async_human_in_the_loop/main.py
def get_user_input(question_for_user: str):
        print("--------------------------QUESTION_FOR_USER--------------------------")
        print(question_for_user)
        print("---------------------------------------------------------------------")
        user_input = input("Enter your input: ")
        return user_input



from autogen_agentchat.messages import ModelClientStreamingChunkEvent

import random
from uuid import uuid4
from _types import GroupChatMessage
from _types import MessageChunk
from _types import RequestToSpeak
from _types import UIAgentConfig
from rich.console import Console
from rich.markdown import Markdown

# From core_distributed-group-chat/_agents.py
class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_message: str,
        ui_config: UIAgentConfig,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []
        self._ui_config = ui_config
        self.console = Console()

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),  # type: ignore[union-attr]
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))

        console_message = f"\n{'-'*80}\n**{self.id.type}**: {completion.content}"
        self.console.print(Markdown(console_message))

        await publish_message_to_ui_and_backend(
            runtime=self,
            source=self.id.type,
            user_message=completion.content,
            ui_config=self._ui_config,
            group_chat_topic_type=self._group_chat_topic_type,
        )

# From core_distributed-group-chat/_agents.py
class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        ui_config: UIAgentConfig,
        max_rounds: int = 3,
    ) -> None:
        super().__init__("Group chat manager")
        self._model_client = model_client
        self._num_rounds = 0
        self._participant_topic_types = participant_topic_types
        self._chat_history: List[GroupChatMessage] = []
        self._max_rounds = max_rounds
        self.console = Console()
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None
        self._ui_config = ui_config

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        assert isinstance(message.body, UserMessage)

        self._chat_history.append(message.body)  # type: ignore[reportargumenttype,arg-type]

        # Format message history.
        messages: List[str] = []
        for msg in self._chat_history:
            if isinstance(msg.content, str):  # type: ignore[attr-defined]
                messages.append(f"{msg.source}: {msg.content}")  # type: ignore[attr-defined]
            elif isinstance(msg.content, list):  # type: ignore[attr-defined]
                messages.append(f"{msg.source}: {', '.join(msg.content)}")  # type: ignore[attr-defined,reportUnknownArgumentType]
        history = "\n".join(messages)
        # Format roles.
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types, self._participant_descriptions, strict=True
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        participants = str(
            [
                topic_type
                for topic_type in self._participant_topic_types
                if topic_type != self._previous_participant_topic_type
            ]
        )

        selector_prompt = f"""You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. if you think it's enough talking (for example they have talked for {self._max_rounds} rounds), return 'FINISH'.
"""
        system_message = SystemMessage(content=selector_prompt)
        completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)

        assert isinstance(
            completion.content, str
        ), f"Completion content must be a string, but is: {type(completion.content)}"

        if completion.content.upper() == "FINISH":
            finish_msg = "I think it's enough iterations on the story! Thanks for collaborating!"
            manager_message = f"\n{'-'*80}\n Manager ({id(self)}): {finish_msg}"
            await publish_message_to_ui(
                runtime=self, source=self.id.type, user_message=finish_msg, ui_config=self._ui_config
            )
            self.console.print(Markdown(manager_message))
            return

        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                self.console.print(
                    Markdown(f"\n{'-'*80}\n Manager ({id(self)}): Asking `{selected_topic_type}` to speak")
                )
                await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
                return
        raise ValueError(f"Invalid role selected: {completion.content}")

# From core_distributed-group-chat/_agents.py
class UIAgent(RoutedAgent):
    """Handles UI-related tasks and message processing for the distributed group chat system."""

    def __init__(self, on_message_chunk_func: Callable[[MessageChunk], Awaitable[None]]) -> None:
        super().__init__("UI Agent")
        self._on_message_chunk_func = on_message_chunk_func

    @message_handler
    async def handle_message_chunk(self, message: MessageChunk, ctx: MessageContext) -> None:
        await self._on_message_chunk_func(message)

from _agents import BaseGroupChatAgent
from _types import AppConfig
from _utils import get_serializers
from _utils import load_config
from _utils import set_all_log_levels
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


from autogen_ext.models.openai.config import AzureOpenAIClientConfiguration

# From core_distributed-group-chat/_types.py
class MessageChunk:
    message_id: str
    text: str
    author: str
    finished: bool

    def __str__(self) -> str:
        return f"{self.author}({self.message_id}): {self.text}"

# From core_distributed-group-chat/_types.py
class HostConfig(BaseModel):
    hostname: str
    port: int

    @property
    def address(self) -> str:
        return f"{self.hostname}:{self.port}"

# From core_distributed-group-chat/_types.py
class GroupChatManagerConfig(BaseModel):
    topic_type: str
    max_rounds: int

# From core_distributed-group-chat/_types.py
class ChatAgentConfig(BaseModel):
    topic_type: str
    description: str
    system_message: str

# From core_distributed-group-chat/_types.py
class UIAgentConfig(BaseModel):
    topic_type: str
    artificial_stream_delay_seconds: Dict[str, float]

    @property
    def min_delay(self) -> float:
        return self.artificial_stream_delay_seconds.get("min", 0.0)

    @property
    def max_delay(self) -> float:
        return self.artificial_stream_delay_seconds.get("max", 0.0)

# From core_distributed-group-chat/_types.py
class AppConfig(BaseModel):
    host: HostConfig
    group_chat_manager: GroupChatManagerConfig
    writer_agent: ChatAgentConfig
    editor_agent: ChatAgentConfig
    ui_agent: UIAgentConfig
    client_config: AzureOpenAIClientConfiguration = None

# From core_distributed-group-chat/_types.py
def address(self) -> str:
        return f"{self.hostname}:{self.port}"

# From core_distributed-group-chat/_types.py
def min_delay(self) -> float:
        return self.artificial_stream_delay_seconds.get("min", 0.0)

# From core_distributed-group-chat/_types.py
def max_delay(self) -> float:
        return self.artificial_stream_delay_seconds.get("max", 0.0)

import time
from contextlib import asynccontextmanager
from fastapi import Request
from fastapi.responses import StreamingResponse

# From core_streaming_response_fastapi/app.py
class UserRequest:
    """
    Represents the chat history, containing a list of messages.
    Each message is expected to be a dictionary with 'source' and 'content' keys.
    """

    messages: List[Dict[str, str]]

# From core_streaming_response_fastapi/app.py
class MyAgent(RoutedAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient) -> None:
        super().__init__(name)
        self._system_messages = [SystemMessage(content="You are a helpful assistant.")]
        self._model_client = model_client
        self._response_queue = response_queue

    @message_handler
    async def handle_user_message(self, message: UserRequest, ctx: MessageContext) -> AgentResponse:
        accumulated_content = ""  # To store the full response.
        try:
            _message = message.messages
            user_messages: List[LLMMessage] = []
            for m in _message:
                if m["source"] == "user":
                    user_messages.append(UserMessage(content=m["source"], source=m["source"]))
                else:
                    user_messages.append(AssistantMessage(content=m["source"], source=m["source"]))
            # Create a stream of messages to the model client.
            async for i in self._model_client.create_stream(user_messages, cancellation_token=ctx.cancellation_token):
                if isinstance(i, str):
                    accumulated_content += i
                    await self._response_queue.put(i)
                else:
                    break
            await self._response_queue.put(STREAM_DONE)
            return AgentResponse(content=accumulated_content)
        except Exception as e:
            await self._response_queue.put("ERROR:" + str(e))
            return AgentResponse(content=str(e))

from typing import NoReturn

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class AskToGreet:
    content: str

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class Greeting:
    content: str

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class ReturnedGreeting:
    content: str

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class Feedback:
    content: str

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class ReturnedFeedback:
    content: str

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class ReceiveAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Receive Agent")

    @message_handler
    async def on_greet(self, message: Greeting, ctx: MessageContext) -> None:
        await self.publish_message(ReturnedGreeting(f"Returned greeting: {message.content}"), topic_id=DefaultTopicId())

    @message_handler
    async def on_feedback(self, message: Feedback, ctx: MessageContext) -> None:
        await self.publish_message(ReturnedFeedback(f"Returned feedback: {message.content}"), topic_id=DefaultTopicId())

    async def on_unhandled_message(self, message: Any, ctx: MessageContext) -> NoReturn:  # type: ignore
        print(f"Unhandled message: {message}")

# From core_grpc_worker_runtime/run_worker_pub_sub.py
class GreeterAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Greeter Agent")

    @message_handler
    async def on_ask(self, message: AskToGreet, ctx: MessageContext) -> None:
        await self.publish_message(Greeting(f"Hello, {message.content}!"), topic_id=DefaultTopicId())

    @message_handler
    async def on_returned_greet(self, message: ReturnedGreeting, ctx: MessageContext) -> None:
        await self.publish_message(Feedback(f"Feedback: {message.content}"), topic_id=DefaultTopicId())

    async def on_unhandled_message(self, message: Any, ctx: MessageContext) -> NoReturn:  # type: ignore
        print(f"Unhandled message: {message}")



# From core_grpc_worker_runtime/agents.py
class CascadingMessage:
    round: int

# From core_grpc_worker_runtime/agents.py
class ReceiveMessageEvent:
    round: int
    sender: str
    recipient: str

# From core_grpc_worker_runtime/agents.py
class CascadingAgent(RoutedAgent):
    def __init__(self, max_rounds: int) -> None:
        super().__init__("A cascading agent.")
        self.max_rounds = max_rounds

    @message_handler
    async def on_new_message(self, message: CascadingMessage, ctx: MessageContext) -> None:
        await self.publish_message(
            ReceiveMessageEvent(round=message.round, sender=str(ctx.sender), recipient=str(self.id)),
            topic_id=DefaultTopicId(),
        )
        if message.round == self.max_rounds:
            return
        await self.publish_message(CascadingMessage(round=message.round + 1), topic_id=DefaultTopicId())

# From core_grpc_worker_runtime/agents.py
class ObserverAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("An observer agent.")

    @message_handler
    async def on_receive_message(self, message: ReceiveMessageEvent, ctx: MessageContext) -> None:
        print(f"[Round {message.round}]: Message from {message.sender} to {message.recipient}.")



import openai
import pinecone
import nltk
from langchain.text_splitter import NLTKTextSplitter

# From Teenage-AGI/agent.py
def generate(prompt):
    completion = openai.ChatCompletion.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."},
        {"role": "system", "content": "Keep your thoughts relatively simple and concise"},
        {"role": "user", "content": prompt},
        ]
    )

    return completion.choices[0].message["content"]

# From Teenage-AGI/agent.py
def get_ada_embedding(text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]

# From Teenage-AGI/agent.py
def read_txtFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# From Teenage-AGI/agent.py
def createIndex(self, table_name=None):
        # Create Pinecone index
        if(table_name):
            self.table_name = table_name

        if(self.table_name == None):
            return

        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)

# From Teenage-AGI/agent.py
def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        if thought_type==INFORMATION:
            new_thought = "This is information fed to you by the user:\n" + new_thought
        elif thought_type==QUERIES:
            new_thought = "The user has said to you before:\n" + new_thought
        elif thought_type==THOUGHTS:
            # Not needed since already in prompts.yaml
            # new_thought = "You have previously thought:\n" + new_thought
            pass
        elif thought_type==ACTIONS:
            # Not needed since already in prompts.yaml as external thought memory
            pass

        vector = get_ada_embedding(new_thought)
        upsert_response = self.memory.upsert(
        vectors=[
            {
            'id':f"thought-{self.thought_id_count}", 
            'values':vector, 
            'metadata':
                {"thought_string": new_thought
                }
            }],
	    namespace=thought_type,
        )

        self.thought_id_count += 1

# From Teenage-AGI/agent.py
def internalThought(self, query) -> str:
        query_embedding = get_ada_embedding(query)
        query_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=QUERIES)
        thought_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
        results = query_results.matches + thought_results.matches
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        top_matches = "\n\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])
        #print(top_matches)
        
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{last_message}", self.last_message)
        print("------------INTERNAL THOUGHT PROMPT------------")
        print(internalThoughtPrompt)
        internal_thought = generate(internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here
        
        # Debugging purposes
        #print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        self.updateMemory(internalMemoryPrompt, THOUGHTS)
        return internal_thought, top_matches

# From Teenage-AGI/agent.py
def action(self, query) -> str:
        internal_thought, top_matches = self.internalThought(query)
        
        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        print("------------EXTERNAL THOUGHT PROMPT------------")
        print(externalThoughtPrompt)
        external_thought = generate(externalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace("{external_thought}", external_thought)
        self.updateMemory(externalMemoryPrompt, THOUGHTS)
        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), QUERIES)
        self.last_message = query
        return external_thought

# From Teenage-AGI/agent.py
def think(self, text) -> str:
        self.updateMemory(text, THOUGHTS)

# From Teenage-AGI/agent.py
def read(self, text) -> str:
        texts = text_splitter.split_text(text)
        vectors = []
        for t in texts:
            t = "This is information fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id':f"thought-{self.thought_id_count}", 
                'values':vector, 
                'metadata':
                    {"thought_string": t, 
                     }
                })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
        vectors,
	    namespace=INFORMATION,
        )

# From Teenage-AGI/agent.py
def readDoc(self, text) -> str:
        texts = text_splitter.split_text(read_txtFile(text))
        vectors = []
        for t in texts:
            t = "This is a document fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id':f"thought-{self.thought_id_count}", 
                'values':vector, 
                'metadata':
                    {"thought_string": t, 
                     }
                })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
        vectors,
	    namespace=INFORMATION,
        )

import sqlite3
import importlib.util
import string
import secrets
from langgraph.checkpoint.sqlite import SqliteSaver

# From AgentK/utils.py
def all_tool_functions():
    tools = list_tools()
    tool_funcs = []
    
    for tool in tools:
        try:
            module = load_module(f"tools/{tool}.py")
            tool_func = getattr(module, tool)
            tool_funcs.append(tool_func)
        except Exception as e:
            print(f"WARN: Could not load tool \"{tool}\". {e.__class__.__name__}: {e}")
    
    return tool_funcs

# From AgentK/utils.py
def list_broken_tools():
    tools = list_tools()
    broken_tools = {}
    
    for tool in tools:
        try:
            module = load_module(f"tools/{tool}.py")
            getattr(module, tool)
            del sys.modules[module.__name__]
        except Exception as e:
            exception_trace = traceback.format_exc()
            broken_tools[tool] = [e, exception_trace]
    
    return broken_tools

# From AgentK/utils.py
def list_tools():
    """
    list all tools available in the tools directory

    :return: list of tools
    """
    import os
    tools = []
    for file in os.listdir("tools"):
        if file.endswith(".py"):
            tools.append(file[:-3])

    return tools

# From AgentK/utils.py
def all_agents(exclude=["hermes"]):
    agents = list_agents()
    agents = [agent for agent in agents if agent not in exclude]
    agent_funcs = {}
    
    for agent in agents:
        try:
            module = load_module(f"agents/{agent}.py")
            agent_func = getattr(module, agent)
            agent_funcs[agent] = agent_func.__doc__
            del sys.modules[module.__name__]
        except Exception as e:
            print(f"WARN: Could not load agent \"{agent}\". {e.__class__.__name__}: {e}")
    
    return agent_funcs

# From AgentK/utils.py
def list_broken_agents():
    agents = list_agents()
    broken_agents = {}
    
    for agent in agents:
        try:
            module = load_module(f"agents/{agent}.py")
            getattr(module, agent)
            del sys.modules[module.__name__]
        except Exception as e:
            exception_trace = traceback.format_exc()
            broken_agents[agent] = [e, exception_trace]
    
    return broken_agents

# From AgentK/utils.py
def list_agents():
    """
    list all agents available in the agents directory

    :return: list of agents
    """
    import os
    agents = []
    for file in os.listdir("agents"):
        if file.endswith(".py") and file != "__init__.py":
            agents.append(file[:-3])

    return agents

# From AgentK/utils.py
def gensym(length=32, prefix="gensym_"):
    """
    generates a fairly unique symbol, used to make a module name,
    used as a helper function for load_module

    :return: generated symbol
    """
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    symbol = "".join([secrets.choice(alphabet) for i in range(length)])

    return prefix + symbol

# From AgentK/utils.py
def load_module(source, module_name=None):
    """
    reads file source and loads it as a module

    :param source: file to load
    :param module_name: name of module to register in sys.modules
    :return: loaded module
    """

    if module_name is None:
        module_name = gensym()

    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

from agents import hermes

from langchain_core.tools import tool
import utils

# From tools/list_available_agents.py
def list_available_agents():
    """List the name of available agents along with the type of task it's designed to be assigned."""
    return utils.all_agents()


# From tools/assign_agent_to_task.py
def assign_agent_to_task(agent_name: str, task: str):
    """Assign an agent to a task. This function returns the response from the agent."""
    print(f"Assigning agent {agent_name} to task: {task}")
    # Handle the case where the call to the agent fails (might be a job for the toolmaker)
    try:
        agent_module = utils.load_module(f"agents/{agent_name}.py")
        agent_function = getattr(agent_module, agent_name)
        result = agent_function(task=task)
        del sys.modules[agent_module.__name__]
        response = result["messages"][-1].content
        print(f"{agent_name} responded:")
        print(response)
        return response
    except Exception as e:
        exception_trace = traceback.format_exc()
        error = f"An error occurred while assigning {agent_name} to task {task}:\n {e}\n{exception_trace}"
        print(error)
        return error

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
import config

# From agents/agent_smith.py
def reasoning(state: MessagesState):
    print()
    print("agent_smith is thinking...")
    messages = state['messages']
    tooled_up_model = config.default_langchain_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}

# From agents/agent_smith.py
def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        if not last_message.content.strip() == "":
            print("agent_smith thought this:")
            print(last_message.content)
        print()
        print("agent_smith is acting by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"
    
    return END

# From agents/agent_smith.py
def agent_smith(task: str) -> str:
    """Designs and implements new agents, each designed to play a unique role."""
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )

from hfAgent import agents
from json import JSONDecodeError
from pathlib import Path
import huggingface_hub
from transformers.tools import HfAgent

from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain.schema import BaseMessage
from langchain.llms.base import LLM
from FreeLLM import HuggingChatAPI
from FreeLLM import ChatGPTAPI
from FreeLLM import BingChatAPI
import streamlit
from streamlit_chat_media import message

# From Free-Auto-GPT/Camel.py
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: None,
    ) -> None:
        self.system_message = system_message.content
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)
        output_message = self.model(str(input_message.content))
        self.update_messages(output_message)
        print(f"AI Assistant:\n\n{output_message}\n\n")
        return output_message

# From Free-Auto-GPT/Camel.py
def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

# From Free-Auto-GPT/Camel.py
def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

# From Free-Auto-GPT/Camel.py
def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

# From Free-Auto-GPT/Camel.py
def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)
        output_message = self.model(str(input_message.content))
        self.update_messages(output_message)
        print(f"AI Assistant:\n\n{output_message}\n\n")
        return output_message

# From Free-Auto-GPT/Camel.py
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
        assistant_sys_template = SystemMessagePromptTemplate.from_template(
            template=assistant_inception_prompt
        )
        assistant_sys_msg = assistant_sys_template.format_messages(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]

        user_sys_template = SystemMessagePromptTemplate.from_template(
            template=user_inception_prompt
        )
        user_sys_msg = user_sys_template.format_messages(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]

        return assistant_sys_msg, user_sys_msg

import requests
from huggingface_hub import HfFolder
from huggingface_hub import hf_hub_download
from huggingface_hub import list_spaces
from transformers.utils import logging
from transformers.tools.base import TASK_MAPPING
from transformers.tools.base import TOOL_CONFIG_FILE
from transformers.tools.base import Tool
from transformers.tools.base import load_tool
from transformers.tools.base import supports_remote
from transformers.tools.prompts import CHAT_MESSAGE_PROMPT
from transformers.tools.prompts import CHAT_PROMPT_TEMPLATE
from transformers.tools.prompts import RUN_PROMPT_TEMPLATE
from transformers.tools.python_interpreter import evaluate
from FreeLLM import BardChatAPI

# From hfAgent/agents.py
class PreTool:
    task: str
    description: str
    repo_id: str

# From hfAgent/agents.py
class OpenAiAgent(Agent):
    """
    Agent that uses the openai API to generate code.

    <Tip warning={true}>

    The openAI models are used in generation mode, so even for the `chat()` API, it's better to use models like
    `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

    </Tip>

    Args:
        model (`str`, *optional*, defaults to `"text-davinci-003"`):
            The name of the OpenAI model to use.
        api_key (`str`, *optional*):
            The API key to use. If unset, will look for the environment variable `"OPENAI_API_KEY"`.
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import OpenAiAgent

    agent = OpenAiAgent(model="text-davinci-003", api_key=xxx)
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(
        self,
        model="text-davinci-003",
        api_key=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):
        #if not is_openai_available():
            #raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "You need an openai key to use `OpenAIAgent`. You can get one here: Get one here "
                "https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = "
                "xxx."
            )
        else:
            openai.api_key = api_key
        self.model = model
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        if "gpt" in self.model:
            return [self._chat_generate(prompt, stop) for prompt in prompts]
        else:
            return self._completion_generate(prompts, stop)

    def generate_one(self, prompt, stop):
        if "gpt" in self.model:
            return self._chat_generate(prompt, stop)
        else:
            return self._completion_generate([prompt], stop)[0]

    def _chat_generate(self, prompt, stop):
        result = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=stop,
        )
        return result["choices"][0]["message"]["content"]

    def _completion_generate(self, prompts, stop):
        result = openai.Completion.create(
            model=self.model,
            prompt=prompts,
            temperature=0,
            stop=stop,
            max_tokens=200,
        )
        return [answer["text"] for answer in result["choices"]]

# From hfAgent/agents.py
class HfAgent(Agent):
    """
    Agent that uses and inference endpoint to generate code.

    Args:
        url_endpoint (`str`):
            The name of the url endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import HfAgent

    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(
        self, url_endpoint, token=None, chat_prompt_template=None, run_prompt_template=None, additional_tools=None
    ):
        self.url_endpoint = url_endpoint
        if token is None:
            self.token = f"Bearer {HfFolder().get_token()}"
        elif token.startswith("Bearer") or token.startswith("Basic"):
            self.token = token
        else:
            self.token = f"Bearer {token}"
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):
        headers = {"Authorization": self.token}
        inputs = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "return_full_text": False, "stop": stop},
        }

        response = requests.post(self.url_endpoint, json=inputs, headers=headers)
        if response.status_code == 429:
            print("Getting rate-limited, waiting a tiny bit before trying again.")
            time.sleep(1)
            return self._generate_one(prompt)
        elif response.status_code != 200:
            raise ValueError(f"Error {response.status_code}: {response.json()}")

        result = response.json()[0]["generated_text"]
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return result

# From hfAgent/agents.py
class ChatGPTAgent(Agent):
    """
    Agent that uses and inference endpoint of CHATGPT by IntelligenzaArtificialeItalia.net

    Args:
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from hfAgent import ChatGPTAgent

    agent = ChatGPTAgent("TOKEN")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """
    
    def __init__(
        self, token, chat_prompt_template=None, run_prompt_template=None, additional_tools=None, llm=None, model=None
    ):
        
        if token is None:
            ValueError("You must provide a ChatGPT token")
        else:
            from .FreeLLM import ChatGPTAPI
            import asyncio
            self.token = token
            if model is not None:
                self.llm = ChatGPTAPI.ChatGPT(token = self.token, model = "gpt-4")
            else:
                self.llm = ChatGPTAPI.ChatGPT(token = self.token)
            
            
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):

        result = self.llm(prompt + "\nRemember to use the following stop sequence: " + str(stop))  
         
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return result

# From hfAgent/agents.py
class HuggingChatAgent(Agent):
    """
    Agent that uses and inference endpoint of HuggingCHAT by IntelligenzaArtificialeItalia.net

    Args:
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from hfAgent import HuggingChatAgent

    agent = HuggingChatAgent()
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    agent.chat("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """
    
    def __init__(
        self, chat_prompt_template=None, run_prompt_template=None, additional_tools=None, llm=None, model=None
    ):
        
        import json
        from pathlib import Path
        from json import JSONDecodeError

        emailHF = os.getenv("emailHF", "your-emailHF")
        pswHF = os.getenv("pswHF", "your-pswHF")
        if emailHF != "your-emailHF" or pswHF != "your-pswHF":
            os.environ["emailHF"] = emailHF
            os.environ["pswHF"] = pswHF
        else:
            raise ValueError(
                "HuggingChat Token EMPTY. Edit the .env file and put your HuggingChat credentials"
            )

        from .FreeLLM import HuggingChatAPI
        self.llm = HuggingChatAPI.HuggingChat(email=emailHF, psw=pswHF)

            
        
            
            
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):

        result = self.llm(prompt + "\nRemember to use the following stop sequence: " + str(stop))  
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return result

# From hfAgent/agents.py
class BingChatAgent(Agent):
    """
    Agent that uses and inference endpoint of BingCHAT by IntelligenzaArtificialeItalia.net

    Args:
        cookiepath (`str`):
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from hfAgent import BingChatAgent

    agent = BingChatAgent("cookie-path")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """
    
    def __init__(
        self, cookiepath, chat_prompt_template=None, run_prompt_template=None, additional_tools=None, llm=None, model=None , conversation = "balanced"
    ):
        

        from .FreeLLM import BingChatAPI
        
        if cookiepath is None:
            ValueError("You must provide a cookie path")
        else:
            self.cookiepath = cookiepath
            if conversation == "balanced":
                self.llm = BingChatAPI.BingChat(cookiepath = self.cookiepath, conversation_style = "balanced")
            elif conversation == "creative":
                self.llm = BingChatAPI.BingChat(cookiepath = self.cookiepath, conversation_style = "creative")
            elif conversation == "precise":
                self.llm = BingChatAPI.BingChat(cookiepath = self.cookiepath, conversation_style = "precise")
            
            
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):

        result = self.llm(prompt + "\nRemember to use the following stop sequence: " + str(stop))  
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return result

# From hfAgent/agents.py
class BardChatAgent(Agent):
    """
    Agent that uses and inference endpoint of Bard Chat by IntelligenzaArtificialeItalia.net

    Args:
        token (`str`):
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from hfAgent import BardChatAgent

    agent = BardChatAgent("token")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """
    
    def __init__(
        self, token ,chat_prompt_template=None, run_prompt_template=None, additional_tools=None, llm=None, model=None , conversation = "balanced"
    ):
        

        from .FreeLLM import BardChatAPI
        
        if token is None:
            ValueError("You must provide a cookie path")
        else:
            self.token = token
            self.llm = BardChatAPI.BardChat(cookie = self.token)
            
            
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):

        result = self.llm(prompt + "\nRemember to use the following stop sequence: " + str(stop))  
         
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return result

# From hfAgent/agents.py
def get_remote_tools(organization="huggingface-tools"):
    spaces = list_spaces(author=organization)
    tools = {}
    for space_info in spaces:
        repo_id = space_info.id
        resolved_config_file = hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space")
        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)

        task = repo_id.split("/")[-1]
        tools[config["name"]] = PreTool(task=task, description=config["description"], repo_id=repo_id)

    return tools

# From hfAgent/agents.py
def resolve_tools(code, toolbox, remote=False, cached_tools=None):
    if cached_tools is None:
        resolved_tools = BASE_PYTHON_TOOLS.copy()
    else:
        resolved_tools = cached_tools
    for name, tool in toolbox.items():
        if name not in code or name in resolved_tools:
            continue

        if isinstance(tool, Tool):
            resolved_tools[name] = tool
        else:
            task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
            _remote = remote and supports_remote(task_or_repo_id)
            resolved_tools[name] = load_tool(task_or_repo_id, remote=_remote)

    return resolved_tools

# From hfAgent/agents.py
def get_tool_creation_code(code, toolbox, remote=False):
    code_lines = ["from transformers import load_tool", ""]
    for name, tool in toolbox.items():
        if name not in code or isinstance(tool, Tool):
            continue

        task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
        line = f'{name} = load_tool("{task_or_repo_id}"'
        if remote:
            line += ", remote=True"
        line += ")"
        code_lines.append(line)

    return "\n".join(code_lines) + "\n"

# From hfAgent/agents.py
def clean_code_for_chat(result):
    lines = result.split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
        idx += 1
    explanation = "\n".join(lines[:idx]).strip()
    if idx == len(lines):
        return explanation, None

    idx += 1
    start_idx = idx
    while not lines[idx].lstrip().startswith("```"):
        idx += 1
    code = "\n".join(lines[start_idx:idx]).strip()
    
     #if code start with "py`"  or "python`"
    if code.startswith("py`"):
        code = code[3:]
    elif code.startswith("python`"):
        code = code[7:]
    elif code.startswith("`"):
        code = code[1:]
        
    if code.endswith("`"):
        code = code[:-1]
        
    return explanation, code

# From hfAgent/agents.py
def clean_code_for_run(result):
    result = f"I will use the following {result}"
    try:
        explanation, code = result.split("Answer:")
    except ValueError:
        explanation = result
        code = "#Problem with the code"
        
    explanation = explanation.strip()
    code = code.strip()

    code_lines = code.split("\n")
    if code_lines[0] in ["```", "```py", "```python", "py`", "`"]:
        code_lines = code_lines[1:]
    if code_lines[-1] == "```":
        code_lines = code_lines[:-1]
    code = "\n".join(code_lines)
    
    #if code start with "py`"  or "python`"
    if code.startswith("py`"):
        code = code[3:]
    elif code.startswith("python`"):
        code = code[7:]
    elif code.startswith("`"):
        code = code[1:]
        
    if code.endswith("`"):
        code = code[:-1]
        

    return explanation, code

# From hfAgent/agents.py
def toolbox(self) -> Dict[str, Tool]:
        """Get all tool currently available to the agent"""
        return self._toolbox

# From hfAgent/agents.py
def format_prompt(self, task, chat_mode=False):
        description = "\n".join([f"- {name}: {tool.description}" for name, tool in self.toolbox.items()])
        if chat_mode:
            if self.chat_history is None:
                prompt = CHAT_PROMPT_TEMPLATE.replace("<<all_tools>>", description)
            else:
                prompt = self.chat_history
            prompt += CHAT_MESSAGE_PROMPT.replace("<<task>>", task)
        else:
            prompt = self.run_prompt_template.replace("<<all_tools>>", description)
            prompt = prompt.replace("<<prompt>>", task)
        return prompt

# From hfAgent/agents.py
def chat(self, task, *, return_code=False, remote=False, **kwargs):
        """
        Sends a new request to the agent in a chat. Will use the previous ones in its history.

        Args:
            task (`str`): The task to perform
            return_code (`bool`, *optional*, defaults to `False`):
                Whether to just return code and not evaluate it.
            remote (`bool`, *optional*, defaults to `False`):
                Whether or not to use remote tools (inference endpoints) instead of local ones.
            kwargs:
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers import HfAgent

        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        agent.chat("Draw me a picture of rivers and lakes")

        agent.chat("Transform the picture so that there is a rock in there")
        ```
        """
        prompt = self.format_prompt(task, chat_mode=True)
        result = self.generate_one(prompt, stop=["Human:", "====="])
        self.chat_history = prompt + result.strip() + "\n"
        explanation, code = clean_code_for_chat(result)

        print(f"==Explanation from the agent==\n{explanation}")

        if code is not None:
            print(f"\n\n==Code generated by the agent==\n{code}")
            if not return_code:
                print("\n\n==Result==")
                self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
                self.chat_state.update(kwargs)
                return evaluate(code, self.cached_tools, self.chat_state, chat_mode=True)
            else:
                tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
                return f"{tool_code}\n{code}"

# From hfAgent/agents.py
def prepare_for_new_chat(self):
        """
        Clears the history of prior calls to [`~Agent.chat`].
        """
        self.chat_history = None
        self.chat_state = {}
        self.cached_tools = None

# From hfAgent/agents.py
def run(self, task, *, return_code=False, remote=False, **kwargs):
        """
        Sends a request to the agent.

        Args:
            task (`str`): The task to perform
            return_code (`bool`, *optional*, defaults to `False`):
                Whether to just return code and not evaluate it.
            remote (`bool`, *optional*, defaults to `False`):
                Whether or not to use remote tools (inference endpoints) instead of local ones.
            kwargs:
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers import HfAgent

        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        agent.run("Draw me a picture of rivers and lakes")
        ```
        """
        prompt = self.format_prompt(task)
        result = self.generate_one(prompt, stop=["Task:"])
        explanation, code = clean_code_for_run(result)

        print(f"==Explanation from the agent==\n{explanation}")

        print(f"\n\n==Code generated by the agent==\n{code}")
        if not return_code:
            print("\n\n==Result==")
            self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
            return evaluate(code, self.cached_tools, state=kwargs.copy())
        else:
            tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
            return f"{tool_code}\n{code}"

# From hfAgent/agents.py
def generate_one(self, prompt, stop):
        # This is the method to implement in your custom agent.
        raise NotImplementedError

# From hfAgent/agents.py
def generate_many(self, prompts, stop):
        # Override if you have a way to do batch generation faster than one by one
        return [self.generate_one(prompt, stop) for prompt in prompts]

from langchain.agents import create_csv_agent
from langchain.utilities import PythonREPL

from langchain.agents import initialize_agent
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.human.tool import HumanInputRun
from langchain.agents import Tool

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

import pytest
from agentforge.testing.bootstrap import bootstrap_test_env
from agentforge.config import Config
from tests.utils.fakes import FakeChromaStorage
from agentforge.agent import Agent
from agentforge.cog import Cog

# From tests/conftest.py
def isolated_config(tmp_path) -> Config:
    """Return a fresh Config instance bound to an isolated temporary .agentforge."""
    # Copy default setup files into the tmp directory
    setup_src = SRC_PATH / "agentforge" / "setup_files"
    shutil.copytree(setup_src, tmp_path / ".agentforge")

    # Store the original YAML content for ExampleCog
    example_cog_path = tmp_path / ".agentforge" / "cogs" / "example_cog.yaml"
    original_yaml = example_cog_path.read_text()

    # Create the Config instance
    cfg = Config.reset(root_path=str(tmp_path))
    
    # Provide the config to the test
    yield cfg

    # After the test completes, restore the original YAML
    # This ensures that changes made by one test don't affect others
    example_cog_path.write_text(original_yaml)
    
    # Ensure singleton cleaned up for next test
    Config._instance = None

# From tests/conftest.py
def clean_yaml_after_test():
    """Fixture to ensure that ExampleCog.yaml is restored to its original state after each test."""
    # Get the path to the repo's .agentforge directory
    agentforge_dir = REPO_ROOT / ".agentforge"
    example_cog_path = agentforge_dir / "cogs" / "example_cog.yaml"
    
    # Save the original content
    original_content = None
    if example_cog_path.exists():
        original_content = example_cog_path.read_text()
    
    # Let the test run
    yield
    
    # After the test, restore the original content
    if original_content is not None:
        example_cog_path.write_text(original_content)

# From tests/conftest.py
def auto_clean_yaml(request):
    """Automatically apply clean_yaml_after_test to all tests in cog_tests."""
    if "cog_tests" in request.module.__name__:
        request.getfixturevalue("clean_yaml_after_test")

# From tests/conftest.py
def fake_chroma():
    """Return the FakeChromaStorage class and clear registry after test."""
    yield FakeChromaStorage
    FakeChromaStorage.clear_registry()

# From tests/conftest.py
def stubbed_agents(monkeypatch):
    """Monkey-patch Agent.run for ExampleCog agents so no model work happens."""

    from agentforge.agent import Agent

    original_run = Agent.run
    decision_values = ["approve", "reject", "other"]

    def fake_run(self: Agent, **context):  # type: ignore[override]
        # Check if this agent has debug mode enabled and should use original behavior
        if hasattr(self, 'settings') and self.settings.get('system', {}).get('debug', {}).get('mode', False):
            # Let debug mode handle this agent normally
            return original_run(self, **context)
        # Increment branch_call_counts if self._cog is present
        if hasattr(self, '_cog') and hasattr(self._cog, 'branch_call_counts'):
            self._cog.branch_call_counts[self.agent_name] = self._cog.branch_call_counts.get(self.agent_name, 0) + 1
        name_l = self.agent_name.lower()
        
        # Handle specific agent types with their expected outputs
        if "analyze" in name_l:
            return {"analysis": "stub-analysis"}
        elif "decide" in name_l:
            idx = getattr(self, "_call_idx", 0)
            self._call_idx = idx + 1
            val = decision_values[idx % len(decision_values)]
            return {"choice": val, "rationale": "stub"}
        elif "response" in name_l or "respond" in name_l:
            return "FINAL RESPONSE"
        elif "understand" in name_l:
            return {
                "insights": "User is asking about programming topics",
                "user_intent": "Seeking information or help",
                "relevant_topics": ["programming", "learning"],
                "persona_relevant": "User shows interest in technical topics"
            }
        else:
            # For any other agent, check if it's a known test agent that should use original behavior
            if "test" in name_l:
                return original_run(self, **context)
            # Otherwise provide a generic response for cog agents
            return f"Simulated response from {self.agent_name}"

    monkeypatch.setattr(Agent, "run", fake_run, raising=True)

    # Note: _get_response_format_for_agent method was removed during Cog refactor
    # Response format handling is now done by individual agents
    yield

# From tests/conftest.py
def example_cog(isolated_config):
    from agentforge.cog import Cog
    return Cog("example_cog")

# From tests/conftest.py
def cleanup_root_dot_agentforge():
    """Clean up any .agentforge directory at the repo root after all tests have run."""
    # Let the tests run
    yield
    
    # After all tests, clean up any .agentforge directory at the repo root
    root_dot_agentforge = REPO_ROOT / ".agentforge"
    if root_dot_agentforge.exists() and root_dot_agentforge.is_dir():
        try:
            shutil.rmtree(root_dot_agentforge)
            print(f"Cleaned up {root_dot_agentforge}")
        except Exception as e:
            print(f"Failed to clean up {root_dot_agentforge}: {e}")

# From tests/conftest.py
def fake_run(self: Agent, **context):  # type: ignore[override]
        # Check if this agent has debug mode enabled and should use original behavior
        if hasattr(self, 'settings') and self.settings.get('system', {}).get('debug', {}).get('mode', False):
            # Let debug mode handle this agent normally
            return original_run(self, **context)
        # Increment branch_call_counts if self._cog is present
        if hasattr(self, '_cog') and hasattr(self._cog, 'branch_call_counts'):
            self._cog.branch_call_counts[self.agent_name] = self._cog.branch_call_counts.get(self.agent_name, 0) + 1
        name_l = self.agent_name.lower()
        
        # Handle specific agent types with their expected outputs
        if "analyze" in name_l:
            return {"analysis": "stub-analysis"}
        elif "decide" in name_l:
            idx = getattr(self, "_call_idx", 0)
            self._call_idx = idx + 1
            val = decision_values[idx % len(decision_values)]
            return {"choice": val, "rationale": "stub"}
        elif "response" in name_l or "respond" in name_l:
            return "FINAL RESPONSE"
        elif "understand" in name_l:
            return {
                "insights": "User is asking about programming topics",
                "user_intent": "Seeking information or help",
                "relevant_topics": ["programming", "learning"],
                "persona_relevant": "User shows interest in technical topics"
            }
        else:
            # For any other agent, check if it's a known test agent that should use original behavior
            if "test" in name_l:
                return original_run(self, **context)
            # Otherwise provide a generic response for cog agents
            return f"Simulated response from {self.agent_name}"

from config import Config
from agentforge.apis.base_api import BaseModel
from agentforge.utils.logger import Logger
from agentforge.utils.prompt_processor import PromptProcessor
from agentforge.utils.parsing_processor import ParsingProcessor

# From agentforge/agent.py
def load_persona_data(self) -> None:
        """Load persona data if personas are enabled in system settings."""
        if not self.agent_config.settings.system.persona.enabled or not self.agent_config.persona:
            return
        
        self.persona = self.agent_config.persona.copy()
        self.template_data['persona'] = self.persona
        self.logger.debug(f"Persona Data Loaded for '{self.agent_name}'.")

# From agentforge/agent.py
def load_data(self, **kwargs: Any) -> None:
        """
        Load all data needed for prompt generation, including dynamic updates.
        
        Args:
            **kwargs: Additional data to incorporate into template variables.
        """
        if self.agent_config.settings.system.misc.on_the_fly:
            self._initialize_agent_config()

        self.load_additional_data()
        self.template_data.update(kwargs)

# From agentforge/agent.py
def load_additional_data(self) -> None:
        """
        Load custom additional data for the agent.
        Override this method in subclasses to load custom data.
        """
        pass

# From agentforge/agent.py
def process_data(self) -> None:
        """
        Process loaded data before generating prompts.
        Override this method in subclasses to implement custom data processing.
        """
        pass

# From agentforge/agent.py
def render_prompt(self) -> None:
        """Render prompt templates with the current template data."""
        self.prompt = self.prompt_processor.render_prompts(self.prompt_template, self.template_data)

# From agentforge/agent.py
def run_model(self) -> None:
        """Execute the model with the rendered prompt and configured parameters."""
        if self.agent_config.settings.system.debug.mode:
            self.result = self.agent_config.simulated_response
            return

        self._execute_model_generation()

# From agentforge/agent.py
def parse_result(self) -> None:
        """
        Parse the model output using the agent's parse_response_as, if specified.
        The parsed result is stored in self.parsed_result.
        """
        self.parsed_result = self.parsing_processor.parse_by_format(self.result, self.agent_config.parse_response_as)

# From agentforge/agent.py
def post_process_result(self) -> None:
        """
        Extension point for additional processing after parsing the model's response.
        """
        pass

# From agentforge/agent.py
def build_output(self) -> None:
        """
        Build the final output for the agent. By default, the output is set to the parsed result.
        Override this method in subclasses to implement custom output building.
        """
        self.output = self.parsed_result

from agentforge.config_structs.trail_structs import ThoughtTrailEntry
from agentforge.core.agent_registry import AgentRegistry
from agentforge.core.agent_runner import AgentRunner
from agentforge.core.memory_manager import MemoryManager
from agentforge.core.transition_resolver import TransitionResolver
from agentforge.core.trail_recorder import TrailRecorder

# From agentforge/cog.py
class Cog:
    """
    Orchestrates a workflow of agents based on flow configuration.
    
    Cog reads agent definitions, flow transitions, and memory configurations
    from a YAML file and executes agents in the specified order, handling
    decision-based routing and memory management.
    
    The workflow follows a template method pattern with clear separation of
    concerns across semantic sections for improved maintainability and testing.
    """

    def __init__(self, cog_file: str, enable_trail_logging: Optional[bool] = None, log_file: Optional[str] = 'cog'):
        """
        Initialize a Cog instance with necessary configurations and services.
        
        Args:
            cog_file: Name of the cog configuration file
            enable_trail_logging: Override for trail logging setting. Uses config default if None.
            log_file: Name of the log file. Defaults to 'cog'.
        """
        self.cog_file = cog_file
        self.config = Config()
        self.logger = Logger(name='Cog', default_logger=log_file)
        
        # Initialize core configurations
        self._initialize_cog_config()
        self._initialize_core_services()
        self._initialize_trail_logging(enable_trail_logging)
        
        # Initialize execution state
        self.last_executed_agent: Optional[str] = None
        self.branch_call_counts: dict = {}
        self._reset_execution_state()

    # ---------------------------------
    # Configuration & Initialization
    # ---------------------------------

    def _initialize_cog_config(self) -> None:
        """Load and build the cog configuration from the config file."""
        self.cog_config = self.config.load_cog_data(self.cog_file)

    def _initialize_core_services(self) -> None:
        """Initialize all core service components."""
        # Create agents using AgentRegistry
        self.agents = AgentRegistry.build_agents(self.cog_config)
        
        # Initialize core components
        self.mem_mgr = MemoryManager(self.cog_config, self.cog_file)
        self.agent_runner = AgentRunner()
        self.transition_resolver = TransitionResolver(self.cog_config.cog.flow)

    def _initialize_trail_logging(self, enable_trail_logging: Optional[bool]) -> None:
        """Initialize trail logging with appropriate configuration."""
        if enable_trail_logging is None:
            enable_trail_logging = self.cog_config.cog.trail_logging
        self.trail_recorder = TrailRecorder(enabled=enable_trail_logging)

    # ---------------------------------
    # Public Interface
    # ---------------------------------

    def run(self, **kwargs: Any) -> Any:
        """
        Execute the cog by iteratively running agents as defined in the flow.
        
        Args:
            **kwargs: Initial context values to provide to the agents
            
        Returns:
            Any: Based on the 'end' keyword in the final transition:
                - If 'end: true', returns the output of the last agent executed
                - If 'end: <agent_id>', returns the output of that specific agent
                - If 'end: <agent_id>.field.subfield', returns that nested value
                - Otherwise, returns the full internal state
        """
        try:
            self.logger.info(f"Running cog '{self.cog_file}'...")
            # Load chat history with the initial user context so semantic search can use it
            self.mem_mgr.load_chat(_ctx=kwargs, _state={})
            self._execute_workflow(**kwargs)
            result = self._process_execution_result()
            self.logger.info(f"Cog '{self.cog_file}' completed successfully!")
            self.mem_mgr.record_chat(self.context, result)
            return result
        except Exception as e:
            self.logger.error(f"Cog execution failed: {e}")
            raise

    def get_track_flow_trail(self) -> List[ThoughtTrailEntry]:
        """
        Get the trail of agent executions for this cog run.
        
        Returns:
            List of ThoughtTrailEntry objects representing the execution trail
        """
        return self.trail_recorder.get_trail()

    # ---------------------------------
    # State Management
    # ---------------------------------

    def _reset_execution_state(self) -> None:
        """Reset all execution state for a fresh run."""
        self.context: dict = {}  # external context (runtime/user input)
        self.state: dict = {}    # internal state (agent-local/internal data)
        self.branch_call_counts: dict = {}
        self._reset_trail_logging()

    def _reset_trail_logging(self) -> None:
        """Reset trail logging for a new execution."""
        if hasattr(self, 'trail_recorder'):
            self.trail_recorder.reset_trail()

    def _prepare_execution_state(self, **kwargs: Any) -> None:
        """Prepare execution state with provided context."""
        self._reset_execution_state()
        self.context.update(kwargs)
        # Allow custom context loading
        self.load_additional_context(**kwargs)

    def _update_agent_state(self, agent_id: str, output: Any) -> None:
        """Update internal state with agent output."""
        self.state[agent_id] = output
        self.last_executed_agent = agent_id

    # ---------------------------------
    # Flow Execution
    # ---------------------------------

    def _execute_workflow(self, **kwargs: Any) -> None:
        """
        Execute the complete cog workflow from start to completion.
        Template method that orchestrates the entire execution flow.
        """
        self._prepare_execution_state(**kwargs)
        self._execute_agent_flow()

    def _execute_agent_flow(self) -> None:
        """
        Execute the main agent flow from start to completion.
        Orchestrates agent execution with transition resolution.
        """
        self.logger.log("Starting cog execution", "debug", "Flow")
        
        current_agent_id = self.cog_config.cog.flow.start
        self.last_executed_agent = None
        self.transition_resolver.reset_visit_counts()
        
        while current_agent_id:
                self.logger.log(f"Processing agent: {current_agent_id}", "debug", "Flow")
                
                # Execute single agent cycle
                current_agent_id = self._execute_single_agent_cycle(current_agent_id)
                
        self.logger.log("Cog execution completed", "debug", "Flow")

    def _execute_single_agent_cycle(self, agent_id: str) -> Optional[str]:
        """
        Execute a complete cycle for a single agent.
        
        Args:
            agent_id: The ID of the agent to execute
            
        Returns:
            The ID of the next agent to execute, or None if flow should end
        """
        # Handle pre-execution operations
        self._prepare_agent_execution(agent_id)
        
        # Execute the agent
        agent_output = self._execute_agent(agent_id)
        
        # Handle post-execution operations
        self._finalize_agent_execution(agent_id, agent_output)
        
        # Determine next agent in flow
        return self._determine_next_agent(agent_id)

    def _determine_next_agent(self, current_agent_id: str) -> Optional[str]:
        """
        Determine the next agent in the flow based on transition rules.
        
        Args:
            current_agent_id: The ID of the current agent
            
        Returns:
            The ID of the next agent, or None if flow should end
        """
        next_agent_id = self.transition_resolver.get_next_agent(current_agent_id, self.state)
        self.logger.log(f"Next agent: {next_agent_id}", "debug", "Flow")
        if next_agent_id != current_agent_id:
            self._reset_branch_counts()
        return next_agent_id

    # ---------------------------------
    # Agent Execution
    # ---------------------------------

    def _prepare_agent_execution(self, agent_id: str) -> None:
        """
        Prepare for agent execution by handling pre-execution operations.
        
        Args:
            agent_id: The ID of the agent to prepare for execution
        """
        # Call pre-execution hook
        self.pre_agent_execution(agent_id)
        
        # Handle memory operations before agent execution
        self._handle_pre_execution_memory(agent_id)

    def _execute_agent(self, agent_id: str) -> Any:
        """
        Execute a single agent and return its output.
        
        Args:
            agent_id: The ID of the agent to execute
            
        Returns:
            The output from the agent execution
        """
        agent = self.agents.get(agent_id)
        mem = self.mem_mgr.build_mem()
        output = self.agent_runner.run_agent(agent_id, agent, self.context, self.state, mem)
        self.branch_call_counts[agent_id] = self.branch_call_counts.get(agent_id, 0) + 1
        return output

    def _finalize_agent_execution(self, agent_id: str, output: Any) -> None:
        """
        Finalize agent execution by handling post-execution operations.
        
        Args:
            agent_id: The ID of the executed agent
            output: The output from the agent execution
        """
        # Process agent output
        processed_output = self.process_agent_output(agent_id, output)
        
        # Update agent state
        self._update_agent_state(agent_id, processed_output)
        
        # Track output for logging if enabled
        self._track_agent_output(agent_id, processed_output)
        
        # Handle memory operations after agent execution
        self._handle_post_execution_memory(agent_id)
        
        # Call post-execution hook
        self.post_agent_execution(agent_id, processed_output)

    def _track_agent_output(self, agent_id: str, output: Any) -> None:
        """Track agent output in thought flow trail if logging is enabled."""
        self.trail_recorder.record_agent_output(agent_id, output)

    def _handle_execution_error(self, agent_id: Optional[str], error: Exception) -> None:
        """
        Extension point for custom error handling.
        This method is not called automatically - subclasses can call it when needed.
        
        Args:
            agent_id: The ID of the agent that failed, if available
            error: The exception that occurred
        """
        self.logger.log(f"Error during cog execution: {error}", "error", "Flow")

    # ---------------------------------
    # Memory Management
    # ---------------------------------

    def _handle_pre_execution_memory(self, agent_id: str) -> None:
        """
        Handle memory operations before agent execution.
        
        Args:
            agent_id: The ID of the agent about to be executed
        """
        self.mem_mgr.query_before(agent_id, self.context, self.state)

    def _handle_post_execution_memory(self, agent_id: str) -> None:
        """
        Handle memory operations after agent execution.
        
        Args:
            agent_id: The ID of the agent that was executed
        """
        self.mem_mgr.update_after(agent_id, self.context, self.state)

    # ---------------------------------
    # Result Processing
    # ---------------------------------

    def _process_execution_result(self) -> Any:
        """
        Process and return the final execution result based on end conditions.
        
        Returns:
            The processed result based on end transition configuration
        """
        if not self.last_executed_agent:
            return self.state
            
        agent_transition = self.transition_resolver._get_agent_transition(self.last_executed_agent)
        
        # Handle the 'end' keyword - check for end transition
        if agent_transition.type == "end" or agent_transition.end:
            return self._extract_end_result(agent_transition)
            
        # Default behavior: return the full internal state
        return self.state

    def _extract_end_result(self, agent_transition) -> Any:
        """
        Extract the final result based on end transition configuration.
        
        Args:
            agent_transition: The transition configuration for the final agent
            
        Returns:
            The extracted result based on end configuration
        """
        # If `end: true`, return the last agent's output
        if agent_transition.end is True:
            return self.state.get(self.last_executed_agent)
        
        # If end is a string, it could be an agent_id or dot notation
        if isinstance(agent_transition.end, str):
            # Check if it's just an agent ID
            if agent_transition.end in self.state:
                return self.state[agent_transition.end]
            # Otherwise treat as dot notation in state
            return self._get_nested_result(agent_transition.end)
            
        return self.state.get(self.last_executed_agent)

    def _get_nested_result(self, path: str) -> Any:
        """
        Extract nested result using dot notation.
        
        Args:
            path: Dot notation path to extract from state
            
        Returns:
            The value at the specified path
        """
        return ParsingProcessor.get_dot_notated(self.state, path)

    # ---------------------------------
    # Extension Points
    # ---------------------------------

    def load_additional_context(self, **kwargs: Any) -> None:
        """
        Load custom additional context for the cog execution.
        Override this method in subclasses to load custom context data.
        
        Args:
            **kwargs: Context data to potentially augment
        """
        pass

    def process_agent_output(self, agent_id: str, output: Any) -> Any:
        """
        Process agent output before storing in state.
        Override this method in subclasses to implement custom output processing.
        
        Args:
            agent_id: The ID of the agent that produced the output
            output: The raw output from the agent
            
        Returns:
            The processed output to store in state
        """
        return output

    def pre_agent_execution(self, agent_id: str) -> None:
        """
        Hook called before each agent execution.
        Override this method in subclasses for custom pre-execution logic.
        
        Args:
            agent_id: The ID of the agent about to be executed
        """
        pass

    def post_agent_execution(self, agent_id: str, output: Any) -> None:
        """
        Hook called after each agent execution.
        Override this method in subclasses for custom post-execution logic.
        
        Args:
            agent_id: The ID of the agent that was executed
            output: The output from the agent execution
        """
        pass

    def _reset_branch_counts(self):
        self.branch_call_counts.clear()

# From agentforge/cog.py
def get_track_flow_trail(self) -> List[ThoughtTrailEntry]:
        """
        Get the trail of agent executions for this cog run.
        
        Returns:
            List of ThoughtTrailEntry objects representing the execution trail
        """
        return self.trail_recorder.get_trail()

# From agentforge/cog.py
def load_additional_context(self, **kwargs: Any) -> None:
        """
        Load custom additional context for the cog execution.
        Override this method in subclasses to load custom context data.
        
        Args:
            **kwargs: Context data to potentially augment
        """
        pass

# From agentforge/cog.py
def process_agent_output(self, agent_id: str, output: Any) -> Any:
        """
        Process agent output before storing in state.
        Override this method in subclasses to implement custom output processing.
        
        Args:
            agent_id: The ID of the agent that produced the output
            output: The raw output from the agent
            
        Returns:
            The processed output to store in state
        """
        return output

# From agentforge/cog.py
def pre_agent_execution(self, agent_id: str) -> None:
        """
        Hook called before each agent execution.
        Override this method in subclasses for custom pre-execution logic.
        
        Args:
            agent_id: The ID of the agent about to be executed
        """
        pass

# From agentforge/cog.py
def post_agent_execution(self, agent_id: str, output: Any) -> None:
        """
        Hook called after each agent execution.
        Override this method in subclasses for custom post-execution logic.
        
        Args:
            agent_id: The ID of the agent that was executed
            output: The output from the agent execution
        """
        pass

import filecmp

# From agentforge/init_agentforge.py
def user_decision_prompt(existing_file: str) -> str:
    """
    Interactively prompts the user about what to do with a file conflict.
    Returns one of the following single-character codes:
      - 'y' for overriding the file
      - 'n' for skipping this file
      - 'a' for overriding all existing files without asking again
      - 'z' for skipping all existing files without asking again
      - '' (empty) if the user input is invalid
    """
    print(f"\nFile '{existing_file}' already exists and is different from the source.")
    response = input(
        "Select an option:\n"
        "[Y] Override this file\n"
        "[N] Skip this file\n"
        "[A] Override all existing files without asking again\n"
        "[Z] Skip all existing files without asking again\n"
        "Enter your choice (Y/N/A/Z): "
    ).lower()
    valid_choices = {'y', 'n', 'a', 'z'}
    if response in valid_choices:
        return response
    print("Invalid option. Skipping this file by default.")
    return ''

# From agentforge/init_agentforge.py
def should_copy_file(
        src_file: str,
        dst_file: str,
        skip_all: bool,
        override_all: bool
) -> (bool, bool, bool):
    """
    Determines whether to copy a file from src_file to dst_file based on existing
    state flags and user decision. Returns a tuple of three booleans in the form:
      (copy_this_file, new_skip_all, new_override_all).
    """
    if skip_all:
        # We’re skipping all conflicts globally, no copy, just return the updated flags.
        return False, skip_all, override_all

    if not os.path.exists(dst_file):
        # If there's no existing file, proceed with the copy.
        return True, skip_all, override_all

    # If the files match, skip copying; there's no reason to replace it.
    if filecmp.cmp(src_file, dst_file, shallow=False):
        return False, skip_all, override_all

    # If we’re overriding all conflicts globally, skip user prompt and copy.
    if override_all:
        return True, skip_all, override_all

    # Otherwise, prompt the user for a decision.
    decision = user_decision_prompt(os.path.relpath(dst_file))
    if decision == 'a':
        # Override all from now on.
        return True, skip_all, True
    if decision == 'z':
        # Skip all from now on.
        return False, True, override_all
    if decision == 'n':
        # Skip just this file.
        return False, skip_all, override_all
    if decision == 'y':
        # Copy just this file.
        return True, skip_all, override_all

    # If user input is invalid or empty, skip the file by default.
    return False, skip_all, override_all

# From agentforge/init_agentforge.py
def copy_directory(
        root: Path,
        src: Path,
        override_all: bool = False,
        skip_all: bool = False
) -> None:
    """
    Recursively copies files from 'src' to 'root', skipping __pycache__ and __init__.py
    or .pyc files, while respecting user choices about overwriting.
    """
    for current_dir, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        dst_dir = current_dir.replace(str(src), str(root), 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            print(f"Created directory '{os.path.relpath(dst_dir, start=root)}'.")

        for file_ in files:
            if file_ == '__init__.py' or file_.endswith('.pyc'):
                continue

            src_file_str = str(os.path.join(current_dir, file_))
            dst_file_str = str(os.path.join(dst_dir, file_))

            relative_src_path = os.path.relpath(src_file_str, start=src)
            relative_dst_path = os.path.relpath(dst_file_str, start=root)

            do_copy, skip_all, override_all = should_copy_file(
                src_file_str,
                dst_file_str,
                skip_all,
                override_all
            )

            if not do_copy:
                print(f"Skipped '{relative_dst_path}'.")
                continue

            shutil.copy2(src_file_str, dst_file_str)
            print(f"Copied '{relative_src_path}' to '{relative_dst_path}'.")

# From agentforge/init_agentforge.py
def setup_agentforge() -> None:
    """
    Locates the AgentForge package, copies its 'setup_files' directory into
    the current working directory, and provides feedback on the process.
    """
    package_name = 'agentforge'
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"{package_name} is not installed.")
            return

        agentforge_path = spec.submodule_search_locations[0]
        print(f"Found {package_name} at {agentforge_path}")
        installer_path = os.path.join(agentforge_path, 'setup_files')
        project_root = Path.cwd() / ".agentforge"
        if not project_root.exists():
            project_root.mkdir()
            print(f"Created project template directory: {project_root}")
        copy_directory(project_root, Path(installer_path))
        print("AgentForge setup is complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

import importlib
import threading
import pathlib
from ruamel.yaml import YAML
from types import ModuleType
from core.config_manager import ConfigManager
from config_structs import AgentConfig
from config_structs import CogConfig

# From agentforge/config.py
class Config:
    """
    Singleton class for loading, managing, and providing access to AgentForge configuration data.
    """
    _instance = None
    _lock = threading.Lock()
    _debug = False
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, root_path: Optional[str] = None):
        if not hasattr(self, 'is_initialized'):
            self._initialize_attributes(root_path)
            self.is_initialized = True

    @classmethod
    def reset(cls, root_path=None):
        """
        Completely resets the Config singleton, allowing for re-initialization.
        """
        cls._instance = None
        instance = cls(root_path=root_path)
        return instance

    def _initialize_attributes(self, root_path: Optional[str]):
        """
        Initializes all instance attributes for the Config class.
        """
        self.project_root = self.find_project_root(root_path)
        self.config_path = self.project_root / ".agentforge"
        self.data = {}
        self.config_manager = ConfigManager()
        self.load_all_configurations()

    def find_project_root(self, root_path: Optional[str] = None) -> pathlib.Path:
        # If a root path was provided, use it to checking that .agentforge exists
        if root_path:
            custom_root = pathlib.Path(root_path).resolve()
            agentforge_dir = custom_root / ".agentforge"
            if agentforge_dir.is_dir():
                if self._debug: print(f"\n\nUsing custom project root: {custom_root}")
                return custom_root
            # Early return or raise an error if .agentforge isn't found in the custom path
            raise FileNotFoundError(f"No .agentforge found in custom root path: {custom_root}")

        # Otherwise, fall back to the original search logic
        script_dir = pathlib.Path(sys.argv[0]).resolve().parent
        current_dir = script_dir
        if self._debug: print(f"\n\nCurrent working directory: {os.getcwd()}")

        while current_dir != current_dir.parent:
            potential_dir = current_dir / ".agentforge"
            if self._debug: print(f"Checking {potential_dir}")
            if potential_dir.is_dir():
                if self._debug: print(f"Found .agentforge directory at: {current_dir}\n")
                return current_dir
            current_dir = current_dir.parent

        raise FileNotFoundError(f"Could not find the '.agentforge' directory starting from {script_dir}")

    # -----------------------------------
    # Configuration Loading & Saving
    # -----------------------------------

    @staticmethod
    def load_yaml_file(file_path: str) -> Dict[str, Any]:
        """
        Reads and parses a YAML file, returning its contents as a Python dictionary.
        Returns an empty dictionary if the file is not found or an error occurs.
        """
        try:
            with open(file_path, 'r') as yaml_file:
                return yaml.safe_load(yaml_file)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return {}
        except yaml.YAMLError:
            print(f"Error decoding YAML from {file_path}")
            return {}

    def load_all_configurations(self):
        """
        Recursively loads all configuration data from YAML files under each subdirectory of the .agentforge folder.
        """
        with self._lock:
            for subdir, dirs, files in os.walk(self.config_path):
                for file in files:
                    if file.endswith(('.yaml', '.yml')):
                        subdir_path = pathlib.Path(subdir)
                        relative_path = subdir_path.relative_to(self.config_path)
                        nested_dict = self.get_nested_dict(self.data, relative_path.parts)
                        file_path = str(subdir_path / file)
                        data = self.load_yaml_file(file_path)
                        if data:
                            filename_without_ext = os.path.splitext(file)[0]
                            nested_dict[filename_without_ext] = data

    def save(self):
        """
        Saves changes to the configuration back to the system.yaml file,
        preserving structure, formatting, and comments.
        """
        with Config._lock:
            system_yaml_path = self.config_path / 'settings' / 'system.yaml'
            _yaml = YAML()
            _yaml.preserve_quotes = True
            try:
                with open(system_yaml_path, 'r') as yaml_file:
                    existing_data = _yaml.load(yaml_file)
                if 'settings' in self.data and 'system' in self.data['settings']:
                    for key, value in self.data['settings']['system'].items():
                        if isinstance(value, dict) and key in existing_data:
                            existing_data[key].update(value)
                            continue
                        existing_data[key] = value
                    with open(system_yaml_path, 'w') as yaml_file:
                        _yaml.dump(existing_data, yaml_file)
                    return
                print("No system settings to save.")
            except Exception as e:
                print(f"Error saving configuration to {system_yaml_path}: {e}")

    def reload(self):
        """
        Reloads configurations if on-the-fly reloading is enabled in settings.
        """
        if self.settings.system.misc.on_the_fly:
            self.load_all_configurations()

    # -----------------------------------
    # Agent & Cog Configuration
    # -----------------------------------

    def load_agent_data(self, agent_name: str) -> AgentConfig:
        """
        Loads configuration data for a specified agent, applying any overrides in the agent's config.
        Returns a structured AgentConfig object containing everything needed to run that agent.
        """
        agent = self.find_config('prompts', agent_name)
        api_name, class_name, model_name, final_params = self.resolve_model_overrides(agent)
        model = self.get_model(api_name, class_name, model_name)
        persona_data = self.load_persona(agent)
        prompts = self.fix_prompt_placeholders(agent.get('prompts', {}))
        settings = self.data.get('settings', {})
        default_debug_text = settings['system']['debug'].get('simulated_response', 'Simulated Text Goes Here!!!')
        simulated_response = agent.get('simulated_response', default_debug_text).strip()
        raw_agent_data = {
            'name': agent_name,
            'settings': settings,
            'model': model,
            'params': final_params,
            'persona': persona_data,
            'prompts': prompts,
            'simulated_response': simulated_response,
        }
        reserved_fields = {
            'name', 'settings', 'model', 'params', 'persona', 'prompts', 'simulated_response',
            'model_overrides'
        }
        for key, value in agent.items():
            if key not in raw_agent_data and key not in reserved_fields:
                raw_agent_data[key] = value
        return self.config_manager.build_agent_config(raw_agent_data)

    def load_cog_data(self, cog_name: str) -> CogConfig:
        """
        Loads configuration data for a specified cog, returning a validated structured CogConfig object.
        """
        raw_cog_data = self.find_config('cogs', cog_name)
        return self.config_manager.build_cog_config(raw_cog_data)

    # -----------------------------------
    # Model API & Overrides
    # -----------------------------------

    def resolve_model_overrides(self, agent: dict) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Finds and merges all relevant model overrides into a final 4-tuple:
        (api_name, class_name, model_identifier, final_params).
        """
        api_name, model_name, agent_params_override = self._get_agent_api_and_model(agent)
        api_section = self._get_api_section(api_name)
        class_name, model_data = self._find_class_for_model(api_section, model_name)
        model_identifier = self._get_model_identifier(api_name, model_name, model_data)
        final_params = self._merge_params(api_section, class_name, model_data, agent_params_override)
        return api_name, class_name, model_identifier, final_params

    def _get_agent_api_and_model(self, agent: dict) -> Tuple[str, str, Dict[str, Any]]:
        """
        Reads the 'Selected Model' defaults from the YAML and merges any agent-level overrides.
        Returns (api_name, model_name, agent_params_override).
        Raises ValueError if no valid API/Model can be determined.
        """
        selected_model = self.data['settings']['models'].get('default_model', {})
        default_api = selected_model.get('api')
        default_model = selected_model.get('model')
        model_overrides = agent.get('model_overrides', {})
        api_name = model_overrides.get('api', default_api)
        model_name = model_overrides.get('model', default_model)
        agent_params_override = model_overrides.get('params', {})
        if not api_name or not model_name:
            raise ValueError("No valid API/Model found in either Selected Model defaults or agent overrides.")
        return api_name, model_name, agent_params_override

    def _get_api_section(self, api_name: str) -> Dict[str, Any]:
        """
        Returns the relevant subsection of the Model Library for the requested API.
        Raises ValueError if the API is not in the Model Library.
        """
        model_library = self.data['settings']['models'].get('model_library', {})
        if api_name not in model_library:
            raise ValueError(f"API '{api_name}' does not exist in the Model Library.")
        return model_library[api_name]

    @staticmethod
    def _find_class_for_model(api_section: Dict[str, Any], model_name: str) -> Tuple[str, Dict[str, Any]]:
        """
        Finds which class in the API section has the requested model in its 'models' dict.
        Returns (class_name, model_data). Raises ValueError if not found.
        """
        for candidate_class, class_config in api_section.items():
            if candidate_class == 'params':
                continue
            models_dict = class_config.get('models', {})
            if model_name in models_dict:
                return candidate_class, models_dict[model_name]
        raise ValueError(f"Model '{model_name}' not found in this API section.")

    @staticmethod
    def _get_model_identifier(api_name: str, model_name: str, model_data: Dict[str, Any]) -> str:
        """
        Reads the 'identifier' for the selected model from the YAML.
        Raises ValueError if it doesn't exist.
        """
        identifier = model_data.get('identifier')
        if not identifier:
            raise ValueError(f"Identifier not found for Model '{model_name}' under API '{api_name}' in Model Library.")
        return identifier

    @staticmethod
    def _merge_params(api_section: Dict[str, Any], class_name: str, model_data: Dict[str, Any], agent_params_override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges API-level, class-level, model-level params, and agent overrides in ascending specificity.
        Returns the merged dict.
        """
        api_level_params = api_section.get('params', {})
        class_level_params = api_section[class_name].get('params', {})
        model_level_params = model_data.get('params', {})
        merged_params = {**api_level_params, **class_level_params, **model_level_params, **agent_params_override}
        return merged_params

    # -----------------------------------
    # Model Handling
    # -----------------------------------

    @staticmethod
    def resolve_class(full_class_path: str, default_class: Optional[type] = None, context: str = "") -> type:
        """
        Dynamically resolve and return a class from a fully qualified path.
        Raises ValueError or ImportError if the class cannot be found.
        """
        if not full_class_path:
            if default_class is not None:
                return default_class
            raise ValueError(f"No class path provided for {context}")
        parts = full_class_path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid type format for {context}: '{full_class_path}'. Must be fully qualified (e.g., 'mymodule.MyClass').")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                raise ImportError(f"Module '{module_path}' not found for {context}.")
        except (ModuleNotFoundError, ImportError):
            raise ImportError(f"Module '{module_path}' not found for {context}.")
        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            raise ImportError(f"Class '{class_name}' not found in module '{module_path}' for {context}.")
        return getattr(module, class_name)

    @staticmethod
    def get_model(api_name: str, class_name: str, model_identifier: str) -> Any:
        """
        Dynamically imports and instantiates the Python class for the requested API/class/identifier.
        """
        module = Config._get_module(api_name)
        model_class = getattr(module, class_name)
        return model_class(model_identifier)

    @staticmethod
    def _get_module(api_name: str) -> ModuleType:
        """
        Retrieves the module for the given API. Tries the built-in apis folder first; if not found, loads from the custom_apis folder.
        """
        module = Config._try_load_built_in_api(api_name)
        if module is not None:
            return module
        return Config._load_custom_api(api_name)

    @staticmethod
    def _try_load_built_in_api(api_name: str) -> Optional[ModuleType]:
        """
        Attempts to load a built-in API module from the package's apis folder. Returns the module if found; otherwise returns None.
        """
        built_in_api_path = Path(__file__).parent / "apis" / f"{api_name}.py"
        if built_in_api_path.exists():
            return importlib.import_module(f".apis.{api_name}", package=__package__)
        return None

    @staticmethod
    def _load_custom_api(api_name: str) -> ModuleType:
        """
        Loads a custom API module from the project's custom_apis folder. Sets up a dummy package environment so that relative imports work correctly.
        """
        project_root = Config().project_root
        custom_api_dir = project_root / ".agentforge" / "custom_apis"
        custom_api_path = custom_api_dir / f"{api_name}.py"
        if not custom_api_path.exists():
            raise ImportError(
                f"Cannot find API module '{api_name}' in the built-in folder or at {custom_api_path}."
            )
        spec = importlib.util.spec_from_file_location(f"custom_apis.{api_name}", custom_api_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for custom API module '{api_name}'.")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "custom_apis"
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    # -----------------------------------
    # Utility Methods
    # -----------------------------------

    def find_config(self, category: str, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a configuration by name within a specified category.
        Returns the configuration dictionary for the specified name, or raises FileNotFoundError if not found.
        """
        def search_nested_dict(nested_dict, target):
            for key, value in nested_dict.items():
                if key == target:
                    return value
                elif isinstance(value, dict):
                    result = search_nested_dict(value, target)
                    if result is not None:
                        return result
            return None
        config = search_nested_dict(self.data.get(category, {}), config_name)
        if not config:
            raise FileNotFoundError(f"Config '{config_name}' not found in configuration.")
        return config

    def find_file_in_directory(self, directory: str, filename: str):
        """
        Recursively search for a file within a directory and its subdirectories.
        Returns the full path to the file if found, None otherwise.
        """
        directory = pathlib.Path(pathlib.Path(self.config_path) / directory)
        for file_path in directory.rglob(filename):
            return file_path
        return None

    def fix_prompt_placeholders(self, prompts):
        """
        Recursively traverse the prompts dictionary and convert any mappings like {'user_input': None} into strings '{user_input}'.
        Returns the fixed prompts data structure with placeholders as strings.
        """
        if isinstance(prompts, dict):
            if len(prompts) == 1 and list(prompts.values())[0] is None:
                key = list(prompts.keys())[0]
                if re.match(self.pattern, key):
                    return f"{{{key}}}"
            fixed_prompts = {}
            for key, value in prompts.items():
                fixed_key = self.fix_prompt_placeholders(key)
                fixed_value = self.fix_prompt_placeholders(value)
                fixed_prompts[fixed_key] = fixed_value
            return fixed_prompts
        if isinstance(prompts, list):
            return [self.fix_prompt_placeholders(item) for item in prompts]
        if not prompts:
            return ''
        return prompts

    @staticmethod
    def get_nested_dict(data: dict, path_parts: tuple):
        """
        Gets or creates a nested dictionary given the parts of a relative path.
        Returns a reference to the nested dictionary at the end of the path.
        """
        for part in path_parts:
            if part not in data:
                data[part] = {}
            data = data[part]
        return data

    @property
    def settings(self):
        """
        Returns the loaded settings as a Settings dataclass for dot notation access.
        """
        settings_dict = self.data.get('settings', {})
        return self.config_manager._build_settings(settings_dict)

    # -----------------------------------
    # Persona Handling
    # -----------------------------------

    def resolve_persona(self, cog_config: Optional[Dict[str, Any]] = None, agent_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Resolves the persona to use based on the deterministic hierarchy:
        1. Cog-defined persona (highest priority)
        2. Agent-defined persona
        3. System default persona (lowest priority)
        Returns the resolved persona data or None if personas are disabled.
        """
        settings = self.data['settings']
        if not settings['system']['persona'].get('enabled', False):
            return None
        persona_name = None
        if cog_config and 'persona' in cog_config:
            persona_name = cog_config['persona']
        elif agent_config and 'persona' in agent_config:
            persona_candidate = agent_config['persona']
            if isinstance(persona_candidate, str):
                persona_name = persona_candidate
        else:
            persona_name = settings['system']['persona'].get('name', 'default_assistant')
        if persona_name and persona_name not in self.data.get('personas', {}):
            raise FileNotFoundError(
                f"Selected Persona '{persona_name}' not found. "
                "Please make sure the corresponding persona file is in the personas folder"
            )
        return self.data['personas'][persona_name] if persona_name else None

    def load_persona(self, agent_config: dict) -> Optional[Dict[str, Any]]:
        """
        Loads the persona for the agent, if personas are enabled.
        Returns the loaded persona data.
        Raises FileNotFoundError if the persona file is not found.
        """
        persona_data = self.resolve_persona(agent_config=agent_config)
        return persona_data

# From agentforge/config.py
def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Reads and parses a YAML file, returning its contents as a Python dictionary.

    Parameters:
        file_path (str): The path to the YAML file to be read.

    Returns:
        dict: The contents of the YAML file as a dictionary. If the file is not found
        or an error occurs during parsing, an empty dictionary is returned.
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return {}
    except yaml.YAMLError:
        print(f"Error decoding YAML from {file_path}")
        return {}

# From agentforge/config.py
def find_project_root(self, root_path: Optional[str] = None) -> pathlib.Path:
        # If a root path was provided, use it to checking that .agentforge exists
        if root_path:
            custom_root = pathlib.Path(root_path).resolve()
            agentforge_dir = custom_root / ".agentforge"
            if agentforge_dir.is_dir():
                if self._debug: print(f"\n\nUsing custom project root: {custom_root}")
                return custom_root
            # Early return or raise an error if .agentforge isn't found in the custom path
            raise FileNotFoundError(f"No .agentforge found in custom root path: {custom_root}")

        # Otherwise, fall back to the original search logic
        script_dir = pathlib.Path(sys.argv[0]).resolve().parent
        current_dir = script_dir
        if self._debug: print(f"\n\nCurrent working directory: {os.getcwd()}")

        while current_dir != current_dir.parent:
            potential_dir = current_dir / ".agentforge"
            if self._debug: print(f"Checking {potential_dir}")
            if potential_dir.is_dir():
                if self._debug: print(f"Found .agentforge directory at: {current_dir}\n")
                return current_dir
            current_dir = current_dir.parent

        raise FileNotFoundError(f"Could not find the '.agentforge' directory starting from {script_dir}")

# From agentforge/config.py
def load_all_configurations(self):
        """
        Recursively loads all configuration data from YAML files under each subdirectory of the .agentforge folder.
        """
        with self._lock:
            for subdir, dirs, files in os.walk(self.config_path):
                for file in files:
                    if file.endswith(('.yaml', '.yml')):
                        subdir_path = pathlib.Path(subdir)
                        relative_path = subdir_path.relative_to(self.config_path)
                        nested_dict = self.get_nested_dict(self.data, relative_path.parts)
                        file_path = str(subdir_path / file)
                        data = self.load_yaml_file(file_path)
                        if data:
                            filename_without_ext = os.path.splitext(file)[0]
                            nested_dict[filename_without_ext] = data

# From agentforge/config.py
def save(self):
        """
        Saves changes to the configuration back to the system.yaml file,
        preserving structure, formatting, and comments.
        """
        with Config._lock:
            system_yaml_path = self.config_path / 'settings' / 'system.yaml'
            _yaml = YAML()
            _yaml.preserve_quotes = True
            try:
                with open(system_yaml_path, 'r') as yaml_file:
                    existing_data = _yaml.load(yaml_file)
                if 'settings' in self.data and 'system' in self.data['settings']:
                    for key, value in self.data['settings']['system'].items():
                        if isinstance(value, dict) and key in existing_data:
                            existing_data[key].update(value)
                            continue
                        existing_data[key] = value
                    with open(system_yaml_path, 'w') as yaml_file:
                        _yaml.dump(existing_data, yaml_file)
                    return
                print("No system settings to save.")
            except Exception as e:
                print(f"Error saving configuration to {system_yaml_path}: {e}")

# From agentforge/config.py
def reload(self):
        """
        Reloads configurations if on-the-fly reloading is enabled in settings.
        """
        if self.settings.system.misc.on_the_fly:
            self.load_all_configurations()

# From agentforge/config.py
def load_agent_data(self, agent_name: str) -> AgentConfig:
        """
        Loads configuration data for a specified agent, applying any overrides in the agent's config.
        Returns a structured AgentConfig object containing everything needed to run that agent.
        """
        agent = self.find_config('prompts', agent_name)
        api_name, class_name, model_name, final_params = self.resolve_model_overrides(agent)
        model = self.get_model(api_name, class_name, model_name)
        persona_data = self.load_persona(agent)
        prompts = self.fix_prompt_placeholders(agent.get('prompts', {}))
        settings = self.data.get('settings', {})
        default_debug_text = settings['system']['debug'].get('simulated_response', 'Simulated Text Goes Here!!!')
        simulated_response = agent.get('simulated_response', default_debug_text).strip()
        raw_agent_data = {
            'name': agent_name,
            'settings': settings,
            'model': model,
            'params': final_params,
            'persona': persona_data,
            'prompts': prompts,
            'simulated_response': simulated_response,
        }
        reserved_fields = {
            'name', 'settings', 'model', 'params', 'persona', 'prompts', 'simulated_response',
            'model_overrides'
        }
        for key, value in agent.items():
            if key not in raw_agent_data and key not in reserved_fields:
                raw_agent_data[key] = value
        return self.config_manager.build_agent_config(raw_agent_data)

# From agentforge/config.py
def load_cog_data(self, cog_name: str) -> CogConfig:
        """
        Loads configuration data for a specified cog, returning a validated structured CogConfig object.
        """
        raw_cog_data = self.find_config('cogs', cog_name)
        return self.config_manager.build_cog_config(raw_cog_data)

# From agentforge/config.py
def resolve_model_overrides(self, agent: dict) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Finds and merges all relevant model overrides into a final 4-tuple:
        (api_name, class_name, model_identifier, final_params).
        """
        api_name, model_name, agent_params_override = self._get_agent_api_and_model(agent)
        api_section = self._get_api_section(api_name)
        class_name, model_data = self._find_class_for_model(api_section, model_name)
        model_identifier = self._get_model_identifier(api_name, model_name, model_data)
        final_params = self._merge_params(api_section, class_name, model_data, agent_params_override)
        return api_name, class_name, model_identifier, final_params

# From agentforge/config.py
def resolve_class(full_class_path: str, default_class: Optional[type] = None, context: str = "") -> type:
        """
        Dynamically resolve and return a class from a fully qualified path.
        Raises ValueError or ImportError if the class cannot be found.
        """
        if not full_class_path:
            if default_class is not None:
                return default_class
            raise ValueError(f"No class path provided for {context}")
        parts = full_class_path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid type format for {context}: '{full_class_path}'. Must be fully qualified (e.g., 'mymodule.MyClass').")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                raise ImportError(f"Module '{module_path}' not found for {context}.")
        except (ModuleNotFoundError, ImportError):
            raise ImportError(f"Module '{module_path}' not found for {context}.")
        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            raise ImportError(f"Class '{class_name}' not found in module '{module_path}' for {context}.")
        return getattr(module, class_name)

# From agentforge/config.py
def get_model(api_name: str, class_name: str, model_identifier: str) -> Any:
        """
        Dynamically imports and instantiates the Python class for the requested API/class/identifier.
        """
        module = Config._get_module(api_name)
        model_class = getattr(module, class_name)
        return model_class(model_identifier)

# From agentforge/config.py
def find_config(self, category: str, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a configuration by name within a specified category.
        Returns the configuration dictionary for the specified name, or raises FileNotFoundError if not found.
        """
        def search_nested_dict(nested_dict, target):
            for key, value in nested_dict.items():
                if key == target:
                    return value
                elif isinstance(value, dict):
                    result = search_nested_dict(value, target)
                    if result is not None:
                        return result
            return None
        config = search_nested_dict(self.data.get(category, {}), config_name)
        if not config:
            raise FileNotFoundError(f"Config '{config_name}' not found in configuration.")
        return config

# From agentforge/config.py
def find_file_in_directory(self, directory: str, filename: str):
        """
        Recursively search for a file within a directory and its subdirectories.
        Returns the full path to the file if found, None otherwise.
        """
        directory = pathlib.Path(pathlib.Path(self.config_path) / directory)
        for file_path in directory.rglob(filename):
            return file_path
        return None

# From agentforge/config.py
def fix_prompt_placeholders(self, prompts):
        """
        Recursively traverse the prompts dictionary and convert any mappings like {'user_input': None} into strings '{user_input}'.
        Returns the fixed prompts data structure with placeholders as strings.
        """
        if isinstance(prompts, dict):
            if len(prompts) == 1 and list(prompts.values())[0] is None:
                key = list(prompts.keys())[0]
                if re.match(self.pattern, key):
                    return f"{{{key}}}"
            fixed_prompts = {}
            for key, value in prompts.items():
                fixed_key = self.fix_prompt_placeholders(key)
                fixed_value = self.fix_prompt_placeholders(value)
                fixed_prompts[fixed_key] = fixed_value
            return fixed_prompts
        if isinstance(prompts, list):
            return [self.fix_prompt_placeholders(item) for item in prompts]
        if not prompts:
            return ''
        return prompts

# From agentforge/config.py
def get_nested_dict(data: dict, path_parts: tuple):
        """
        Gets or creates a nested dictionary given the parts of a relative path.
        Returns a reference to the nested dictionary at the end of the path.
        """
        for part in path_parts:
            if part not in data:
                data[part] = {}
            data = data[part]
        return data

# From agentforge/config.py
def settings(self):
        """
        Returns the loaded settings as a Settings dataclass for dot notation access.
        """
        settings_dict = self.data.get('settings', {})
        return self.config_manager._build_settings(settings_dict)

# From agentforge/config.py
def resolve_persona(self, cog_config: Optional[Dict[str, Any]] = None, agent_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Resolves the persona to use based on the deterministic hierarchy:
        1. Cog-defined persona (highest priority)
        2. Agent-defined persona
        3. System default persona (lowest priority)
        Returns the resolved persona data or None if personas are disabled.
        """
        settings = self.data['settings']
        if not settings['system']['persona'].get('enabled', False):
            return None
        persona_name = None
        if cog_config and 'persona' in cog_config:
            persona_name = cog_config['persona']
        elif agent_config and 'persona' in agent_config:
            persona_candidate = agent_config['persona']
            if isinstance(persona_candidate, str):
                persona_name = persona_candidate
        else:
            persona_name = settings['system']['persona'].get('name', 'default_assistant')
        if persona_name and persona_name not in self.data.get('personas', {}):
            raise FileNotFoundError(
                f"Selected Persona '{persona_name}' not found. "
                "Please make sure the corresponding persona file is in the personas folder"
            )
        return self.data['personas'][persona_name] if persona_name else None

# From agentforge/config.py
def load_persona(self, agent_config: dict) -> Optional[Dict[str, Any]]:
        """
        Loads the persona for the agent, if personas are enabled.
        Returns the loaded persona data.
        Raises FileNotFoundError if the persona file is not found.
        """
        persona_data = self.resolve_persona(agent_config=agent_config)
        return persona_data

# From agentforge/config.py
def search_nested_dict(nested_dict, target):
            for key, value in nested_dict.items():
                if key == target:
                    return value
                elif isinstance(value, dict):
                    result = search_nested_dict(value, target)
                    if result is not None:
                        return result
            return None


# From core/trail_recorder.py
class TrailRecorder:
    """Encapsulates thought trail tracking and related logging functionality."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the trail recorder.
        
        Args:
            enabled: Whether trail recording is enabled
        """
        self.logger = Logger(name="TrailRecorder", default_logger="trail")
        self.enabled = enabled
        self.trail: List[ThoughtTrailEntry] = []
        self._execution_counter = 0
        
        # TODO: Consider adding max_entries parameter for trail size limits in the future
        # self.max_entries = max_entries
    
    def record_agent_output(self, agent_id: str, output: any, notes: Optional[str] = None, error: Optional[str] = None) -> None:
        """
        Record an agent's output in the trail.
        
        Args:
            agent_id: ID of the agent that produced the output
            output: The agent's output to record
            notes: Optional notes about this execution
            error: Optional error information if the agent failed
        """
        if not self.enabled:
            return
            
        self._execution_counter += 1
        entry = ThoughtTrailEntry(
            agent_id=agent_id,
            output=output,
            notes=notes,
            execution_order=self._execution_counter,
            error=error
        )
        self.trail.append(entry)
        self._log_trail_entry(entry)
        
        # TODO: Implement trail size management if max_entries is added
        # if self.max_entries and len(self.trail) > self.max_entries:
        #     self.trail.pop(0)  # Remove oldest entry
    
    def get_trail(self) -> List[ThoughtTrailEntry]:
        """
        Get a copy of the current trail.
        
        Returns:
            List of ThoughtTrailEntry objects representing the execution trail
        """
        return self.trail.copy()
    
    def reset_trail(self) -> None:
        """Clear the trail and reset counters."""
        self.trail.clear()
        self._execution_counter = 0
    
    def _log_trail_entry(self, entry: ThoughtTrailEntry) -> None:
        """
        Log trail entry to debug output.
        
        Args:
            entry: The trail entry to log
        """
        log_message = f"******\n{entry.agent_id}\n******\n{entry.output}\n******"
        if entry.error:
            log_message += f"\nERROR: {entry.error}"
        
        self.logger.debug(log_message)

# From core/trail_recorder.py
def record_agent_output(self, agent_id: str, output: any, notes: Optional[str] = None, error: Optional[str] = None) -> None:
        """
        Record an agent's output in the trail.
        
        Args:
            agent_id: ID of the agent that produced the output
            output: The agent's output to record
            notes: Optional notes about this execution
            error: Optional error information if the agent failed
        """
        if not self.enabled:
            return
            
        self._execution_counter += 1
        entry = ThoughtTrailEntry(
            agent_id=agent_id,
            output=output,
            notes=notes,
            execution_order=self._execution_counter,
            error=error
        )
        self.trail.append(entry)
        self._log_trail_entry(entry)

# From core/trail_recorder.py
def get_trail(self) -> List[ThoughtTrailEntry]:
        """
        Get a copy of the current trail.
        
        Returns:
            List of ThoughtTrailEntry objects representing the execution trail
        """
        return self.trail.copy()

# From core/trail_recorder.py
def reset_trail(self) -> None:
        """Clear the trail and reset counters."""
        self.trail.clear()
        self._execution_counter = 0

from agentforge.config_structs import CogConfig

# From core/agent_registry.py
class AgentRegistry:
    """
    Factory for building agent instances from cog configuration.
    Handles class resolution, template assignment, and instantiation.
    """

    @staticmethod
    def build_agents(cog_config: CogConfig) -> Dict[str, Agent]:
        """
        Build a mapping of agent IDs to initialized agent instances from cog configuration.
        
        Args:
            cog_config: Structured cog configuration object
            
        Returns:
            Dict[str, Agent]: Mapping of agent IDs to agent instances
        """
        config = Config()
        agents = {}
        
        for agent_def in cog_config.cog.agents:
            agent_id = agent_def.id
            agent_class = AgentRegistry._resolve_agent_class(agent_def, config)
            agent_name = agent_def.template_file or agent_def.id
            agents[agent_id] = agent_class(agent_name=agent_name)
            
        return agents

    @staticmethod
    def _resolve_agent_class(agent_def, config: Config) -> type:
        """
        Resolve and return the agent class for a given agent definition.
        Assumes validation has already been performed by ConfigManager.
        
        Args:
            agent_def: Agent definition from cog config
            config: Config instance for class resolution
            
        Returns:
            type: The resolved agent class
        """
        return config.resolve_class(
            agent_def.type, 
            default_class=Agent,
            context=f"agent '{agent_def.id}'"
        )

# From core/agent_registry.py
def build_agents(cog_config: CogConfig) -> Dict[str, Agent]:
        """
        Build a mapping of agent IDs to initialized agent instances from cog configuration.
        
        Args:
            cog_config: Structured cog configuration object
            
        Returns:
            Dict[str, Agent]: Mapping of agent IDs to agent instances
        """
        config = Config()
        agents = {}
        
        for agent_def in cog_config.cog.agents:
            agent_id = agent_def.id
            agent_class = AgentRegistry._resolve_agent_class(agent_def, config)
            agent_name = agent_def.template_file or agent_def.id
            agents[agent_id] = agent_class(agent_name=agent_name)
            
        return agents

from dataclasses import field
from config_structs import PersonaSettings
from config_structs import DebugSettings
from config_structs import LoggingSettings
from config_structs import MiscSettings
from config_structs import PathSettings
from config_structs import SystemSettings
from config_structs import Settings
from config_structs import CogAgentDef
from config_structs import CogMemoryDef
from config_structs import CogFlowTransition
from config_structs import CogFlow
from config_structs import CogDefinition

# From core/config_manager.py
class ConfigManager:
    """
    Responsible for all configuration validation, normalization, merging, and structured config object construction for AgentForge.
    
    - Accepts raw config dicts (from Config)
    - Validates schema, types, and required fields
    - Normalizes and merges config data as needed
    - Returns structured config objects for use by Agent, Cog, etc.
    
    NOTE: Config objects are designed to be immutable by convention - they should never be mutated in place.
    For hot-reload support, always replace the entire config object with a new one from ConfigManager.
    """

    def __init__(self):
        """Initialize ConfigManager with required processors."""
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize any required processors or internal state."""
        pass

    def debug_print(self, msg: str, settings: dict):
        """Print debug messages if debug mode is enabled in settings dict."""
        debug_mode = False
        # Try to get debug flag from settings dict
        if isinstance(settings, dict):
            debug_mode = settings.get('system', {}).get('debug', {}).get('mode', False)
        if debug_mode:
            print(msg)

    # ==============================================================================
    # Builder Methods
    # ==============================================================================

    def build_agent_config(self, raw_agent_data: Dict[str, Any]) -> AgentConfig:
        """
        Validates and normalizes a raw agent config dict, returning a structured AgentConfig object.
        Raises ValueError if required fields are missing or invalid.
        """
        try:
            self._validate_agent_config(raw_agent_data)
            agent_config = self._normalize_agent_config(raw_agent_data)
            self.debug_print(f"Agent config for '{raw_agent_data.get('name', '<unknown>')}' built successfully.", raw_agent_data.get('settings', {}))
            return agent_config
        except Exception as e:
            self.debug_print(f"Failed to build agent config: {e}", raw_agent_data.get('settings', {}))
            raise

    def _validate_agent_config(self, raw_agent_data: Dict[str, Any]) -> None:
        """Validate required fields and structure for agent config."""
        required_keys = ['params', 'prompts', 'settings']
        for key in required_keys:
            if key not in raw_agent_data:
                raise ValueError(f"Agent config missing required key '{key}' for agent '{raw_agent_data.get('name', '<unknown>')}'.")

        if not raw_agent_data.get('name'):
            raise ValueError("Agent config missing required 'name' field.")

        prompts = raw_agent_data.get('prompts', {})
        if not prompts or not isinstance(prompts, dict):
            raise ValueError(f"Agent '{raw_agent_data.get('name', '<unknown>')}' must have a non-empty 'prompts' dictionary.")

        try:
            self._validate_prompt_format(prompts)
        except ValueError as e:
            raise ValueError(f"Agent '{raw_agent_data.get('name', '<unknown>')}' has invalid prompt format: {e}")

        model = raw_agent_data.get('model')
        if model is None:
            raise ValueError(f"Agent '{raw_agent_data.get('name', '<unknown>')}' must have a 'model' specified.")

        params = raw_agent_data.get('params', {})
        if not isinstance(params, dict):
            raise ValueError(f"Agent '{raw_agent_data.get('name', '<unknown>')}' params must be a dictionary.")

        persona = raw_agent_data.get('persona')
        if persona is not None and not isinstance(persona, dict):
            self.debug_print(f"Agent '{raw_agent_data.get('name', '<unknown>')}' persona should be a dictionary, got {type(persona)}", raw_agent_data.get('settings', {}))

    def _normalize_agent_config(self, raw_agent_data: Dict[str, Any]) -> AgentConfig:
        """Normalize and construct AgentConfig object from validated raw data."""
        settings = self._build_settings(raw_agent_data['settings'])
        reserved_fields = {'name', 'settings', 'model', 'params', 'prompts', 'persona', 'simulated_response', 'parse_response_as'}
        custom_fields = {key: value for key, value in raw_agent_data.items() if key not in reserved_fields}
        return AgentConfig(
            name=raw_agent_data['name'],
            settings=settings,
            model=raw_agent_data['model'],
            params=raw_agent_data['params'],
            prompts=raw_agent_data['prompts'],
            persona=raw_agent_data.get('persona'),
            simulated_response=raw_agent_data.get('simulated_response'),
            parse_response_as=raw_agent_data.get('parse_response_as'),
            custom_fields=custom_fields
        )

    def _validate_prompt_format(self, prompts: Dict[str, Any]) -> None:
        """
        Validates that the prompts dictionary has the correct format.
        Raises ValueError if the prompts do not contain only 'system' and 'user' keys,
        or if the sub-prompts are not dictionaries or strings.
        """
        if set(prompts.keys()) != {'system', 'user'}:
            raise ValueError(
                "Prompts should contain only 'system' and 'user' keys. "
                "Please check the prompt YAML file format."
            )
        for prompt_type in ['system', 'user']:
            prompt_value = prompts.get(prompt_type, {})
            if not isinstance(prompt_value, (dict, str)):
                raise ValueError(
                    f"The '{prompt_type}' prompt should be either a string or a dictionary of sub-prompts."
                )

    def build_cog_config(self, raw_cog_data: Dict[str, Any]) -> CogConfig:
        """
        Validates and normalizes a raw cog config dict, returning a structured CogConfig object.
        Raises ValueError if required fields are missing or invalid.
        """
        try:
            self._validate_cog_config(raw_cog_data)
            cog_config = self._normalize_cog_config(raw_cog_data)
            # Try to get debug flag from cog settings if present, else from agent/global settings if available
            settings = None
            if 'settings' in raw_cog_data:
                settings = raw_cog_data['settings']
            elif 'cog' in raw_cog_data and 'settings' in raw_cog_data['cog']:
                settings = raw_cog_data['cog']['settings']
            self.debug_print("Cog config built successfully.", settings or {})
            return cog_config
        except Exception as e:
            settings = None
            if 'settings' in raw_cog_data:
                settings = raw_cog_data['settings']
            elif 'cog' in raw_cog_data and 'settings' in raw_cog_data['cog']:
                settings = raw_cog_data['cog']['settings']
            self.debug_print(f"Failed to build cog config: {e}", settings or {})
            raise

    def _validate_cog_config(self, raw_cog_data: Dict[str, Any]) -> None:
        """Validate required fields and structure for cog config."""
        if 'cog' not in raw_cog_data:
            raise ValueError("Cog config must have a 'cog' dictionary defined.")
        raw_cog = raw_cog_data['cog']
        if not isinstance(raw_cog, dict):
            raise ValueError("Cog 'cog' value must be a dictionary.")
        if 'flow' not in raw_cog or raw_cog.get('flow') is None:
            raise ValueError("Cog 'flow' must be defined.")
        if 'agents' not in raw_cog or not isinstance(raw_cog.get('agents'), list):
            raise ValueError("Cog 'agents' must be a list.")
        # Additional validation can be added here as needed

    def _normalize_cog_config(self, raw_cog_data: Dict[str, Any]) -> CogConfig:
        """Normalize and construct CogConfig object from validated raw data."""
        raw_cog = raw_cog_data['cog']
        agents = self._build_cog_agents(raw_cog.get('agents', []))
        memory = self._build_cog_memory(raw_cog.get('memory', []))
        flow = self._build_cog_flow(raw_cog.get('flow'))
        self._validate_agent_modules(agents)
        self._validate_flow_references(flow, agents)
        cog_def = CogDefinition(
            name=raw_cog.get('name', ''),
            description=raw_cog.get('description'),
            persona=raw_cog.get('persona'),
            trail_logging=raw_cog.get('trail_logging', True),
            agents=agents,
            memory=memory,
            flow=flow
        )
        custom_fields = {key: value for key, value in raw_cog_data.items() if key != 'cog'}
        return CogConfig(
            cog=cog_def,
            custom_fields=custom_fields
        )

    # ==============================================================================
    # Private Helper Methods
    # ==============================================================================

    def _build_settings(self, raw_settings: Dict[str, Any]) -> Settings:
        """Build structured Settings object from raw settings dict."""
        raw_system = raw_settings.get('system', {})
        
        persona_settings = PersonaSettings(
            enabled=raw_system.get('persona', {}).get('enabled', True),
            name=raw_system.get('persona', {}).get('name', 'default_assistant'),
            static_char_cap=raw_system.get('persona', {}).get('static_char_cap', 8000)
        )
        
        debug_settings = DebugSettings(
            mode=raw_system.get('debug', {}).get('mode', False),
            save_memory=raw_system.get('debug', {}).get('save_memory', False),
            simulated_response=raw_system.get('debug', {}).get('simulated_response', 
                "Text designed to simulate an LLM response for debugging purposes without invoking the model.")
        )
        
        logging_settings = LoggingSettings(
            enabled=raw_system.get('logging', {}).get('enabled', True),
            console_level=raw_system.get('logging', {}).get('console_level', 'warning'),
            folder=raw_system.get('logging', {}).get('folder', './logs'),
            files=raw_system.get('logging', {}).get('files', {})
        )
        
        misc_settings = MiscSettings(
            on_the_fly=raw_system.get('misc', {}).get('on_the_fly', True)
        )
        
        path_settings = PathSettings(
            files=raw_system.get('paths', {}).get('files', './files')
        )
        
        system_settings = SystemSettings(
            persona=persona_settings,
            debug=debug_settings,
            logging=logging_settings,
            misc=misc_settings,
            paths=path_settings
        )
        
        return Settings(
            system=system_settings,
            models=raw_settings.get('models', {}),
            storage=raw_settings.get('storage', {})
        )

    def _build_cog_agents(self, raw_agents: List[Dict[str, Any]]) -> List[CogAgentDef]:
        """Build list of CogAgentDef objects from raw agent definitions."""
        if not isinstance(raw_agents, list):
            raise ValueError("Cog agents must be a list.")
        
        agents = []
        for agent_def in raw_agents:
            if not isinstance(agent_def, dict):
                raise ValueError("Each agent definition must be a dictionary.")
            
            agent_id = agent_def.get('id')
            if not agent_id:
                raise ValueError("Every agent must have an 'id'.")
            
            # Validate that at least one of type or template_file is present
            if 'type' not in agent_def and 'template_file' not in agent_def:
                raise ValueError(f"Agent '{agent_id}' must have at least a 'type' or a 'template_file' defined.")
            
            agents.append(CogAgentDef(
                id=agent_id,
                template_file=agent_def.get('template_file'),
                type=agent_def.get('type')
            ))
        
        return agents

    def _build_cog_memory(self, raw_memory: List[Dict[str, Any]]) -> List[CogMemoryDef]:
        """Build list of CogMemoryDef objects from raw memory definitions."""
        if not isinstance(raw_memory, list):
            return []  # Memory is optional
        
        memory_nodes = []
        for mem_def in raw_memory:
            if not isinstance(mem_def, dict):
                raise ValueError("Each memory definition must be a dictionary.")
            
            mem_id = mem_def.get('id')
            if not mem_id:
                raise ValueError("Every memory node must have an 'id'.")
            
            # Normalize query_before and update_after to handle both string and list formats
            query_before = mem_def.get('query_before')
            if isinstance(query_before, str):
                query_before = [query_before]
            elif query_before is None:
                query_before = []
            
            update_after = mem_def.get('update_after')
            if isinstance(update_after, str):
                update_after = [update_after]
            elif update_after is None:
                update_after = []
            
            memory_nodes.append(CogMemoryDef(
                id=mem_id,
                type=mem_def.get('type'),
                collection_id=mem_def.get('collection_id'),
                query_before=query_before,
                query_keys=mem_def.get('query_keys', []),
                update_after=update_after,
                update_keys=mem_def.get('update_keys', [])
            ))
        
        return memory_nodes

    def _build_cog_flow(self, raw_flow: Optional[Dict[str, Any]]) -> Optional[CogFlow]:
        """Build CogFlow object from raw flow definition."""
        if not raw_flow:
            raise ValueError("Flow must be defined.")
        
        if not isinstance(raw_flow, dict):
            raise ValueError("Flow must be a dictionary.")
        
        # Validate required flow fields
        if 'start' not in raw_flow:
            raise ValueError("Flow must have a 'start' key.")
        
        if 'transitions' not in raw_flow:
            raise ValueError("Flow must have a 'transitions' dictionary defined.")
        
        raw_transitions = raw_flow['transitions']
        if not isinstance(raw_transitions, dict):
            raise ValueError("Flow 'transitions' must be a dictionary.")
        
        # Parse transitions
        transitions = {}
        for agent_id, transition_def in raw_transitions.items():
            transitions[agent_id] = self._parse_flow_transition(transition_def)
        
        return CogFlow(
            start=raw_flow['start'],
            transitions=transitions
        )

    def _parse_flow_transition(self, transition_def: Any) -> CogFlowTransition:
        """Parse a single flow transition definition into a CogFlowTransition object."""
        if isinstance(transition_def, str):
            # Direct transition to another agent
            return CogFlowTransition(
                type="direct",
                next_agent=transition_def
            )
        
        if isinstance(transition_def, dict):
            # Check if this is an end transition
            if 'end' in transition_def:
                return CogFlowTransition(
                    type="end",
                    end=transition_def['end']
                )
            
            # Check for decision transition
            reserved_keys = {'fallback', 'max_visits', 'end'}
            decision_key = None
            decision_map = {}
            
            for key, value in transition_def.items():
                if key not in reserved_keys:
                    if decision_key is not None:
                        raise ValueError(f"Multiple decision keys found in transition: {decision_key}, {key}")
                    decision_key = key
                    if isinstance(value, dict):
                        decision_map = value
                    else:
                        # Single value decision
                        decision_map = {str(value): value}
            
            if decision_key:
                return CogFlowTransition(
                    type="decision",
                    decision_key=decision_key,
                    decision_map=decision_map,
                    fallback=transition_def.get('fallback'),
                    max_visits=transition_def.get('max_visits')
                )
            
            # If no decision key found, this might be a malformed transition
            raise ValueError(f"Invalid transition definition: {transition_def}")
        
        raise ValueError(f"Transition must be string or dict, got: {type(transition_def)}")

    def _validate_agent_modules(self, agents: List[CogAgentDef]) -> None:
        """Validate that agent modules exist and can be imported."""
        for agent in agents:
            if agent.type:
                self._validate_module_exists(agent.type, agent.id)

    def _validate_module_exists(self, full_class_path: str, agent_id: str) -> None:
        """
        Validate that the module exists and contains the specified class.
        
        Args:
            full_class_path: The full module.ClassName path
            agent_id: The agent ID for error messages
            
        Raises:
            ValueError: If the module path format is invalid
            ImportError: If the module or class cannot be found
        """
        from agentforge.config import Config
        # Use Config.resolve_class to validate - we don't need the returned class
        Config.resolve_class(full_class_path, context=f"agent '{agent_id}'")

    def _validate_flow_references(self, flow: CogFlow, agents: List[CogAgentDef]) -> None:
        """Validate that flow references valid agent IDs."""
        if not flow:
            return
            
        agent_ids = {agent.id for agent in agents}
        
        # Validate start agent exists
        if flow.start not in agent_ids:
            raise ValueError(f"Flow start agent '{flow.start}' not found in agents list.")
        
        # Validate all transition references
        for agent_id, transition in flow.transitions.items():
            if agent_id not in agent_ids:
                raise ValueError(f"Transition defined for unknown agent '{agent_id}'.")
            
            # Check transition targets
            if transition.type == "direct" and transition.next_agent:
                if transition.next_agent not in agent_ids:
                    raise ValueError(f"Transition from '{agent_id}' references unknown agent '{transition.next_agent}'.")
            elif transition.type == "decision" and transition.decision_map:
                for decision_value, target_agent in transition.decision_map.items():
                    if target_agent not in agent_ids:
                        raise ValueError(f"Decision transition from '{agent_id}' references unknown agent '{target_agent}'.")
            
            # Check fallback references
            if transition.fallback and transition.fallback not in agent_ids:
                raise ValueError(f"Fallback from '{agent_id}' references unknown agent '{transition.fallback}'.")

        # Warn if no end transition is present
        has_end_transition = any(
            transition.type == "end" or transition.end
            for transition in flow.transitions.values()
        )
        
        if not has_end_transition:
            print("Flow has no 'end:' transition; cog may loop forever.")

# From core/config_manager.py
def debug_print(self, msg: str, settings: dict):
        """Print debug messages if debug mode is enabled in settings dict."""
        debug_mode = False
        # Try to get debug flag from settings dict
        if isinstance(settings, dict):
            debug_mode = settings.get('system', {}).get('debug', {}).get('mode', False)
        if debug_mode:
            print(msg)

# From core/config_manager.py
def build_agent_config(self, raw_agent_data: Dict[str, Any]) -> AgentConfig:
        """
        Validates and normalizes a raw agent config dict, returning a structured AgentConfig object.
        Raises ValueError if required fields are missing or invalid.
        """
        try:
            self._validate_agent_config(raw_agent_data)
            agent_config = self._normalize_agent_config(raw_agent_data)
            self.debug_print(f"Agent config for '{raw_agent_data.get('name', '<unknown>')}' built successfully.", raw_agent_data.get('settings', {}))
            return agent_config
        except Exception as e:
            self.debug_print(f"Failed to build agent config: {e}", raw_agent_data.get('settings', {}))
            raise

# From core/config_manager.py
def build_cog_config(self, raw_cog_data: Dict[str, Any]) -> CogConfig:
        """
        Validates and normalizes a raw cog config dict, returning a structured CogConfig object.
        Raises ValueError if required fields are missing or invalid.
        """
        try:
            self._validate_cog_config(raw_cog_data)
            cog_config = self._normalize_cog_config(raw_cog_data)
            # Try to get debug flag from cog settings if present, else from agent/global settings if available
            settings = None
            if 'settings' in raw_cog_data:
                settings = raw_cog_data['settings']
            elif 'cog' in raw_cog_data and 'settings' in raw_cog_data['cog']:
                settings = raw_cog_data['cog']['settings']
            self.debug_print("Cog config built successfully.", settings or {})
            return cog_config
        except Exception as e:
            settings = None
            if 'settings' in raw_cog_data:
                settings = raw_cog_data['settings']
            elif 'cog' in raw_cog_data and 'settings' in raw_cog_data['cog']:
                settings = raw_cog_data['cog']['settings']
            self.debug_print(f"Failed to build cog config: {e}", settings or {})
            raise

from agentforge.config_structs.cog_config_structs import CogFlow
from agentforge.config_structs.cog_config_structs import CogFlowTransition

# From core/transition_resolver.py
class TransitionResolverError(Exception):
    """Custom exception for TransitionResolver errors."""
    pass

# From core/transition_resolver.py
class TransitionResolver:
    """
    Resolves agent transitions based on current state and flow configuration.
    
    This class encapsulates all transition logic for Cog workflows, including:
    - Direct transitions to next agents
    - Decision-based transitions using agent output
    - End transitions that terminate the flow
    - Visit count tracking for loop prevention
    """
    
    def __init__(self, flow: CogFlow):
        """
        Initialize the TransitionResolver with the given flow configuration.
        """
        self.flow = flow
        self.logger = Logger(name="TransitionResolver", default_logger="transition_resolver")
        self.visit_counts = {}
    
    def get_next_agent(self, current_agent_id: str, agent_outputs: Dict[str, Any]) -> Optional[str]:
        """
        Determines the next agent in the flow based on transition rules.
        Delegates each step to focused helper methods for clarity.
        """
        self.logger.info(f"Getting next agent for {current_agent_id}")
        agent_transition = self._get_agent_transition(current_agent_id)
        if self._is_end_transition(agent_transition):
            return self._handle_end_transition(current_agent_id)
        next_agent = self._handle_transition(current_agent_id, agent_transition, agent_outputs)
        self.logger.info(f"Next agent for {current_agent_id}: {next_agent}")
        return next_agent

    def _is_end_transition(self, transition: CogFlowTransition) -> bool:
        """Check if the transition is an end transition."""
        return transition.type == "end" or getattr(transition, 'end', False)

    def _handle_end_transition(self, current_agent_id: str) -> None:
        """Handle end transition logic and log appropriately."""
        self.logger.log(f"End transition for {current_agent_id}, returning None", "debug", "Transition")
        return None

    def _handle_transition(self, current_agent_id: str, transition: CogFlowTransition, agent_outputs: Dict[str, Any]) -> Optional[str]:
        """Delegate to the appropriate transition handler based on type."""
        if transition.type == 'direct':
            return self._handle_direct_transition(transition)
        if transition.type == 'end':
            return self._handle_end_transition(current_agent_id)
        return self._handle_decision_transition(current_agent_id, transition, agent_outputs)

    def _handle_direct_transition(self, transition: CogFlowTransition) -> Optional[str]:
        """Handle direct transition type."""
        return transition.next_agent

    def reset_visit_counts(self) -> None:
        """Reset visit tracking for new flow execution."""
        self.visit_counts = {}
        self.logger.log("Reset visit counts for new flow execution", "debug", "Transition")
    
    def get_visit_count(self, agent_id: str) -> int:
        """
        Get current visit count for an agent.
        
        Args:
            agent_id: The agent ID to check
            
        Returns:
            The number of times this agent has been visited
        """
        return self.visit_counts.get(agent_id, 0)
    
    def _get_agent_transition(self, agent_id: str) -> CogFlowTransition:
        """
        Get the transition definition for the specified agent.
        Raises TransitionResolverError if no transition is defined.
        """
        transitions = self.flow.transitions
        agent_transition = transitions.get(agent_id)
        if agent_transition is None:
            raise TransitionResolverError(f"No transition defined for agent: {agent_id}")
        self.logger.log(f"Transition data: {agent_transition}", "debug", "Transition")
        return agent_transition
    
    def _increment_visit_count(self, agent_id: str) -> int:
        """
        Increment the visit count for the given agent and return the new count.
        """
        self.visit_counts[agent_id] = self.visit_counts.get(agent_id, 0) + 1
        return self.visit_counts[agent_id]

    def _has_exceeded_max_visits(self, agent_id: str, max_visits: int) -> bool:
        """
        Check if the visit count for the agent exceeds max_visits.
        """
        return self.visit_counts.get(agent_id, 0) > max_visits

    def _handle_decision_transition(self, current_agent_id: str, transition: CogFlowTransition, 
                                  agent_outputs: Dict[str, Any]) -> Optional[str]:
        """
        Handle decision-based transitions:
          - Check if the max_visits limit for this agent is exceeded.
          - Otherwise, use the decision variable to select the next agent.
        """
        # Check if max_visits is exceeded (only for decision transitions)
        if transition.max_visits:
            self._increment_visit_count(current_agent_id)
            if self._has_exceeded_max_visits(current_agent_id, transition.max_visits):
                self.logger.log(f"Max visits ({transition.max_visits}) exceeded for {current_agent_id}, using fallback: {transition.fallback}", "debug", "Transition")
                return transition.fallback
        # Handle decision-based transition
        return self._handle_decision(current_agent_id, transition, agent_outputs)
    
    def _get_decision_value(self, agent_outputs: Dict[str, Any], agent_id: str, decision_key: str) -> Any:
        """
        Retrieve the decision value from the agent's output using the decision key.
        """
        if agent_id not in agent_outputs:
            return None
        return agent_outputs[agent_id].get(decision_key)

    def _map_decision_to_next_agent(self, decision_value: Any, decision_map: Dict[Any, str], fallback_branch: Optional[str]) -> Optional[str]:
        """
        Map the decision value to the next agent using the decision map, with normalization and fallback.
        """
        if decision_value is None:
            return fallback_branch
        str_decision_value = str(decision_value).lower()
        string_key_map = {str(k).lower(): v for k, v in decision_map.items()}
        return string_key_map.get(str_decision_value, fallback_branch)

    def _handle_decision(self, current_agent_id: str, transition: CogFlowTransition, 
                        agent_outputs: Dict[str, Any]) -> Optional[str]:
        """
        Handles decision-based transitions by determining the decision key
        and selecting the appropriate branch based on the agent's output.
        """
        self.logger.log(f"Handling transition for {current_agent_id}: {transition}", "debug", "Decision")
        decision_key = transition.decision_key
        fallback_branch = transition.fallback
        decision_map = transition.decision_map
        self.logger.log(f"Decision key={decision_key}, fallback={fallback_branch}", "debug", "Decision")
        if not decision_key:
            self.logger.log(f"No decision key, returning fallback: {fallback_branch}", "debug", "Decision")
            return fallback_branch
        self.logger.log(f"Decision map={decision_map}", "debug", "Decision")
        decision_value = self._get_decision_value(agent_outputs, current_agent_id, decision_key)
        self.logger.log(f"Decision value={decision_value}", "debug", "Decision")
        next_agent = self._map_decision_to_next_agent(decision_value, decision_map, fallback_branch)
        self.logger.log(f"Next agent='{next_agent}'", "debug", "Decision")
        self.logger.log(f"Decision summary: key='{decision_key}', value='{decision_value}', result='{next_agent}'", 'debug', 'Decision')
        if next_agent is None:
            self.logger.log(f"No matching branch found for decision value '{decision_value}' and no fallback branch defined", "warning", "Decision")
            self.logger.log(f"No match and no fallback", "debug", "Decision")
        self.logger.log(f"Returning next agent='{next_agent}'", "debug", "Decision")
        return next_agent

# From core/transition_resolver.py
def get_next_agent(self, current_agent_id: str, agent_outputs: Dict[str, Any]) -> Optional[str]:
        """
        Determines the next agent in the flow based on transition rules.
        Delegates each step to focused helper methods for clarity.
        """
        self.logger.info(f"Getting next agent for {current_agent_id}")
        agent_transition = self._get_agent_transition(current_agent_id)
        if self._is_end_transition(agent_transition):
            return self._handle_end_transition(current_agent_id)
        next_agent = self._handle_transition(current_agent_id, agent_transition, agent_outputs)
        self.logger.info(f"Next agent for {current_agent_id}: {next_agent}")
        return next_agent

# From core/transition_resolver.py
def reset_visit_counts(self) -> None:
        """Reset visit tracking for new flow execution."""
        self.visit_counts = {}
        self.logger.log("Reset visit counts for new flow execution", "debug", "Transition")

# From core/transition_resolver.py
def get_visit_count(self, agent_id: str) -> int:
        """
        Get current visit count for an agent.
        
        Args:
            agent_id: The agent ID to check
            
        Returns:
            The number of times this agent has been visited
        """
        return self.visit_counts.get(agent_id, 0)


# From core/agent_runner.py
class AgentRunner:
    """
    Handles the execution of individual agents with retry logic and logging.
    Encapsulates agent invocation concerns separate from flow orchestration.
    """

    def __init__(self):
        """Initialize AgentRunner with its own logger instance."""
        self.logger = Logger("AgentRunner", "agent_runner")

    def run_agent(self, agent_id: str, agent, context: dict, state: dict, memory: dict, max_attempts: int = 3) -> Any:
        """
        Execute an agent with retry logic.
        
        Args:
            agent_id: The ID of the agent to execute (for logging)
            agent: The agent instance to execute
            context: External context data to pass to agent
            state: Internal state data to pass to agent
            memory: Memory data to pass to agent
            max_attempts: Maximum number of retry attempts
            
        Returns:
            Agent output on success
            
        Raises:
            Exception: If agent fails after max attempts
        """
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            self.logger.debug(f"Executing agent '{agent_id}' (attempt {attempts}/{max_attempts})")
            
            agent_output = agent.run(_ctx=context, _state=state, _mem=memory)
            
            if not agent_output:
                self.logger.warning(f"No output from agent '{agent_id}', retrying... (Attempt {attempts})")
                continue
                
            self.logger.debug(f"Agent '{agent_id}' executed successfully on attempt {attempts}")
            return agent_output

        self.logger.error(f"Max attempts reached for agent '{agent_id}' with no valid output.")
        raise Exception(f"Failed to get valid response from {agent_id}. We recommend checking the agent's input/output logs.")

# From core/agent_runner.py
def run_agent(self, agent_id: str, agent, context: dict, state: dict, memory: dict, max_attempts: int = 3) -> Any:
        """
        Execute an agent with retry logic.
        
        Args:
            agent_id: The ID of the agent to execute (for logging)
            agent: The agent instance to execute
            context: External context data to pass to agent
            state: Internal state data to pass to agent
            memory: Memory data to pass to agent
            max_attempts: Maximum number of retry attempts
            
        Returns:
            Agent output on success
            
        Raises:
            Exception: If agent fails after max attempts
        """
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            self.logger.debug(f"Executing agent '{agent_id}' (attempt {attempts}/{max_attempts})")
            
            agent_output = agent.run(_ctx=context, _state=state, _mem=memory)
            
            if not agent_output:
                self.logger.warning(f"No output from agent '{agent_id}', retrying... (Attempt {attempts})")
                continue
                
            self.logger.debug(f"Agent '{agent_id}' executed successfully on attempt {attempts}")
            return agent_output

        self.logger.error(f"Max attempts reached for agent '{agent_id}' with no valid output.")
        raise Exception(f"Failed to get valid response from {agent_id}. We recommend checking the agent's input/output logs.")


# From config_structs/cog_config_structs.py
class CogAgentDef:
    """Definition of an agent within a cog."""
    id: str
    template_file: Optional[str] = None
    type: Optional[str] = None

# From config_structs/cog_config_structs.py
class CogMemoryDef:
    """Definition of a memory node within a cog."""
    id: str
    type: Optional[str] = None  # Full class path for memory type
    collection_id: Optional[str] = None
    query_before: Union[str, List[str], None] = None
    query_keys: List[str] = field(default_factory=list)
    update_after: Union[str, List[str], None] = None
    update_keys: List[str] = field(default_factory=list)

# From config_structs/cog_config_structs.py
class CogFlowTransition:
    """A single flow transition definition."""
    # Can be a string (direct transition), dict (decision), or special end marker
    type: str  # "direct", "decision", "end"
    next_agent: Optional[str] = None  # For direct transitions
    decision_key: Optional[str] = None  # For decision transitions
    decision_map: Dict[str, str] = field(default_factory=dict)  # For decision transitions
    fallback: Optional[str] = None
    max_visits: Optional[int] = None
    end: bool = False

# From config_structs/cog_config_structs.py
class CogFlow:
    """Flow definition for a cog."""
    start: str
    transitions: Dict[str, CogFlowTransition]

# From config_structs/cog_config_structs.py
class CogDefinition:
    """The core cog definition structure."""
    name: str
    description: Optional[str] = None
    persona: Optional[str] = None  # Persona override for the cog
    trail_logging: bool = True
    agents: List[CogAgentDef] = field(default_factory=list)
    memory: List[CogMemoryDef] = field(default_factory=list)
    flow: Optional[CogFlow] = None
    chat_memory_enabled: Optional[bool] = None
    chat_history_max_results: Optional[int] = None

# From config_structs/cog_config_structs.py
class CogConfig:
    """
    Structured configuration object for cogs.
    
    NOTE: This object should not be mutated in place. For hot-reload support,
    replace the entire object with a new one from ConfigManager.build_cog_config().
    """
    cog: CogDefinition
    # Support for additional top-level fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)


# From config_structs/agent_config_structs.py
class PersonaSettings:
    """Persona configuration from system settings."""
    enabled: bool = True
    name: str = "default_assistant"
    static_char_cap: int = 8000

# From config_structs/agent_config_structs.py
class DebugSettings:
    """Debug configuration from system settings."""
    mode: bool = False
    save_memory: bool = False
    simulated_response: str = "Text designed to simulate an LLM response for debugging purposes without invoking the model."

# From config_structs/agent_config_structs.py
class LoggingSettings:
    """Logging configuration from system settings."""
    enabled: bool = True
    console_level: str = "warning"
    folder: str = "./logs"
    files: Dict[str, str] = field(default_factory=dict)

# From config_structs/agent_config_structs.py
class MiscSettings:
    """Miscellaneous system settings."""
    on_the_fly: bool = True

# From config_structs/agent_config_structs.py
class PathSettings:
    """System file path settings."""
    files: str = "./files"

# From config_structs/agent_config_structs.py
class SystemSettings:
    """System settings structure from settings/system.yaml."""
    persona: PersonaSettings
    debug: DebugSettings
    logging: LoggingSettings
    misc: MiscSettings
    paths: PathSettings

# From config_structs/agent_config_structs.py
class Settings:
    """Complete settings structure containing system, models, and storage."""
    system: SystemSettings
    models: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)

# From config_structs/agent_config_structs.py
class AgentConfig:
    """
    Structured configuration object for agents.
    
    NOTE: This object should not be mutated in place. For hot-reload support,
    replace the entire object with a new one from ConfigManager.build_agent_config().
    """
    name: str
    settings: Settings
    model: Any
    params: Dict[str, Any]
    prompts: Dict[str, Any]
    persona: Optional[Dict[str, Any]] = None
    simulated_response: Optional[str] = None
    parse_response_as: Optional[str] = None
    # Support for additional custom fields from YAML
    custom_fields: Dict[str, Any] = field(default_factory=dict)

from pydantic import validator
from reworkd_platform.web.api.agent.analysis import Analysis

# From schemas/agent.py
class ModelSettings(BaseModel):
    model: LLM_Model = Field(default="gpt-3.5-turbo")
    custom_api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=0)
    language: str = Field(default="English")

    @validator("max_tokens")
    def validate_max_tokens(cls, v: float, values: Dict[str, Any]) -> float:
        model = values["model"]
        if v > (max_tokens := LLM_MODEL_MAX_TOKENS[model]):
            raise ValueError(f"Model {model} only supports {max_tokens} tokens")
        return v

# From schemas/agent.py
class AgentRunCreate(BaseModel):
    goal: str
    model_settings: ModelSettings = Field(default=ModelSettings())

# From schemas/agent.py
class AgentRun(AgentRunCreate):
    run_id: str

# From schemas/agent.py
class AgentTaskAnalyze(AgentRun):
    task: str
    tool_names: List[str] = Field(default=[])
    model_settings: ModelSettings = Field(default=ModelSettings())

# From schemas/agent.py
class AgentTaskExecute(AgentRun):
    task: str
    analysis: Analysis

# From schemas/agent.py
class AgentTaskCreate(AgentRun):
    tasks: List[str] = Field(default=[])
    last_task: Optional[str] = Field(default=None)
    result: Optional[str] = Field(default=None)
    completed_tasks: List[str] = Field(default=[])

# From schemas/agent.py
class AgentSummarize(AgentRun):
    results: List[str] = Field(default=[])

# From schemas/agent.py
class AgentChat(AgentRun):
    message: str
    results: List[str] = Field(default=[])

# From schemas/agent.py
class NewTasksResponse(BaseModel):
    run_id: str
    new_tasks: List[str] = Field(alias="newTasks")

# From schemas/agent.py
class RunCount(BaseModel):
    count: int
    first_run: Optional[datetime]
    last_run: Optional[datetime]

# From schemas/agent.py
def validate_max_tokens(cls, v: float, values: Dict[str, Any]) -> float:
        model = values["model"]
        if v > (max_tokens := LLM_MODEL_MAX_TOKENS[model]):
            raise ValueError(f"Model {model} only supports {max_tokens} tokens")
        return v

from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from reworkd_platform.db.crud.base import BaseCrud
from reworkd_platform.db.models.agent import AgentRun
from reworkd_platform.db.models.agent import AgentTask
from reworkd_platform.schemas.agent import Loop_Step
from reworkd_platform.schemas.user import UserBase
from reworkd_platform.settings import settings
from reworkd_platform.web.api.errors import MaxLoopsError
from reworkd_platform.web.api.errors import MultipleSummaryError

# From crud/agent.py
class AgentCRUD(BaseCrud):
    def __init__(self, session: AsyncSession, user: UserBase):
        super().__init__(session)
        self.user = user

    async def create_run(self, goal: str) -> AgentRun:
        return await AgentRun(
            user_id=self.user.id,
            goal=goal,
        ).save(self.session)

    async def create_task(self, run_id: str, type_: Loop_Step) -> AgentTask:
        await self.validate_task_count(run_id, type_)
        return await AgentTask(
            run_id=run_id,
            type_=type_,
        ).save(self.session)

    async def validate_task_count(self, run_id: str, type_: str) -> None:
        if not await AgentRun.get(self.session, run_id):
            raise HTTPException(404, f"Run {run_id} not found")

        query = select(func.count(AgentTask.id)).where(
            and_(
                AgentTask.run_id == run_id,
                AgentTask.type_ == type_,
            )
        )

        task_count = (await self.session.execute(query)).scalar_one()
        max_ = settings.max_loops

        if task_count >= max_:
            raise MaxLoopsError(
                StopIteration(),
                f"Max loops of {max_} exceeded, shutting down.",
                429,
                should_log=False,
            )

        if type_ == "summarize" and task_count > 1:
            raise MultipleSummaryError(
                StopIteration(),
                "Multiple summary tasks are not allowed",
                429,
            )

from sqlalchemy import DateTime
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import mapped_column
from reworkd_platform.db.base import Base

# From models/agent.py
class AgentTask(Base):
    __tablename__ = "agent_task"

    run_id = mapped_column(String, nullable=False)
    type_ = mapped_column(String, nullable=False, name="type")
    create_date = mapped_column(
        DateTime, name="create_date", server_default=func.now(), nullable=False
    )

from reworkd_platform.web.api.memory.memory import AgentMemory

# From memory/null.py
class NullAgentMemory(AgentMemory):
    """
    NullObjectPattern for AgentMemory
    Used when database connections cannot be established
    """

    def __enter__(self) -> AgentMemory:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass

    def add_tasks(self, tasks: List[str]) -> List[str]:
        return []

    def get_similar_tasks(self, query: str, score_threshold: float = 0) -> List[str]:
        return []

    def reset_class(self) -> None:
        pass

# From memory/null.py
def add_tasks(self, tasks: List[str]) -> List[str]:
        return []

# From memory/null.py
def get_similar_tasks(self, query: str, score_threshold: float = 0) -> List[str]:
        return []

# From memory/null.py
def reset_class(self) -> None:
        pass

from fastapi import Body
from fastapi import Depends
from reworkd_platform.db.crud.agent import AgentCRUD
from reworkd_platform.db.dependencies import get_db_session
from reworkd_platform.schemas.agent import AgentChat
from reworkd_platform.schemas.agent import AgentRun
from reworkd_platform.schemas.agent import AgentRunCreate
from reworkd_platform.schemas.agent import AgentSummarize
from reworkd_platform.schemas.agent import AgentTaskAnalyze
from reworkd_platform.schemas.agent import AgentTaskCreate
from reworkd_platform.schemas.agent import AgentTaskExecute
from reworkd_platform.web.api.dependencies import get_current_user

# From agent/dependancies.py
def agent_crud(
    user: UserBase = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentCRUD:
    return AgentCRUD(session, user)

from reworkd_platform.web.api.agent.agent_service.agent_service import AgentService
from reworkd_platform.web.api.agent.agent_service.agent_service import Analysis
from reworkd_platform.web.api.agent.stream_mock import stream_string

# From agent_service/mock_agent_service.py
class MockAgentService(AgentService):
    async def start_goal_agent(self, **kwargs: Any) -> List[str]:
        time.sleep(1)
        return ["Task X", "Task Y", "Task Z"]

    async def create_tasks_agent(self, **kwargs: Any) -> List[str]:
        time.sleep(1)
        return ["Some random task that doesn't exist"]

    async def analyze_task_agent(self, **kwargs: Any) -> Analysis:
        time.sleep(1.5)
        return Analysis(
            action="reason",
            arg="Mock analysis",
            reasoning="Mock to avoid wasting money calling the OpenAI API.",
        )

    async def execute_task_agent(self, **kwargs: Any) -> FastAPIStreamingResponse:
        time.sleep(0.5)
        return stream_string(
            """ This is going to be a longer task result such that
        We make the stream of this string take time and feel long. The reality is... this is a mock! 
        
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
        when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
        It has survived not only five centuries, but also the leap into electronic typesetting, remaining unchanged.
        """
            + kwargs.get("task", "task"),
            True,
        )

    async def summarize_task_agent(
        self,
        *,
        goal: str,
        results: List[str],
    ) -> FastAPIStreamingResponse:
        time.sleep(0.5)
        return stream_string(
            """ This is going to be a longer task result such that
        We make the stream of this string take time and feel long. The reality is... this is a mock! 
        
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
        when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
        It has survived not only five centuries, but also the leap into electronic typesetting, remaining unchanged.
        """,
            True,
        )

    async def chat(
        self,
        *,
        message: str,
        results: List[str],
    ) -> FastAPIStreamingResponse:
        time.sleep(0.5)
        return stream_string(
            "What do you want dude?",
            True,
        )


# From agent_service/agent_service.py
class AgentService(Protocol):
    async def start_goal_agent(self, *, goal: str) -> List[str]:
        pass

    async def analyze_task_agent(
        self, *, goal: str, task: str, tool_names: List[str]
    ) -> Analysis:
        pass

    async def execute_task_agent(
        self,
        *,
        goal: str,
        task: str,
        analysis: Analysis,
    ) -> FastAPIStreamingResponse:
        pass

    async def create_tasks_agent(
        self,
        *,
        goal: str,
        tasks: List[str],
        last_task: str,
        result: str,
        completed_tasks: Optional[List[str]] = None,
    ) -> List[str]:
        pass

    async def summarize_task_agent(
        self,
        *,
        goal: str,
        results: List[str],
    ) -> FastAPIStreamingResponse:
        pass

    async def chat(
        self,
        *,
        message: str,
        results: List[str],
    ) -> FastAPIStreamingResponse:
        pass

from lanarky.responses import StreamingResponse
from langchain import LLMChain
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from loguru import logger
from pydantic import ValidationError
from reworkd_platform.db.crud.oauth import OAuthCrud
from reworkd_platform.schemas.agent import ModelSettings
from reworkd_platform.services.tokenizer.token_service import TokenService
from reworkd_platform.web.api.agent.analysis import AnalysisArguments
from reworkd_platform.web.api.agent.helpers import call_model_with_handling
from reworkd_platform.web.api.agent.helpers import openai_error_handler
from reworkd_platform.web.api.agent.helpers import parse_with_handling
from reworkd_platform.web.api.agent.model_factory import WrappedChatOpenAI
from reworkd_platform.web.api.agent.prompts import analyze_task_prompt
from reworkd_platform.web.api.agent.prompts import chat_prompt
from reworkd_platform.web.api.agent.prompts import create_tasks_prompt
from reworkd_platform.web.api.agent.prompts import start_goal_prompt
from reworkd_platform.web.api.agent.task_output_parser import TaskOutputParser
from reworkd_platform.web.api.agent.tools.open_ai_function import get_tool_function
from reworkd_platform.web.api.agent.tools.tools import get_default_tool
from reworkd_platform.web.api.agent.tools.tools import get_tool_from_name
from reworkd_platform.web.api.agent.tools.tools import get_tool_name
from reworkd_platform.web.api.agent.tools.tools import get_user_tools
from reworkd_platform.web.api.agent.tools.utils import summarize
from reworkd_platform.web.api.errors import OpenAIError

# From agent_service/open_ai_agent_service.py
class OpenAIAgentService(AgentService):
    def __init__(
        self,
        model: WrappedChatOpenAI,
        settings: ModelSettings,
        token_service: TokenService,
        callbacks: Optional[List[AsyncCallbackHandler]],
        user: UserBase,
        oauth_crud: OAuthCrud,
    ):
        self.model = model
        self.settings = settings
        self.token_service = token_service
        self.callbacks = callbacks
        self.user = user
        self.oauth_crud = oauth_crud

    async def start_goal_agent(self, *, goal: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate(prompt=start_goal_prompt)]
        )

        self.token_service.calculate_max_tokens(
            self.model,
            prompt.format_prompt(
                goal=goal,
                language=self.settings.language,
            ).to_string(),
        )

        completion = await call_model_with_handling(
            self.model,
            ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=start_goal_prompt)]
            ),
            {"goal": goal, "language": self.settings.language},
            settings=self.settings,
            callbacks=self.callbacks,
        )

        task_output_parser = TaskOutputParser(completed_tasks=[])
        tasks = parse_with_handling(task_output_parser, completion)

        return tasks

    async def analyze_task_agent(
        self, *, goal: str, task: str, tool_names: List[str]
    ) -> Analysis:
        user_tools = await get_user_tools(tool_names, self.user, self.oauth_crud)
        functions = list(map(get_tool_function, user_tools))
        prompt = analyze_task_prompt.format_prompt(
            goal=goal,
            task=task,
            language=self.settings.language,
        )

        self.token_service.calculate_max_tokens(
            self.model,
            prompt.to_string(),
            str(functions),
        )

        message = await openai_error_handler(
            func=self.model.apredict_messages,
            messages=prompt.to_messages(),
            functions=functions,
            settings=self.settings,
            callbacks=self.callbacks,
        )

        function_call = message.additional_kwargs.get("function_call", {})
        completion = function_call.get("arguments", "")

        try:
            pydantic_parser = PydanticOutputParser(pydantic_object=AnalysisArguments)
            analysis_arguments = parse_with_handling(pydantic_parser, completion)
            return Analysis(
                action=function_call.get("name", get_tool_name(get_default_tool())),
                **analysis_arguments.dict(),
            )
        except (OpenAIError, ValidationError):
            return Analysis.get_default_analysis(task)

    async def execute_task_agent(
        self,
        *,
        goal: str,
        task: str,
        analysis: Analysis,
    ) -> StreamingResponse:
        # TODO: More mature way of calculating max_tokens
        if self.model.max_tokens > 3000:
            self.model.max_tokens = max(self.model.max_tokens - 1000, 3000)

        tool_class = get_tool_from_name(analysis.action)
        return await tool_class(self.model, self.settings.language).call(
            goal,
            task,
            analysis.arg,
            self.user,
            self.oauth_crud,
        )

    async def create_tasks_agent(
        self,
        *,
        goal: str,
        tasks: List[str],
        last_task: str,
        result: str,
        completed_tasks: Optional[List[str]] = None,
    ) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate(prompt=create_tasks_prompt)]
        )

        args = {
            "goal": goal,
            "language": self.settings.language,
            "tasks": "\n".join(tasks),
            "lastTask": last_task,
            "result": result,
        }

        self.token_service.calculate_max_tokens(
            self.model, prompt.format_prompt(**args).to_string()
        )

        completion = await call_model_with_handling(
            self.model, prompt, args, settings=self.settings, callbacks=self.callbacks
        )

        previous_tasks = (completed_tasks or []) + tasks
        return [completion] if completion not in previous_tasks else []

    async def summarize_task_agent(
        self,
        *,
        goal: str,
        results: List[str],
    ) -> FastAPIStreamingResponse:
        self.model.model_name = "gpt-3.5-turbo-16k"
        self.model.max_tokens = 8000  # Total tokens = prompt tokens + completion tokens

        snippet_max_tokens = 7000  # Leave room for the rest of the prompt
        text_tokens = self.token_service.tokenize("".join(results))
        text = self.token_service.detokenize(text_tokens[0:snippet_max_tokens])
        logger.info(f"Summarizing text: {text}")

        return summarize(
            model=self.model,
            language=self.settings.language,
            goal=goal,
            text=text,
        )

    async def chat(
        self,
        *,
        message: str,
        results: List[str],
    ) -> FastAPIStreamingResponse:
        self.model.model_name = "gpt-3.5-turbo-16k"
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=chat_prompt),
                *[HumanMessage(content=result) for result in results],
                HumanMessage(content=message),
            ]
        )

        self.token_service.calculate_max_tokens(
            self.model,
            prompt.format_prompt(
                language=self.settings.language,
            ).to_string(),
        )

        chain = LLMChain(llm=self.model, prompt=prompt)

        return StreamingResponse.from_chain(
            chain,
            {"language": self.settings.language},
            media_type="text/event-stream",
        )

from reworkd_platform.schemas.agent import LLM_Model
from reworkd_platform.services.tokenizer.dependencies import get_token_service
from reworkd_platform.web.api.agent.agent_service.mock_agent_service import MockAgentService
from reworkd_platform.web.api.agent.agent_service.open_ai_agent_service import OpenAIAgentService
from reworkd_platform.web.api.agent.model_factory import create_model

# From agent_service/agent_service_provider.py
def get_agent_service(
    validator: Callable[..., Coroutine[Any, Any, AgentRun]],
    streaming: bool = False,
    llm_model: Optional[LLM_Model] = None,
) -> Callable[..., AgentService]:
    def func(
        run: AgentRun = Depends(validator),
        user: UserBase = Depends(get_current_user),
        token_service: TokenService = Depends(get_token_service),
        oauth_crud: OAuthCrud = Depends(OAuthCrud.inject),
    ) -> AgentService:
        if settings.ff_mock_mode_enabled:
            return MockAgentService()

        model = create_model(
            settings,
            run.model_settings,
            user,
            streaming=streaming,
            force_model=llm_model,
        )

        return OpenAIAgentService(
            model,
            run.model_settings,
            token_service,
            callbacks=None,
            user=user,
            oauth_crud=oauth_crud,
        )

    return func

# From agent_service/agent_service_provider.py
def func(
        run: AgentRun = Depends(validator),
        user: UserBase = Depends(get_current_user),
        token_service: TokenService = Depends(get_token_service),
        oauth_crud: OAuthCrud = Depends(OAuthCrud.inject),
    ) -> AgentService:
        if settings.ff_mock_mode_enabled:
            return MockAgentService()

        model = create_model(
            settings,
            run.model_settings,
            user,
            streaming=streaming,
            force_model=llm_model,
        )

        return OpenAIAgentService(
            model,
            run.model_settings,
            token_service,
            callbacks=None,
            user=user,
            oauth_crud=oauth_crud,
        )

from functionz.core.framework import func

# From drafts/react_agent.py
def react_agent(input_text) -> str:
    def map_python_type_to_json(python_type: str) -> dict:
        type_mapping = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {"type": "string"}},
            "dict": {"type": "object"},
            "Any": {"type": "string"}
        }
        return type_mapping.get(python_type, {"type": "string"})

    try:
        # Enable verbose logging for LiteLLM
        litellm.set_verbose = True

        # Get available functions using get_all_functions_wrapper
        all_functions = get_all_functions_wrapper()

        # Extract function names from the structured data
        available_function_names = [func_info['name'] for func_info in all_functions]

        # Fetch available functions from the database
        tools = []
        for func_name in available_function_names:
            # Retrieve function details using get_function_wrapper
            function_data = get_function_wrapper(func_name)
            if function_data:
                # Construct the tool definition for LiteLLM
                tool = {
                    "type": "function",
                    "function": {
                        "name": function_data['name'],
                        "description": function_data['metadata'].get('description', ''),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        },
                    },
                }

                # Map input_parameters to the tool's parameters
                for param in function_data.get('input_parameters', []):
                    json_schema = map_python_type_to_json(param['type'])
                    tool['function']['parameters']['properties'][param['name']] = {
                        **json_schema,
                        "description": param.get('description', '')
                    }
                    if param.get('required', False):
                        tool['function']['parameters']['required'].append(param['name'])

                tools.append(tool)
            else:
                raise ValueError(f"Function '{func_name}' not found in the database.")

        # Initialize function call history
        function_call_history = []

        # Initialize chat context with system message
        system_prompt = (
            "You are an AI assistant that uses a chain-of-thought reasoning process to solve tasks. "
            "Let's think step by step to solve the following problem. "
            "You have access to the following functions which you can use to complete the task. "
            "Explain your reasoning in detail, including any functions you use and their outputs. "
            "At the end of your reasoning, provide the final answer after 'Answer:'. "
            "Before finalizing, review your reasoning for any errors or inconsistencies. "
            "Avoid repeating function calls with the same arguments you've already tried. "
            "Here is the history of function calls you have made so far: {{function_call_history}}"
        )

        chat_context = [
            {"role": "system", "content": system_prompt.replace("{{function_call_history}}", "None")},
            {"role": "user", "content": input_text},
        ]

        # Initialize loop parameters
        max_iterations = 5
        iteration = 0

        full_reasoning_path = ""

        while iteration < max_iterations:
            iteration += 1

            # Update the system prompt with the current function call history
            if function_call_history:
                history_str = "\n".join([
                    f"- {call['function_name']} with arguments {call['arguments']} produced output: {call['output']}"
                    for call in function_call_history
                ])
            else:
                history_str = "None"

            chat_context[0]['content'] = system_prompt.replace("{{function_call_history}}", history_str)

            # Call LiteLLM's completion API with the chat context and tools
            response = litellm.completion(
                model="gpt-4-turbo",
                messages=chat_context,
                tools=tools,
                tool_choice="auto",
                max_tokens=1500,
                temperature=0.7
            )

            # Extract the message from the response
            response_message = response['choices'][0]['message']

            # Append the assistant's message to the chat context and full reasoning path
            chat_context.append(response_message)
            full_reasoning_path += f"\nIteration {iteration}:\n{response_message['content']}\n"

            # Check if the assistant wants to call any functions
            tool_calls = response_message.get('tool_calls', [])

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])
                    tool_call_id = tool_call['id']

                    # Check if this function call with these arguments has already been made
                    if any(
                        call['function_name'] == function_name and call['arguments'] == function_args
                        for call in function_call_history
                    ):
                        function_response = f"Function '{function_name}' with arguments {function_args} has already been called. Please try a different approach."
                    else:
                        # Execute the function using execute_function_wrapper
                        try:
                            function_output = execute_function_wrapper(function_name, **function_args)
                            function_call_history.append({
                                'function_name': function_name,
                                'arguments': function_args,
                                'output': function_output
                            })
                            function_response = f"Function '{function_name}' executed successfully with output: {function_output}"
                        except Exception as e:
                            function_response = f"Error executing function '{function_name}': {str(e)}"

                    # Ensure function_response is a string
                    if not isinstance(function_response, str):
                        function_response = json.dumps(function_response)

                    # Append the function response to the chat context and full reasoning path
                    chat_context.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                    full_reasoning_path += f"Function Call: {function_name}\nArguments: {function_args}\nOutput: {function_response}\n"

                # Continue the loop to allow the assistant to process the function outputs
                continue
            else:
                # No function calls, assume task is complete
                break

        # Extract the final answer from the last assistant message
        final_answer = response_message['content'].split('Answer:')[-1].strip() if 'Answer:' in response_message['content'] else response_message['content']

        # Compile the full response including reasoning steps and function call history
        if function_call_history:
            function_calls_str = "\n".join([
                f"Function '{call['function_name']}' called with arguments {call['arguments']}, produced output: {call['output']}"
                for call in function_call_history
            ])
        else:
            function_calls_str = "No functions were called."

        full_response = (
            f"Full Reasoning Path:\n{full_reasoning_path}\n\n"
            f"Functions Used:\n{function_calls_str}\n\n"
            f"Final Answer:\n{final_answer}"
        )

        return full_response

    except Exception as e:
        return f"An error occurred: {str(e)}\n\nFull reasoning path so far:\n{full_reasoning_path}"

# From drafts/react_agent.py
def map_python_type_to_json(python_type: str) -> dict:
        type_mapping = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {"type": "string"}},
            "dict": {"type": "object"},
            "Any": {"type": "string"}
        }
        return type_mapping.get(python_type, {"type": "string"})

from python.helpers import extract_tools
from python.helpers import rate_limiter
from python.helpers import files
from python.helpers import errors
from python.helpers.print_style import PrintStyle
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from python.tools.unknown import Unknown
from python.helpers.tool import Tool
from python.tools import memory_tool

# From agent-zero/agent.py
def message_loop(self, msg: str):
        try:
            printer = PrintStyle(italic=True, font_color="#b3ffd9", padding=False)    
            user_message = files.read_file("./prompts/fw.user_message.md", message=msg)
            self.append_message(user_message, human=True) # Append the user's input to the history                        
            memories = self.fetch_memories(True)
                
            while True: # let the agent iterate on his thoughts until he stops by using a tool
                Agent.streaming_agent = self #mark self as current streamer
                agent_response = ""
                self.intervention_status = False # reset interventon status

                try:

                    system = self.system_prompt + "\n\n" + self.tools_prompt
                    memories = self.fetch_memories()
                    if memories: system+= "\n\n"+memories

                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(content=system),
                        MessagesPlaceholder(variable_name="messages") ])
                    
                    inputs = {"messages": self.history}
                    chain = prompt | self.config.chat_model

                    formatted_inputs = prompt.format(messages=self.history)
                    tokens = int(len(formatted_inputs)/4)     
                    self.rate_limiter.limit_call_and_input(tokens)
                    
                    # output that the agent is starting
                    PrintStyle(bold=True, font_color="green", padding=True, background_color="white").print(f"{self.agent_name}: Starting a message:")
                                            
                    for chunk in chain.stream(inputs):
                        if self.handle_intervention(agent_response): break # wait for intervention and handle it, if paused

                        if isinstance(chunk, str): content = chunk
                        elif hasattr(chunk, "content"): content = str(chunk.content)
                        else: content = str(chunk)
                        
                        if content:
                            printer.stream(content) # output the agent response stream                
                            agent_response += content # concatenate stream into the response

                    self.rate_limiter.set_output_tokens(int(len(agent_response)/4))
                    
                    if not self.handle_intervention(agent_response):
                        if self.last_message == agent_response: #if assistant_response is the same as last message in history, let him know
                            self.append_message(agent_response) # Append the assistant's response to the history
                            warning_msg = files.read_file("./prompts/fw.msg_repeat.md")
                            self.append_message(warning_msg, human=True) # Append warning message to the history
                            PrintStyle(font_color="orange", padding=True).print(warning_msg)

                        else: #otherwise proceed with tool
                            self.append_message(agent_response) # Append the assistant's response to the history
                            tools_result = self.process_tools(agent_response) # process tools requested in agent message
                            if tools_result: return tools_result #break the execution if the task is done

                # Forward errors to the LLM, maybe he can fix them
                except Exception as e:
                    error_message = errors.format_error(e)
                    msg_response = files.read_file("./prompts/fw.error.md", error=error_message) # error message template
                    self.append_message(msg_response, human=True)
                    PrintStyle(font_color="red", padding=True).print(msg_response)
                    
        finally:
            Agent.streaming_agent = None

# From agent-zero/agent.py
def get_data(self, field:str):
        return self.data.get(field, None)

# From agent-zero/agent.py
def set_data(self, field:str, value):
        self.data[field] = value

# From agent-zero/agent.py
def append_message(self, msg: str, human: bool = False):
        message_type = "human" if human else "ai"
        if self.history and self.history[-1].type == message_type:
            self.history[-1].content += "\n\n" + msg
        else:
            new_message = HumanMessage(content=msg) if human else AIMessage(content=msg)
            self.history.append(new_message)
            self.cleanup_history(self.config.msgs_keep_max, self.config.msgs_keep_start, self.config.msgs_keep_end)
        if message_type=="ai":
            self.last_message = msg

# From agent-zero/agent.py
def concat_messages(self,messages):
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

# From agent-zero/agent.py
def send_adhoc_message(self, system: str, msg: str, output_label:str):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system),
            HumanMessage(content=msg)])

        chain = prompt | self.config.utility_model
        response = ""
        printer = None

        if output_label:
            PrintStyle(bold=True, font_color="orange", padding=True, background_color="white").print(f"{self.agent_name}: {output_label}:")
            printer = PrintStyle(italic=True, font_color="orange", padding=False)                

        formatted_inputs = prompt.format()
        tokens = int(len(formatted_inputs)/4)     
        self.rate_limiter.limit_call_and_input(tokens)
    
        for chunk in chain.stream({}):
            if self.handle_intervention(): break # wait for intervention and handle it, if paused

            if isinstance(chunk, str): content = chunk
            elif hasattr(chunk, "content"): content = str(chunk.content)
            else: content = str(chunk)

            if printer: printer.stream(content)
            response+=content

        self.rate_limiter.set_output_tokens(int(len(response)/4))

        return response

# From agent-zero/agent.py
def get_last_message(self):
        if self.history:
            return self.history[-1]

# From agent-zero/agent.py
def replace_middle_messages(self,middle_messages):
        cleanup_prompt = files.read_file("./prompts/fw.msg_cleanup.md")
        summary = self.send_adhoc_message(system=cleanup_prompt,msg=self.concat_messages(middle_messages), output_label="Mid messages cleanup summary")
        new_human_message = HumanMessage(content=summary)
        return [new_human_message]

# From agent-zero/agent.py
def cleanup_history(self, max:int, keep_start:int, keep_end:int):
        if len(self.history) <= max:
            return self.history

        first_x = self.history[:keep_start]
        last_y = self.history[-keep_end:]

        # Identify the middle part
        middle_part = self.history[keep_start:-keep_end]

        # Ensure the first message in the middle is "human", if not, move one message back
        if middle_part and middle_part[0].type != "human":
            if len(first_x) > 0:
                middle_part.insert(0, first_x.pop())

        # Ensure the middle part has an odd number of messages
        if len(middle_part) % 2 == 0:
            middle_part = middle_part[:-1]

        # Replace the middle part using the replacement function
        new_middle_part = self.replace_middle_messages(middle_part)

        self.history = first_x + new_middle_part + last_y

        return self.history

# From agent-zero/agent.py
def handle_intervention(self, progress:str="") -> bool:
        while self.paused: time.sleep(0.1) # wait if paused
        if self.intervention_message and not self.intervention_status: # if there is an intervention message, but not yet processed
            if progress.strip(): self.append_message(progress) # append the response generated so far
            user_msg = files.read_file("./prompts/fw.intervention.md", user_message=self.intervention_message) # format the user intervention template
            self.append_message(user_msg,human=True) # append the intervention message
            self.intervention_message = "" # reset the intervention message
            self.intervention_status = True
        return self.intervention_status

# From agent-zero/agent.py
def process_tools(self, msg: str):
        # search for tool usage requests in agent message
        tool_request = extract_tools.json_parse_dirty(msg)

        if tool_request is not None:
            tool_name = tool_request.get("tool_name", "")
            tool_args = tool_request.get("tool_args", {})

            tool = self.get_tool(
                        tool_name,
                        tool_args,
                        msg)
                
            if self.handle_intervention(): return # wait if paused and handle intervention message if needed
            tool.before_execution(**tool_args)
            if self.handle_intervention(): return # wait if paused and handle intervention message if needed
            response = tool.execute(**tool_args)
            if self.handle_intervention(): return # wait if paused and handle intervention message if needed
            tool.after_execution(response)
            if self.handle_intervention(): return # wait if paused and handle intervention message if needed
            if response.break_loop: return response.message
        else:
            msg = files.read_file("prompts/fw.msg_misformat.md")
            self.append_message(msg, human=True)
            PrintStyle(font_color="red", padding=True).print(msg)

# From agent-zero/agent.py
def get_tool(self, name: str, args: dict, message: str, **kwargs):
        from python.tools.unknown import Unknown 
        from python.helpers.tool import Tool
        
        tool_class = Unknown
        if files.exists("python/tools",f"{name}.py"): 
            module = importlib.import_module("python.tools." + name)  # Import the module
            class_list = inspect.getmembers(module, inspect.isclass)  # Get all functions in the module

            for cls in class_list:
                if cls[1] is not Tool and issubclass(cls[1], Tool):
                    tool_class = cls[1]
                    break

        return tool_class(agent=self, name=name, args=args, message=message, **kwargs)

# From agent-zero/agent.py
def fetch_memories(self,reset_skip=False):
        if self.config.auto_memory_count<=0: return ""
        if reset_skip: self.memory_skip_counter = 0

        if self.memory_skip_counter > 0:
            self.memory_skip_counter-=1
            return ""
        else:
            self.memory_skip_counter = self.config.auto_memory_skip
            from python.tools import memory_tool
            messages = self.concat_messages(self.history)
            memories = memory_tool.search(self,messages)
            input = {
                "conversation_history" : messages,
                "raw_memories": memories
            }
            cleanup_prompt = files.read_file("./prompts/msg.memory_cleanup.md").replace("{", "{{")       
            clean_memories = self.send_adhoc_message(cleanup_prompt,json.dumps(input), output_label="Memory injection")
            return clean_memories

# From agent-zero/agent.py
def call_extension(self, name: str, **kwargs) -> Any:
        pass

