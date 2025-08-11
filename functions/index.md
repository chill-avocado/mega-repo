# Repository Index

This document provides an index of all repositories categorized by functionality.

## Categories

### Agent Frameworks

#### Repositories

- [AgentForge](../repos/AgentForge)
- [AgentGPT](../repos/AgentGPT)
- [AgentK](../repos/AgentK)
- [Auto-GPT-MetaTrader-Plugin](../repos/Auto-GPT-MetaTrader-Plugin)
- [Free-Auto-GPT](../repos/Free-Auto-GPT)
- [agent-zero](../repos/agent-zero)
- [autogen](../repos/autogen)
- [babyagi](../repos/babyagi)
- [Teenage-AGI](../repos/Teenage-AGI)
- [openhands](../repos/openhands)

#### Common Components

- [agent_base](./common/agent_base.py)
- [memory_system](./common/memory_system.py)

### User Interfaces

#### Repositories

- [open-webui](../repos/open-webui)
- [web-ui](../repos/web-ui)
- [draw-a-ui](../repos/draw-a-ui)

#### Common Components

- [chat_interface](./common/chat_interface.py)

### Os Interaction

#### Repositories

- [MacOS-Agent](../repos/MacOS-Agent)
- [self-operating-computer](../repos/self-operating-computer)
- [mcp-remote-macos-use](../repos/mcp-remote-macos-use)

#### Common Components

- [keyboard_control](./common/keyboard_control.py)
- [screen_capture](./common/screen_capture.py)

### Browser Automation

#### Repositories

- [browser-use](../repos/browser-use)

#### Common Components

- [browser_control](./common/browser_control.py)

### Code Execution

#### Repositories

- [open-interpreter](../repos/open-interpreter)
- [pyCodeAGI](../repos/pyCodeAGI)
- [SuperCoder](../repos/SuperCoder)
- [automata](../repos/automata)

#### Common Components

- [code_executor](./common/code_executor.py)

### Cognitive Systems

#### Repositories

- [opencog](../repos/opencog)
- [openagi](../repos/openagi)
- [AGI-Samantha](../repos/AGI-Samantha)

#### Common Components

- [knowledge_graph](./common/knowledge_graph.py)

### Evolution Optimization

#### Repositories

- [openevolve](../repos/openevolve)

#### Common Components

- [genetic_algorithm](./common/genetic_algorithm.py)

### Integration

#### Repositories

- [Unification](../repos/Unification)
- [LocalAGI](../repos/LocalAGI)

#### Common Components

- [system_connector](./common/system_connector.py)

### Nlp

#### Repositories

- [senpai](../repos/senpai)
- [html-agility-pack](../repos/html-agility-pack)

#### Common Components

- [text_processor](./common/text_processor.py)

### Documentation

#### Repositories

- [Awesome-AGI](../repos/Awesome-AGI)
- [awesome-agi-cocosci](../repos/awesome-agi-cocosci)

#### Common Components


## Repositories

### AgentForge

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `setup.py`
  - `src/agentforge/apis/mixins/vision_mixin.py`
  - `src/agentforge/modules/actions.py`
  - `src/agentforge/storage/chroma_storage.py`
  - `src/agentforge/storage/persona_memory.py`
  - ... and 19 more files
- **Agent Lifecycle**
  - `src/agentforge/cog.py`
  - `src/agentforge/config_structs/cog_config_structs.py`
  - `src/agentforge/core/agent_registry.py`
  - `src/agentforge/core/config_manager.py`
  - `src/agentforge/storage/scratchpad.py`
  - ... and 11 more files
- **Memory**
  - `src/agentforge/cog.py`
  - `src/agentforge/config_structs/__init__.py`
  - `src/agentforge/config_structs/agent_config_structs.py`
  - `src/agentforge/config_structs/cog_config_structs.py`
  - `src/agentforge/core/agent_runner.py`
  - ... and 28 more files
- **Planning**
  - `src/agentforge/utils/prompt_processor.py`
  - `tests/integration_tests/test_persona_memory_integration.py`
  - `tests/utils/test_parsing_processor_default_fences.py`
- **Multi Agent**
  - `tests/core_tests/test_agent_registry.py`

### AgentGPT

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `next/src/components/Input.tsx`
  - `next/src/components/Label.tsx`
  - `next/src/components/Tooltip.tsx`
  - `next/src/components/console/ChatMessage.tsx`
  - `next/src/components/dialog/HelpDialog.tsx`
  - ... and 43 more files
- **Agent Lifecycle**
  - `next/src/components/drawer/LeftSidebar.tsx`
  - `next/src/components/landing/OpenSource.tsx`
  - `next/src/hooks/useAgent.ts`
  - `next/src/server/api/routers/agentRouter.ts`
  - `next/src/services/agent/agent-api.ts`
  - ... and 14 more files
- **Planning**
  - `next/src/components/NavBar.tsx`
  - `next/src/components/console/ExampleAgents.tsx`
  - `next/src/components/templates/TemplateData.tsx`
  - `platform/reworkd_platform/services/pinecone/pinecone.py`
  - `platform/reworkd_platform/web/api/agent/dependancies.py`
  - ... and 3 more files
- **Memory**
  - `next/src/components/landing/Section.tsx`
  - `next/src/components/templates/TemplateData.tsx`
  - `platform/reworkd_platform/services/pinecone/pinecone.py`
  - `platform/reworkd_platform/tests/agent/test_task_output_parser.py`
  - `platform/reworkd_platform/tests/memory/memory_with_fallback_test.py`
  - ... and 4 more files

### AgentK

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `agents/agent_smith.py`
  - `agents/hermes.py`
  - `agents/software_engineer.py`
  - `agents/tool_maker.py`
  - `agents/web_researcher.py`
  - ... and 18 more files
- **Agent Lifecycle**
  - `agents/agent_smith.py`
  - `agents/hermes.py`
  - `agents/software_engineer.py`
  - `agents/tool_maker.py`
- **Planning**
  - `agents/agent_smith.py`
  - `agents/hermes.py`

### Auto-GPT-MetaTrader-Plugin

Category: Agent Frameworks

#### Functionality

- **Planning**
  - `src/auto_gpt_metatrader/__init__.py`
- **Tool Use**
  - `src/auto_gpt_metatrader/trading.py`

### Free-Auto-GPT

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `AUTOGPT.py`
  - `BABYAGI.py`
  - `Camel.py`
  - `Embedding/HuggingFaceEmbedding.py`
  - `FreeLLM/ChatGPTAPI.py`
  - ... and 9 more files
- **Agent Lifecycle**
  - `AUTOGPT.py`
  - `OtherAgent/csvAgent.py`
  - `OtherAgent/customAgent.py`
  - `OtherAgent/pythonAgent.py`
- **Memory**
  - `AUTOGPT.py`
  - `BABYAGI.py`
  - `FreeLLM/BardChatAPI.py`
  - `FreeLLM/BingChatAPI.py`
  - `FreeLLM/ChatGPTAPI.py`
  - ... and 13 more files
- **Planning**
  - `BABYAGI.py`
  - `hfAgent/agents.py`

### agent-zero

Category: Agent Frameworks

#### Functionality

- **Memory**
  - `agent.py`
  - `main.py`
  - `python/helpers/vdb.py`
  - `python/helpers/vector_db.py`
  - `python/tools/knowledge_tool.py`
  - ... and 3 more files
- **Tool Use**
  - `agent.py`
  - `main.py`
  - `python/helpers/extract_tools.py`
  - `python/helpers/tool.py`
  - `python/tools/call_subordinate.py`
  - ... and 7 more files
- **Agent Lifecycle**
  - `main.py`
  - `python/tools/call_subordinate.py`

### autogen

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Anthropic_Agent_With_Prompt_Caching.cs`
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent_With_Tool.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/CreateAnAgent.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/FunctionCallCodeSnippet.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/MistralAICodeSnippet.cs`
  - ... and 409 more files
- **Multi Agent**
  - `dotnet/src/AutoGen.Anthropic/Middleware/AnthropicMessageConnector.cs`
  - `dotnet/src/AutoGen.AzureAIInference/Middleware/AzureAIInferenceChatRequestMessageConnector.cs`
  - `dotnet/src/AutoGen.Gemini/Middleware/GeminiMessageConnector.cs`
  - `dotnet/src/AutoGen.Ollama/Middlewares/OllamaMessageConnector.cs`
  - `dotnet/src/AutoGen.OpenAI.V1/Middleware/OpenAIChatRequestMessageConnector.cs`
  - ... and 27 more files
- **Memory**
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Anthropic_Agent_With_Prompt_Caching.cs`
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent_With_Tool.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example01_AssistantAgent.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example04_Dynamic_GroupChat_Coding_Task.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example11_Sequential_GroupChat_Example.cs`
  - ... and 160 more files
- **Planning**
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Anthropic_Agent_With_Prompt_Caching.cs`
  - `dotnet/samples/dev-team/DevTeam.Backend/Agents/Developer/DeveloperPrompts.cs`
  - `dotnet/samples/dev-team/DevTeam.Backend/Agents/DeveloperLead/DeveloperLead.cs`
  - `dotnet/samples/dev-team/DevTeam.Backend/Agents/DeveloperLead/DeveloperLeadPrompts.cs`
  - `dotnet/samples/dev-team/DevTeam.Backend/Agents/Hubber.cs`
  - ... and 27 more files
- **Agent Lifecycle**
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent.cs`
  - `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent_With_Tool.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/CreateAnAgent.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/GetStartCodeSnippet.cs`
  - `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/MiddlewareAgentCodeSnippet.cs`
  - ... and 153 more files

### babyagi

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `babyagi/functionz/core/registration.py`
  - `babyagi/functionz/packs/default/default_functions.py`
  - `babyagi/functionz/packs/default/function_calling_chat.py`
  - `babyagi/functionz/packs/drafts/generate_function.py`
  - `babyagi/functionz/packs/drafts/react_agent.py`
  - ... and 2 more files
- **Planning**
  - `babyagi/functionz/packs/drafts/react_agent.py`
- **Memory**
  - `babyagi/dashboard/static/js/function_details.js`
  - `babyagi/functionz/packs/default/function_calling_chat.py`
  - `babyagi/functionz/packs/drafts/react_agent.py`

### Teenage-AGI

Category: Agent Frameworks

#### Functionality

- **Memory**
  - `agent.py`
  - `main.py`
- **Tool Use**
  - `agent.py`
  - `main.py`

### openhands

Category: Agent Frameworks

#### Functionality

- **Tool Use**
  - `build_vscode.py`
  - `evaluation/benchmarks/EDA/game.py`
  - `evaluation/benchmarks/EDA/run_infer.py`
  - `evaluation/benchmarks/agent_bench/helper.py`
  - `evaluation/benchmarks/agent_bench/run_infer.py`
  - ... and 429 more files
- **Agent Lifecycle**
  - `evaluation/benchmarks/agent_bench/run_infer.py`
  - `evaluation/benchmarks/aider_bench/run_infer.py`
  - `evaluation/integration_tests/run_infer.py`
  - `frontend/src/api/open-hands.ts`
  - `frontend/src/api/open-hands.types.ts`
  - ... and 61 more files
- **Memory**
  - `evaluation/benchmarks/EDA/run_infer.py`
  - `evaluation/benchmarks/agent_bench/run_infer.py`
  - `evaluation/benchmarks/aider_bench/run_infer.py`
  - `evaluation/benchmarks/aider_bench/scripts/summarize_results.py`
  - `evaluation/benchmarks/biocoder/run_infer.py`
  - ... and 172 more files
- **Planning**
  - `evaluation/benchmarks/EDA/game.py`
  - `evaluation/benchmarks/discoverybench/eval_utils/eval_w_subhypo_gen.py`
  - `evaluation/benchmarks/gorilla/utils.py`
  - `evaluation/benchmarks/logic_reasoning/logic_inference.py`
  - `evaluation/benchmarks/ml_bench/run_analysis.py`
  - ... and 26 more files
- **Multi Agent**
  - `openhands/agenthub/browsing_agent/browsing_agent.py`
  - `openhands/agenthub/codeact_agent/tools/browser.py`
  - `openhands/agenthub/readonly_agent/tools/glob.py`
  - `openhands/agenthub/readonly_agent/tools/grep.py`
  - `openhands/controller/agent_controller.py`
  - ... and 6 more files

### open-webui

Category: User Interfaces

#### Functionality

- **Chat Interface**
  - `backend/open_webui/config.py`
  - `backend/open_webui/constants.py`
  - `backend/open_webui/env.py`
  - `backend/open_webui/functions.py`
  - `backend/open_webui/internal/migrations/001_initial_schema.py`
  - ... and 88 more files
- **Components**
  - `backend/open_webui/config.py`
  - `backend/open_webui/constants.py`
  - `backend/open_webui/env.py`
  - `backend/open_webui/functions.py`
  - `backend/open_webui/internal/migrations/001_initial_schema.py`
  - ... and 109 more files
- **Visualization**
  - `backend/open_webui/config.py`
  - `backend/open_webui/env.py`
  - `backend/open_webui/main.py`
  - `backend/open_webui/models/models.py`
  - `backend/open_webui/retrieval/web/serply.py`
  - ... and 11 more files
- **Authentication**
  - `backend/open_webui/config.py`
  - `backend/open_webui/constants.py`
  - `backend/open_webui/env.py`
  - `backend/open_webui/internal/migrations/001_initial_schema.py`
  - `backend/open_webui/internal/migrations/006_migrate_timestamps_and_charfields.py`
  - ... and 101 more files
- **Responsive**
  - `backend/open_webui/config.py`
  - `backend/open_webui/retrieval/web/serply.py`
  - `src/lib/stores/index.ts`
  - `src/lib/utils/onedrive-file-picker.ts`
  - `svelte.config.js`

### web-ui

Category: User Interfaces

#### Functionality

- **Components**
  - `src/agent/browser_use/browser_use_agent.py`
  - `src/agent/deep_research/deep_research_agent.py`
  - `src/controller/custom_controller.py`
  - `src/utils/llm_provider.py`
  - `src/utils/mcp_client.py`
  - ... and 10 more files
- **Chat Interface**
  - `src/agent/browser_use/browser_use_agent.py`
  - `src/agent/deep_research/deep_research_agent.py`
  - `src/controller/custom_controller.py`
  - `src/utils/config.py`
  - `src/utils/llm_provider.py`
  - ... and 4 more files
- **Responsive**
  - `tests/test_agents.py`
  - `tests/test_controller.py`
- **Authentication**
  - `tests/test_llm_api.py`
- **Visualization**
  - `src/agent/deep_research/deep_research_agent.py`
  - `src/utils/config.py`
  - `src/utils/llm_provider.py`
  - `src/webui/components/browser_use_agent_tab.py`
  - `src/webui/components/deep_research_agent_tab.py`

### draw-a-ui

Category: User Interfaces

#### Functionality

- **Components**
  - `app/page.tsx`
  - `components/PreviewModal.tsx`
  - `lib/getSvgAsImage.ts`
  - `lib/png.ts`
  - `tailwind.config.ts`
- **Visualization**
  - `lib/getSvgAsImage.ts`
- **Chat Interface**
  - `app/api/toHtml/route.ts`
  - `app/page.tsx`
- **Responsive**
  - `app/layout.tsx`

### MacOS-Agent

Category: Os Interaction

#### Functionality

- **Input Control**
  - `macos_agent_server.py`
- **File System**
  - `macos_agent_server.py`

### self-operating-computer

Category: Os Interaction

#### Functionality

- **Screen Capture**
  - `evaluate.py`
  - `operate/models/apis.py`
  - `operate/models/prompts.py`
  - `operate/utils/ocr.py`
  - `operate/utils/screenshot.py`
- **Input Control**
  - `operate/main.py`
  - `operate/models/prompts.py`
  - `operate/operate.py`
  - `operate/utils/operating_system.py`
- **File System**
  - `operate/utils/ocr.py`

### mcp-remote-macos-use

Category: Os Interaction

#### Functionality

- **File System**
  - `conftest.py`
  - `src/action_handlers.py`
  - `src/mcp_remote_macos_use/__init__.py`
  - `src/mcp_remote_macos_use/server.py`
  - `tests/__init__.py`
  - ... and 3 more files
- **Screen Capture**
  - `src/action_handlers.py`
  - `src/mcp_remote_macos_use/server.py`
  - `src/vnc_client.py`
  - `tests/test_action_handlers.py`
  - `tests/test_vnc_client.py`
- **Input Control**
  - `src/action_handlers.py`
  - `src/mcp_remote_macos_use/server.py`
  - `src/vnc_client.py`
  - `tests/test_action_handlers.py`
  - `tests/test_server.py`
- **System Monitoring**
  - `src/vnc_client.py`
- **App Control**
  - `src/action_handlers.py`

### browser-use

Category: Browser Automation

#### Functionality

- **Navigation**
  - `browser_use/__init__.py`
  - `browser_use/agent/cloud_events.py`
  - `browser_use/agent/message_manager/service.py`
  - `browser_use/agent/prompts.py`
  - `browser_use/agent/service.py`
  - ... and 146 more files
- **Interaction**
  - `browser_use/__init__.py`
  - `browser_use/agent/cloud_events.py`
  - `browser_use/agent/gif.py`
  - `browser_use/agent/message_manager/service.py`
  - `browser_use/agent/message_manager/views.py`
  - ... and 162 more files
- **Headless**
  - `browser_use/agent/cloud_events.py`
  - `browser_use/browser/profile.py`
  - `browser_use/browser/session.py`
  - `browser_use/cli.py`
  - `browser_use/config.py`
  - ... and 70 more files
- **Scraping**
  - `browser_use/agent/message_manager/service.py`
  - `browser_use/agent/prompts.py`
  - `browser_use/agent/service.py`
  - `browser_use/browser/profile.py`
  - `browser_use/controller/service.py`
  - ... and 7 more files
- **Automation**
  - `examples/getting_started/04_multi_step_task.py`
  - `examples/mcp/advanced_server.py`
  - `examples/ui/command_line.py`
  - `examples/ui/streamlit_demo.py`
  - `examples/use-cases/captcha.py`
  - ... and 4 more files

### open-interpreter

Category: Code Execution

#### Functionality

- **Execution**
  - `interpreter/computer_use/loop.py`
  - `interpreter/computer_use/tools/base.py`
  - `interpreter/computer_use/tools/bash.py`
  - `interpreter/computer_use/tools/edit.py`
  - `interpreter/computer_use/tools/run.py`
  - ... and 52 more files
- **Development**
  - `interpreter/computer_use/loop.py`
  - `interpreter/computer_use/tools/bash.py`
  - `interpreter/computer_use/tools/edit.py`
  - `interpreter/computer_use/tools/run.py`
  - `interpreter/core/async_core.py`
  - ... and 50 more files
- **Code Generation**
  - `interpreter/core/computer/terminal/languages/jupyter_language.py`
  - `interpreter/core/llm/llm.py`
  - `interpreter/core/llm/run_function_calling_llm.py`
  - `interpreter/core/llm/run_tool_calling_llm.py`
  - `interpreter/terminal_interface/profiles/defaults/assistant.py`
  - ... and 3 more files
- **Language Support**
  - `interpreter/computer_use/unused_markdown.py`
  - `interpreter/core/computer/computer.py`
  - `interpreter/core/computer/display/display.py`
  - `interpreter/core/computer/mouse/mouse.py`
  - `interpreter/core/computer/skills/skills.py`
  - ... and 41 more files
- **Analysis**
  - `interpreter/terminal_interface/profiles/defaults/the01.py`

### pyCodeAGI

Category: Code Execution

#### Functionality

- **Code Generation**
  - `pycodeagi-gpt4.py`
  - `pycodeagi.py`
- **Execution**
  - `pycodeagi-gpt4.py`
  - `pycodeagi.py`
- **Language Support**
  - `pycodeagi-gpt4.py`
  - `pycodeagi.py`
- **Development**
  - `pycodeagi-gpt4.py`
  - `pycodeagi.py`

### SuperCoder

Category: Code Execution

#### Functionality

- **Execution**
  - `app/models/dtos/asynq_task/create_job_payload.go`
  - `app/repositories/execution.go`
  - `app/services/design_story_review_service.go`
  - `app/services/execution_service.go`
  - `app/services/pull_request_comments_service.go`
  - ... and 29 more files
- **Language Support**
  - `app/gateways/websocket_gateway.go`
  - `app/tasks/create_execution_job_task.go`
  - `app/workflow_executors/step_executors/impl/django_server_step_executor.go`
  - `app/workflow_executors/step_executors/impl/flask_reset_db_step_executor.go`
  - `app/workflow_executors/step_executors/impl/flask_sever_test_executor.go`
  - ... and 9 more files
- **Development**
  - `app/client/git_provider/gitness_git_provider.go`
  - `app/config/ai_developer_execution.go`
  - `app/config/config.go`
  - `app/controllers/pull_request_controller.go`
  - `app/models/dtos/gitness/types.go`
  - ... and 52 more files
- **Analysis**
  - `app/workflow_executors/step_executors/impl/next_js_server_test_executor.go`
  - `gui/src/app/imagePath.tsx`
  - `gui/src/components/HomeComponents/GithubStarModal.tsx`
- **Code Generation**
  - `app/models/execution_step.go`
  - `app/workflow_executors/design_nextjs_workflow_config.go`
  - `app/workflow_executors/python_django_workflow_config.go`
  - `app/workflow_executors/python_flask_workflow_config.go`
  - `app/workflow_executors/step_executors/code_generation_executor.go`
  - ... and 9 more files

### automata

Category: Code Execution

#### Functionality

- **Language Support**
  - `automata/cli/commands.py`
  - `automata/cli/install_indexing.py`
  - `automata/code_parsers/py/py_reader.py`
  - `automata/code_writers/py/__init__.py`
  - `automata/code_writers/py/py_code_writer.py`
  - ... and 45 more files
- **Execution**
  - `automata/agent/openai_agent.py`
  - `automata/cli/commands.py`
  - `automata/cli/install_indexing.py`
  - `automata/cli/options.py`
  - `automata/cli/scripts/run_agent_eval.py`
  - ... and 33 more files
- **Development**
  - `automata/agent/agent.py`
  - `automata/agent/error.py`
  - `automata/agent/openai_agent.py`
  - `automata/cli/env_operations.py`
  - `automata/cli/scripts/run_agent.py`
  - ... and 96 more files
- **Code Generation**
  - `automata/symbol/scip_pb2.py`
  - `automata/tests/unit/agent_eval/conftest.py`
  - `automata/tests/unit/agent_eval/test_agent_eval_actions_results.py`
  - `automata/tests/unit/code_writers/test_py_writer.py`
  - `research/study_agency/study_human_eval/completion_provider.py`
  - ... and 3 more files
- **Analysis**
  - `research/study_agency/study_leetcode/leetcode_constants.py`
  - `research/study_agency/study_leetcode/leetcode_problem_solver.py`

### opencog

Category: Cognitive Systems

#### Functionality

- **Reasoning**
  - `opencog/nlp/chatbot/telegram_bot.py`

### openagi

Category: Cognitive Systems

#### Functionality

- **Learning**
  - `example/job_search.py`
  - `example/youtube_study_plan.py`
  - `src/openagi/llms/base.py`
  - `src/openagi/prompts/task_creator.py`
- **Cognitive Processes**
  - `example/job_post.py`
  - `src/openagi/prompts/task_creator.py`
  - `src/openagi/utils/llmTasks.py`
- **Reasoning**
  - `example/market_research.py`
  - `src/openagi/llms/azure.py`
  - `src/openagi/llms/cohere.py`
  - `src/openagi/llms/gemini.py`
  - `src/openagi/llms/groq.py`
  - ... and 6 more files
- **Knowledge Representation**
  - `src/openagi/memory/base.py`
- **Memory Systems**
  - `src/openagi/memory/base.py`

### AGI-Samantha

Category: Cognitive Systems

#### Functionality

- **Reasoning**
  - `AGI-1.py`
- **Memory Systems**
  - `AGI-1.py`
- **Cognitive Processes**
  - `AGI-1.py`

### openevolve

Category: Evolution Optimization

#### Functionality

- **Evolutionary Algorithms**
  - `examples/attention_optimization/initial_program.py`
  - `examples/attention_optimization/scripts/mlir_lowering_pipeline.py`
  - `examples/attention_optimization/scripts/mlir_syntax_test.py`
  - `examples/attention_optimization/tests/test_evaluator.py`
  - `examples/attention_optimization/tests/test_results.py`
  - ... and 24 more files
- **Benchmarking**
  - `examples/attention_optimization/evaluator.py`
  - `examples/attention_optimization/legacy/prev_sim__works_evaluator.py`
  - `examples/attention_optimization/scripts/debug_real_execution.py`
  - `examples/attention_optimization/tests/test_evaluator.py`
  - `examples/circle_packing/evaluator.py`
  - ... and 34 more files
- **Adaptive Learning**
  - `examples/circle_packing_with_artifacts/evaluator.py`
  - `examples/mlx_metal_kernel_opt/qwen3_benchmark_suite.py`
  - `examples/online_judge_programming/submit.py`
  - `examples/rust_adaptive_sort/evaluator.py`
  - `examples/rust_adaptive_sort/initial_program.rs`
  - ... and 8 more files
- **Optimization**
  - `examples/attention_optimization/evaluator.py`
  - `examples/attention_optimization/initial_program.py`
  - `examples/attention_optimization/legacy/prev_sim__works_evaluator.py`
  - `examples/attention_optimization/tests/test_evaluator.py`
  - `examples/circle_packing/best_program.py`
  - ... and 9 more files
- **Multi Objective**
  - `examples/attention_optimization/initial_program.py`
  - `examples/signal_processing/evaluator.py`

### Unification

Category: Integration

#### Functionality

- **Api Standardization**
  - `features/mcp-remote-macos-use-main/src/vnc_client.py`
  - `features/mcp-remote-macos-use-main/tests/test_vnc_client.py`
- **Orchestration**
  - `features/AgentK-master/agents/hermes.py`

### LocalAGI

Category: Integration

#### Functionality

- **System Integration**
  - `services/actions/githubprreviewer_test.go`
- **Compatibility**
  - `pkg/stdio/client.go`

### senpai

Category: Nlp

#### Functionality

- **Entity Recognition**
  - `senpai/agent.py`
  - `senpai/memory.py`

### html-agility-pack

Category: Nlp

#### Functionality

- **Document Processing**
  - `src/HtmlAgilityPack.Shared/HtmlDocument.cs`
  - `src/HtmlAgilityPack.Shared/HtmlElementFlag.cs`
  - `src/HtmlAgilityPack.Shared/HtmlParseError.cs`
  - `src/HtmlAgilityPack.Shared/HtmlParseErrorCode.cs`
  - `src/HtmlAgilityPack.Shared/Metro/HtmlWeb.cs`
  - ... and 2 more files
- **Entity Recognition**
  - `src/HtmlAgilityPack.Shared/HtmlAttribute.cs`
  - `src/HtmlAgilityPack.Shared/HtmlAttributeCollection.cs`
  - `src/HtmlAgilityPack.Shared/HtmlCommentNode.cs`
  - `src/HtmlAgilityPack.Shared/HtmlConsoleListener.cs`
  - `src/HtmlAgilityPack.Shared/HtmlDocument.cs`
  - ... and 21 more files
- **Sentiment**
  - `src/HtmlAgilityPack.Shared/HtmlDocument.cs`
  - `src/HtmlAgilityPack.Shared/HtmlEntity.cs`
- **Text Understanding**
  - `src/HtmlAgilityPack.Shared/MimeTypeMap.cs`

### Awesome-AGI

Category: Documentation

#### Functionality

No specific functionality patterns detected.

### awesome-agi-cocosci

Category: Documentation

#### Functionality

No specific functionality patterns detected.

