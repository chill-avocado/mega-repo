# Functionality Summary

This document provides a summary of the functionality found in each repository.

## Agent Frameworks

### AgentForge

#### Tool Use

Found in files:
- `setup.py`
- `src/agentforge/apis/mixins/vision_mixin.py`
- `src/agentforge/modules/actions.py`
- `src/agentforge/storage/chroma_storage.py`
- `src/agentforge/storage/persona_memory.py`
- `src/agentforge/tools/__init__.py`
- `src/agentforge/tools/get_text.py`
- `src/agentforge/tools/google_search.py`
- `src/agentforge/tools/python_function.py`
- `src/agentforge/tools/web_scrape.py`
- ... and 14 more files

#### Agent Lifecycle

Found in files:
- `src/agentforge/cog.py`
- `src/agentforge/config_structs/cog_config_structs.py`
- `src/agentforge/core/agent_registry.py`
- `src/agentforge/core/config_manager.py`
- `src/agentforge/storage/scratchpad.py`
- `src/agentforge/testing/bootstrap.py`
- `tests/agent_tests/test_persona_context_data_integration.py`
- `tests/cog_tests/test_cog_agent_runner_integration.py`
- `tests/cog_tests/test_cog_trail_logging_and_flow_validation.py`
- `tests/cog_tests/test_example_cog.py`
- ... and 6 more files

#### Memory

Found in files:
- `src/agentforge/cog.py`
- `src/agentforge/config_structs/__init__.py`
- `src/agentforge/config_structs/agent_config_structs.py`
- `src/agentforge/config_structs/cog_config_structs.py`
- `src/agentforge/core/agent_runner.py`
- `src/agentforge/core/config_manager.py`
- `src/agentforge/core/memory_manager.py`
- `src/agentforge/modules/actions.py`
- `src/agentforge/storage/chat_history_memory.py`
- `src/agentforge/storage/chroma_storage.py`
- ... and 23 more files

#### Planning

Found in files:
- `src/agentforge/utils/prompt_processor.py`
- `tests/integration_tests/test_persona_memory_integration.py`
- `tests/utils/test_parsing_processor_default_fences.py`

#### Multi Agent

Found in files:
- `tests/core_tests/test_agent_registry.py`

### AgentGPT

#### Tool Use

Found in files:
- `next/src/components/Input.tsx`
- `next/src/components/Label.tsx`
- `next/src/components/Tooltip.tsx`
- `next/src/components/console/ChatMessage.tsx`
- `next/src/components/dialog/HelpDialog.tsx`
- `next/src/components/dialog/SignInDialog.tsx`
- `next/src/components/dialog/ToolsDialog.tsx`
- `next/src/components/index/landing.tsx`
- `next/src/components/landing/Hero.tsx`
- `next/src/components/landing/OpenSource.tsx`
- ... and 38 more files

#### Agent Lifecycle

Found in files:
- `next/src/components/drawer/LeftSidebar.tsx`
- `next/src/components/landing/OpenSource.tsx`
- `next/src/hooks/useAgent.ts`
- `next/src/server/api/routers/agentRouter.ts`
- `next/src/services/agent/agent-api.ts`
- `next/src/services/agent/agent-work/create-task-work.ts`
- `next/src/services/agent/agent-work/start-task-work.ts`
- `next/src/stores/agentInputStore.ts`
- `next/src/stores/agentStore.ts`
- `platform/reworkd_platform/db/crud/agent.py`
- ... and 9 more files

#### Planning

Found in files:
- `next/src/components/NavBar.tsx`
- `next/src/components/console/ExampleAgents.tsx`
- `next/src/components/templates/TemplateData.tsx`
- `platform/reworkd_platform/services/pinecone/pinecone.py`
- `platform/reworkd_platform/web/api/agent/dependancies.py`
- `platform/reworkd_platform/web/api/agent/helpers.py`
- `platform/reworkd_platform/web/api/agent/prompts.py`
- `platform/reworkd_platform/web/api/agent/tools/open_ai_function.py`

#### Memory

Found in files:
- `next/src/components/landing/Section.tsx`
- `next/src/components/templates/TemplateData.tsx`
- `platform/reworkd_platform/services/pinecone/pinecone.py`
- `platform/reworkd_platform/tests/agent/test_task_output_parser.py`
- `platform/reworkd_platform/tests/memory/memory_with_fallback_test.py`
- `platform/reworkd_platform/web/api/agent/prompts.py`
- `platform/reworkd_platform/web/api/memory/memory.py`
- `platform/reworkd_platform/web/api/memory/memory_with_fallback.py`
- `platform/reworkd_platform/web/api/memory/null.py`

### AgentK

#### Tool Use

Found in files:
- `agents/agent_smith.py`
- `agents/hermes.py`
- `agents/software_engineer.py`
- `agents/tool_maker.py`
- `agents/web_researcher.py`
- `tests/tools/test_delete_file.py`
- `tests/tools/test_fetch_web_page_raw_html.py`
- `tests/tools/test_overwrite_file.py`
- `tests/tools/test_read_file.py`
- `tests/tools/test_request_human_input.py`
- ... and 13 more files

#### Agent Lifecycle

Found in files:
- `agents/agent_smith.py`
- `agents/hermes.py`
- `agents/software_engineer.py`
- `agents/tool_maker.py`

#### Planning

Found in files:
- `agents/agent_smith.py`
- `agents/hermes.py`

### Auto-GPT-MetaTrader-Plugin

#### Planning

Found in files:
- `src/auto_gpt_metatrader/__init__.py`

#### Tool Use

Found in files:
- `src/auto_gpt_metatrader/trading.py`

### Free-Auto-GPT

#### Tool Use

Found in files:
- `AUTOGPT.py`
- `BABYAGI.py`
- `Camel.py`
- `Embedding/HuggingFaceEmbedding.py`
- `FreeLLM/ChatGPTAPI.py`
- `MetaPrompt.py`
- `OtherAgent/FreeLLM/ChatGPTAPI.py`
- `OtherAgent/Tool/browserQA.py`
- `OtherAgent/csvAgent.py`
- `OtherAgent/customAgent.py`
- ... and 4 more files

#### Agent Lifecycle

Found in files:
- `AUTOGPT.py`
- `OtherAgent/csvAgent.py`
- `OtherAgent/customAgent.py`
- `OtherAgent/pythonAgent.py`

#### Memory

Found in files:
- `AUTOGPT.py`
- `BABYAGI.py`
- `FreeLLM/BardChatAPI.py`
- `FreeLLM/BingChatAPI.py`
- `FreeLLM/ChatGPTAPI.py`
- `FreeLLM/HuggingChatAPI.py`
- `MetaPrompt.py`
- `OtherAgent/FreeLLM/BardChatAPI.py`
- `OtherAgent/FreeLLM/BingChatAPI.py`
- `OtherAgent/FreeLLM/ChatGPTAPI.py`
- ... and 8 more files

#### Planning

Found in files:
- `BABYAGI.py`
- `hfAgent/agents.py`

### agent-zero

#### Memory

Found in files:
- `agent.py`
- `main.py`
- `python/helpers/vdb.py`
- `python/helpers/vector_db.py`
- `python/tools/knowledge_tool.py`
- `python/tools/memory_tool.py`
- `python/tools/response.py`
- `python/tools/task_done.py`

#### Tool Use

Found in files:
- `agent.py`
- `main.py`
- `python/helpers/extract_tools.py`
- `python/helpers/tool.py`
- `python/tools/call_subordinate.py`
- `python/tools/code_execution_tool.py`
- `python/tools/knowledge_tool.py`
- `python/tools/memory_tool.py`
- `python/tools/online_knowledge_tool.py`
- `python/tools/response.py`
- ... and 2 more files

#### Agent Lifecycle

Found in files:
- `main.py`
- `python/tools/call_subordinate.py`

### autogen

#### Tool Use

Found in files:
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Anthropic_Agent_With_Prompt_Caching.cs`
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent_With_Tool.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/CreateAnAgent.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/FunctionCallCodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/MistralAICodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/OpenAICodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/TypeSafeFunctionCallCodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example03_Agent_FunctionCall.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example07_Dynamic_GroupChat_Calculate_Fibonacci.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example10_SemanticKernel.cs`
- ... and 404 more files

#### Multi Agent

Found in files:
- `dotnet/src/AutoGen.Anthropic/Middleware/AnthropicMessageConnector.cs`
- `dotnet/src/AutoGen.AzureAIInference/Middleware/AzureAIInferenceChatRequestMessageConnector.cs`
- `dotnet/src/AutoGen.Gemini/Middleware/GeminiMessageConnector.cs`
- `dotnet/src/AutoGen.Ollama/Middlewares/OllamaMessageConnector.cs`
- `dotnet/src/AutoGen.OpenAI.V1/Middleware/OpenAIChatRequestMessageConnector.cs`
- `dotnet/src/AutoGen.OpenAI/Middleware/OpenAIChatRequestMessageConnector.cs`
- `dotnet/src/AutoGen.WebAPI/OpenAI/Service/OpenAIChatCompletionService.cs`
- `dotnet/src/Microsoft.AutoGen/AgentChat/Abstractions/Messages.cs`
- `dotnet/test/AutoGen.AzureAIInference.Tests/ChatCompletionClientAgentTests.cs`
- `dotnet/test/Microsoft.AutoGen.Integration.Tests.AppHosts/core_xlang_hello_python_agent/protos/__init__.py`
- ... and 22 more files

#### Memory

Found in files:
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Anthropic_Agent_With_Prompt_Caching.cs`
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent_With_Tool.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example01_AssistantAgent.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example04_Dynamic_GroupChat_Coding_Task.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example11_Sequential_GroupChat_Example.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example12_TwoAgent_Fill_Application.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example15_GPT4V_BinaryDataImageMessage.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example17_ReActAgent.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/GettingStart/Chat_With_Agent.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/GettingStart/Dynamic_Group_Chat.cs`
- ... and 155 more files

#### Planning

Found in files:
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Anthropic_Agent_With_Prompt_Caching.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Agents/Developer/DeveloperPrompts.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Agents/DeveloperLead/DeveloperLead.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Agents/DeveloperLead/DeveloperLeadPrompts.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Agents/Hubber.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Agents/ProductManager/PMPrompts.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Models/DevPlan.cs`
- `dotnet/samples/dev-team/DevTeam.Backend/Services/GithubWebHookProcessor.cs`
- `dotnet/test/AutoGen.Anthropic.Tests/AnthropicTestUtils.cs`
- `python/packages/agbench/benchmarks/process_logs.py`
- ... and 22 more files

#### Agent Lifecycle

Found in files:
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent.cs`
- `dotnet/samples/AgentChat/AutoGen.Anthropic.Sample/Create_Anthropic_Agent_With_Tool.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/CreateAnAgent.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/GetStartCodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/MiddlewareAgentCodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/MistralAICodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/OpenAICodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/SemanticKernelCodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/CodeSnippet/UserProxyAgentCodeSnippet.cs`
- `dotnet/samples/AgentChat/AutoGen.Basic.Sample/Example02_TwoAgent_MathChat.cs`
- ... and 148 more files

### babyagi

#### Tool Use

Found in files:
- `babyagi/functionz/core/registration.py`
- `babyagi/functionz/packs/default/default_functions.py`
- `babyagi/functionz/packs/default/function_calling_chat.py`
- `babyagi/functionz/packs/drafts/generate_function.py`
- `babyagi/functionz/packs/drafts/react_agent.py`
- `babyagi/functionz/packs/drafts/self_build.py`
- `setup.py`

#### Planning

Found in files:
- `babyagi/functionz/packs/drafts/react_agent.py`

#### Memory

Found in files:
- `babyagi/dashboard/static/js/function_details.js`
- `babyagi/functionz/packs/default/function_calling_chat.py`
- `babyagi/functionz/packs/drafts/react_agent.py`

### Teenage-AGI

#### Memory

Found in files:
- `agent.py`
- `main.py`

#### Tool Use

Found in files:
- `agent.py`
- `main.py`


## User Interfaces

### open-webui

#### Chat Interface

Found in files:
- `backend/open_webui/config.py`
- `backend/open_webui/constants.py`
- `backend/open_webui/env.py`
- `backend/open_webui/functions.py`
- `backend/open_webui/internal/migrations/001_initial_schema.py`
- `backend/open_webui/internal/migrations/002_add_local_sharing.py`
- `backend/open_webui/internal/migrations/004_add_archived.py`
- `backend/open_webui/internal/migrations/005_add_updated_at.py`
- `backend/open_webui/internal/migrations/006_migrate_timestamps_and_charfields.py`
- `backend/open_webui/main.py`
- ... and 83 more files

#### Components

Found in files:
- `backend/open_webui/config.py`
- `backend/open_webui/constants.py`
- `backend/open_webui/env.py`
- `backend/open_webui/functions.py`
- `backend/open_webui/internal/migrations/001_initial_schema.py`
- `backend/open_webui/internal/migrations/010_migrate_modelfiles_to_models.py`
- `backend/open_webui/main.py`
- `backend/open_webui/migrations/versions/242a2047eae0_update_chat_table.py`
- `backend/open_webui/migrations/versions/4ace53fd72c8_update_folder_table_datetime.py`
- `backend/open_webui/models/auths.py`
- ... and 104 more files

#### Visualization

Found in files:
- `backend/open_webui/config.py`
- `backend/open_webui/env.py`
- `backend/open_webui/main.py`
- `backend/open_webui/models/models.py`
- `backend/open_webui/retrieval/web/serply.py`
- `backend/open_webui/routers/audio.py`
- `backend/open_webui/routers/scim.py`
- `backend/open_webui/utils/auth.py`
- `backend/open_webui/utils/code_interpreter.py`
- `backend/open_webui/utils/middleware.py`
- ... and 6 more files

#### Authentication

Found in files:
- `backend/open_webui/config.py`
- `backend/open_webui/constants.py`
- `backend/open_webui/env.py`
- `backend/open_webui/internal/migrations/001_initial_schema.py`
- `backend/open_webui/internal/migrations/006_migrate_timestamps_and_charfields.py`
- `backend/open_webui/internal/migrations/017_add_user_oauth_sub.py`
- `backend/open_webui/internal/migrations/018_add_function_is_global.py`
- `backend/open_webui/main.py`
- `backend/open_webui/migrations/env.py`
- `backend/open_webui/migrations/versions/7e5b5dc7342b_init.py`
- ... and 96 more files

#### Responsive

Found in files:
- `backend/open_webui/config.py`
- `backend/open_webui/retrieval/web/serply.py`
- `src/lib/stores/index.ts`
- `src/lib/utils/onedrive-file-picker.ts`
- `svelte.config.js`

### web-ui

#### Components

Found in files:
- `src/agent/browser_use/browser_use_agent.py`
- `src/agent/deep_research/deep_research_agent.py`
- `src/controller/custom_controller.py`
- `src/utils/llm_provider.py`
- `src/utils/mcp_client.py`
- `src/webui/components/agent_settings_tab.py`
- `src/webui/components/browser_settings_tab.py`
- `src/webui/components/browser_use_agent_tab.py`
- `src/webui/components/deep_research_agent_tab.py`
- `src/webui/components/load_save_config_tab.py`
- ... and 5 more files

#### Chat Interface

Found in files:
- `src/agent/browser_use/browser_use_agent.py`
- `src/agent/deep_research/deep_research_agent.py`
- `src/controller/custom_controller.py`
- `src/utils/config.py`
- `src/utils/llm_provider.py`
- `src/webui/components/browser_use_agent_tab.py`
- `src/webui/webui_manager.py`
- `tests/test_agents.py`
- `tests/test_llm_api.py`

#### Responsive

Found in files:
- `tests/test_agents.py`
- `tests/test_controller.py`

#### Authentication

Found in files:
- `tests/test_llm_api.py`

#### Visualization

Found in files:
- `src/agent/deep_research/deep_research_agent.py`
- `src/utils/config.py`
- `src/utils/llm_provider.py`
- `src/webui/components/browser_use_agent_tab.py`
- `src/webui/components/deep_research_agent_tab.py`

### draw-a-ui

#### Components

Found in files:
- `app/page.tsx`
- `components/PreviewModal.tsx`
- `lib/getSvgAsImage.ts`
- `lib/png.ts`
- `tailwind.config.ts`

#### Visualization

Found in files:
- `lib/getSvgAsImage.ts`

#### Chat Interface

Found in files:
- `app/api/toHtml/route.ts`
- `app/page.tsx`

#### Responsive

Found in files:
- `app/layout.tsx`


## Os Interaction

### MacOS-Agent

#### Input Control

Found in files:
- `macos_agent_server.py`

#### File System

Found in files:
- `macos_agent_server.py`

### self-operating-computer

#### Screen Capture

Found in files:
- `evaluate.py`
- `operate/models/apis.py`
- `operate/models/prompts.py`
- `operate/utils/ocr.py`
- `operate/utils/screenshot.py`

#### Input Control

Found in files:
- `operate/main.py`
- `operate/models/prompts.py`
- `operate/operate.py`
- `operate/utils/operating_system.py`

#### File System

Found in files:
- `operate/utils/ocr.py`

### mcp-remote-macos-use

#### File System

Found in files:
- `conftest.py`
- `src/action_handlers.py`
- `src/mcp_remote_macos_use/__init__.py`
- `src/mcp_remote_macos_use/server.py`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_action_handlers.py`
- `tests/test_suite.py`

#### Screen Capture

Found in files:
- `src/action_handlers.py`
- `src/mcp_remote_macos_use/server.py`
- `src/vnc_client.py`
- `tests/test_action_handlers.py`
- `tests/test_vnc_client.py`

#### Input Control

Found in files:
- `src/action_handlers.py`
- `src/mcp_remote_macos_use/server.py`
- `src/vnc_client.py`
- `tests/test_action_handlers.py`
- `tests/test_server.py`

#### System Monitoring

Found in files:
- `src/vnc_client.py`

#### App Control

Found in files:
- `src/action_handlers.py`


## Browser Automation

### browser-use

#### Navigation

Found in files:
- `browser_use/__init__.py`
- `browser_use/agent/cloud_events.py`
- `browser_use/agent/message_manager/service.py`
- `browser_use/agent/prompts.py`
- `browser_use/agent/service.py`
- `browser_use/agent/views.py`
- `browser_use/browser/extensions.py`
- `browser_use/browser/profile.py`
- `browser_use/browser/session.py`
- `browser_use/browser/utils.py`
- ... and 141 more files

#### Interaction

Found in files:
- `browser_use/__init__.py`
- `browser_use/agent/cloud_events.py`
- `browser_use/agent/gif.py`
- `browser_use/agent/message_manager/service.py`
- `browser_use/agent/message_manager/views.py`
- `browser_use/agent/prompts.py`
- `browser_use/agent/service.py`
- `browser_use/agent/views.py`
- `browser_use/browser/__init__.py`
- `browser_use/browser/extensions.py`
- ... and 157 more files

#### Headless

Found in files:
- `browser_use/agent/cloud_events.py`
- `browser_use/browser/profile.py`
- `browser_use/browser/session.py`
- `browser_use/cli.py`
- `browser_use/config.py`
- `browser_use/dom/playground/extraction.py`
- `browser_use/dom/playground/process_dom.py`
- `browser_use/dom/playground/test_accessibility.py`
- `browser_use/mcp/controller.py`
- `browser_use/mcp/server.py`
- ... and 65 more files

#### Scraping

Found in files:
- `browser_use/agent/message_manager/service.py`
- `browser_use/agent/prompts.py`
- `browser_use/agent/service.py`
- `browser_use/browser/profile.py`
- `browser_use/controller/service.py`
- `browser_use/dom/service.py`
- `browser_use/llm/anthropic/serializer.py`
- `browser_use/mcp/server.py`
- `examples/file_system/file_system.py`
- `examples/getting_started/03_data_extraction.py`
- ... and 2 more files

#### Automation

Found in files:
- `examples/getting_started/04_multi_step_task.py`
- `examples/mcp/advanced_server.py`
- `examples/ui/command_line.py`
- `examples/ui/streamlit_demo.py`
- `examples/use-cases/captcha.py`
- `examples/use-cases/post-twitter.py`
- `examples/use-cases/scrolling_page.py`
- `examples/use-cases/twitter_post_using_cookies.py`
- `tests/ci/test_filesystem.py`


## Code Execution

### open-interpreter

#### Execution

Found in files:
- `interpreter/computer_use/loop.py`
- `interpreter/computer_use/tools/base.py`
- `interpreter/computer_use/tools/bash.py`
- `interpreter/computer_use/tools/edit.py`
- `interpreter/computer_use/tools/run.py`
- `interpreter/core/archived_server_2.py`
- `interpreter/core/async_core.py`
- `interpreter/core/computer/browser/browser.py`
- `interpreter/core/computer/browser/browser_next.py`
- `interpreter/core/computer/computer.py`
- ... and 47 more files

#### Development

Found in files:
- `interpreter/computer_use/loop.py`
- `interpreter/computer_use/tools/bash.py`
- `interpreter/computer_use/tools/edit.py`
- `interpreter/computer_use/tools/run.py`
- `interpreter/core/async_core.py`
- `interpreter/core/computer/ai/ai.py`
- `interpreter/core/computer/browser/browser.py`
- `interpreter/core/computer/calendar/calendar.py`
- `interpreter/core/computer/contacts/contacts.py`
- `interpreter/core/computer/display/display.py`
- ... and 45 more files

#### Code Generation

Found in files:
- `interpreter/core/computer/terminal/languages/jupyter_language.py`
- `interpreter/core/llm/llm.py`
- `interpreter/core/llm/run_function_calling_llm.py`
- `interpreter/core/llm/run_tool_calling_llm.py`
- `interpreter/terminal_interface/profiles/defaults/assistant.py`
- `interpreter/terminal_interface/profiles/profiles.py`
- `interpreter/terminal_interface/start_terminal_interface.py`
- `tests/test_interpreter.py`

#### Language Support

Found in files:
- `interpreter/computer_use/unused_markdown.py`
- `interpreter/core/computer/computer.py`
- `interpreter/core/computer/display/display.py`
- `interpreter/core/computer/mouse/mouse.py`
- `interpreter/core/computer/skills/skills.py`
- `interpreter/core/computer/terminal/languages/javascript.py`
- `interpreter/core/computer/terminal/languages/jupyter_language.py`
- `interpreter/core/computer/terminal/languages/python.py`
- `interpreter/core/computer/terminal/languages/subprocess_language.py`
- `interpreter/core/computer/terminal/terminal.py`
- ... and 36 more files

#### Analysis

Found in files:
- `interpreter/terminal_interface/profiles/defaults/the01.py`

### pyCodeAGI

#### Code Generation

Found in files:
- `pycodeagi-gpt4.py`
- `pycodeagi.py`

#### Execution

Found in files:
- `pycodeagi-gpt4.py`
- `pycodeagi.py`

#### Language Support

Found in files:
- `pycodeagi-gpt4.py`
- `pycodeagi.py`

#### Development

Found in files:
- `pycodeagi-gpt4.py`
- `pycodeagi.py`

### SuperCoder

#### Execution

Found in files:
- `app/models/dtos/asynq_task/create_job_payload.go`
- `app/repositories/execution.go`
- `app/services/design_story_review_service.go`
- `app/services/execution_service.go`
- `app/services/pull_request_comments_service.go`
- `app/services/story_service.go`
- `app/tasks/create_execution_job_task.go`
- `app/workflow_executors/step_executors/code_generation_executor.go`
- `app/workflow_executors/step_executors/git_commit_executor.go`
- `app/workflow_executors/step_executors/git_make_branch_executor.go`
- ... and 24 more files

#### Language Support

Found in files:
- `app/gateways/websocket_gateway.go`
- `app/tasks/create_execution_job_task.go`
- `app/workflow_executors/step_executors/impl/django_server_step_executor.go`
- `app/workflow_executors/step_executors/impl/flask_reset_db_step_executor.go`
- `app/workflow_executors/step_executors/impl/flask_sever_test_executor.go`
- `app/workflow_executors/step_executors/impl/open_ai_code_generation_executor.go`
- `app/workflow_executors/step_executors/impl/open_ai_update_code_file_executor.go`
- `gui/src/components/CustomDiffEditor/MonacoDiffEditor.tsx`
- `gui/src/components/SyntaxDisplay/SyntaxDisplay.tsx`
- `workspace-service/app/config/config.go`
- ... and 4 more files

#### Development

Found in files:
- `app/client/git_provider/gitness_git_provider.go`
- `app/config/ai_developer_execution.go`
- `app/config/config.go`
- `app/controllers/pull_request_controller.go`
- `app/models/dtos/gitness/types.go`
- `app/services/auth/auth_provider.go`
- `app/services/auth/authenticator.go`
- `app/services/code_download_service.go`
- `app/services/git_providers/git_provider.go`
- `app/services/git_providers/gitness_git_provider_service.go`
- ... and 47 more files

#### Analysis

Found in files:
- `app/workflow_executors/step_executors/impl/next_js_server_test_executor.go`
- `gui/src/app/imagePath.tsx`
- `gui/src/components/HomeComponents/GithubStarModal.tsx`

#### Code Generation

Found in files:
- `app/models/execution_step.go`
- `app/workflow_executors/design_nextjs_workflow_config.go`
- `app/workflow_executors/python_django_workflow_config.go`
- `app/workflow_executors/python_flask_workflow_config.go`
- `app/workflow_executors/step_executors/code_generation_executor.go`
- `app/workflow_executors/step_executors/impl/next_js_server_test_executor.go`
- `app/workflow_executors/step_executors/impl/open_ai_code_generation_executor.go`
- `app/workflow_executors/step_executors/impl/open_ai_next_js_code_generation_executor.go`
- `app/workflow_executors/step_executors/impl/open_ai_next_js_update_code_file_executor.go`
- `app/workflow_executors/step_executors/impl/open_ai_update_code_file_executor.go`
- ... and 4 more files

### automata

#### Language Support

Found in files:
- `automata/cli/commands.py`
- `automata/cli/install_indexing.py`
- `automata/code_parsers/py/py_reader.py`
- `automata/code_writers/py/__init__.py`
- `automata/code_writers/py/py_code_writer.py`
- `automata/code_writers/py/py_doc_writer.py`
- `automata/config/prompt/doc_generation.py`
- `automata/core/base/database/vector_database.py`
- `automata/core/utils.py`
- `automata/eval/agent/code_writing_eval.py`
- ... and 40 more files

#### Execution

Found in files:
- `automata/agent/openai_agent.py`
- `automata/cli/commands.py`
- `automata/cli/install_indexing.py`
- `automata/cli/options.py`
- `automata/cli/scripts/run_agent_eval.py`
- `automata/cli/scripts/run_code_embedding.py`
- `automata/core/base/database/relational_database.py`
- `automata/core/run_handlers.py`
- `automata/eval/agent/agent_eval.py`
- `automata/eval/agent/agent_eval_composite.py`
- ... and 28 more files

#### Development

Found in files:
- `automata/agent/agent.py`
- `automata/agent/error.py`
- `automata/agent/openai_agent.py`
- `automata/cli/env_operations.py`
- `automata/cli/scripts/run_agent.py`
- `automata/cli/scripts/run_code_embedding.py`
- `automata/cli/scripts/run_doc_embedding.py`
- `automata/code_writers/py/py_code_writer.py`
- `automata/code_writers/py/py_doc_writer.py`
- `automata/config/__init__.py`
- ... and 91 more files

#### Code Generation

Found in files:
- `automata/symbol/scip_pb2.py`
- `automata/tests/unit/agent_eval/conftest.py`
- `automata/tests/unit/agent_eval/test_agent_eval_actions_results.py`
- `automata/tests/unit/code_writers/test_py_writer.py`
- `research/study_agency/study_human_eval/completion_provider.py`
- `research/study_agency/study_leetcode/run_automata_problem_solver.py`
- `research/study_agency/study_leetcode/run_replay_leetcode_responses.py`
- `research/study_agency/study_leetcode/run_vanilla_problem_solver.py`

#### Analysis

Found in files:
- `research/study_agency/study_leetcode/leetcode_constants.py`
- `research/study_agency/study_leetcode/leetcode_problem_solver.py`


## Cognitive Systems

### opencog

#### Reasoning

Found in files:
- `opencog/nlp/chatbot/telegram_bot.py`

### openagi

#### Learning

Found in files:
- `example/job_search.py`
- `example/youtube_study_plan.py`
- `src/openagi/llms/base.py`
- `src/openagi/prompts/task_creator.py`

#### Cognitive Processes

Found in files:
- `example/job_post.py`
- `src/openagi/prompts/task_creator.py`
- `src/openagi/utils/llmTasks.py`

#### Reasoning

Found in files:
- `example/market_research.py`
- `src/openagi/llms/azure.py`
- `src/openagi/llms/cohere.py`
- `src/openagi/llms/gemini.py`
- `src/openagi/llms/groq.py`
- `src/openagi/llms/hf.py`
- `src/openagi/llms/mistral.py`
- `src/openagi/llms/ollama.py`
- `src/openagi/llms/openai.py`
- `src/openagi/llms/xai.py`
- ... and 1 more files

#### Knowledge Representation

Found in files:
- `src/openagi/memory/base.py`

#### Memory Systems

Found in files:
- `src/openagi/memory/base.py`

### AGI-Samantha

#### Reasoning

Found in files:
- `AGI-1.py`

#### Memory Systems

Found in files:
- `AGI-1.py`

#### Cognitive Processes

Found in files:
- `AGI-1.py`


## Evolution Optimization

### openevolve

#### Evolutionary Algorithms

Found in files:
- `examples/attention_optimization/initial_program.py`
- `examples/attention_optimization/scripts/mlir_lowering_pipeline.py`
- `examples/attention_optimization/scripts/mlir_syntax_test.py`
- `examples/attention_optimization/tests/test_evaluator.py`
- `examples/attention_optimization/tests/test_results.py`
- `examples/circle_packing/initial_program.py`
- `examples/circle_packing_with_artifacts/initial_program.py`
- `examples/lm_eval/lm-eval.py`
- `examples/mlx_metal_kernel_opt/best_program.py`
- `examples/mlx_metal_kernel_opt/evaluator.py`
- ... and 19 more files

#### Benchmarking

Found in files:
- `examples/attention_optimization/evaluator.py`
- `examples/attention_optimization/legacy/prev_sim__works_evaluator.py`
- `examples/attention_optimization/scripts/debug_real_execution.py`
- `examples/attention_optimization/tests/test_evaluator.py`
- `examples/circle_packing/evaluator.py`
- `examples/circle_packing_with_artifacts/evaluator.py`
- `examples/function_minimization/evaluator.py`
- `examples/llm_prompt_optimization/evaluate_prompts.py`
- `examples/llm_prompt_optimization/evaluator.py`
- `examples/lm_eval/lm-eval.py`
- ... and 29 more files

#### Adaptive Learning

Found in files:
- `examples/circle_packing_with_artifacts/evaluator.py`
- `examples/mlx_metal_kernel_opt/qwen3_benchmark_suite.py`
- `examples/online_judge_programming/submit.py`
- `examples/rust_adaptive_sort/evaluator.py`
- `examples/rust_adaptive_sort/initial_program.rs`
- `examples/signal_processing/evaluator.py`
- `examples/signal_processing/initial_program.py`
- `examples/symbolic_regression/data_api.py`
- `examples/web_scraper_optillm/evaluator.py`
- `openevolve/config.py`
- ... and 3 more files

#### Optimization

Found in files:
- `examples/attention_optimization/evaluator.py`
- `examples/attention_optimization/initial_program.py`
- `examples/attention_optimization/legacy/prev_sim__works_evaluator.py`
- `examples/attention_optimization/tests/test_evaluator.py`
- `examples/circle_packing/best_program.py`
- `examples/mlx_metal_kernel_opt/best_program.py`
- `examples/mlx_metal_kernel_opt/initial_program.py`
- `examples/mlx_metal_kernel_opt/quick_demo.py`
- `examples/mlx_metal_kernel_opt/qwen3_benchmark_suite.py`
- `examples/mlx_metal_kernel_opt/run_benchmarks.py`
- ... and 4 more files

#### Multi Objective

Found in files:
- `examples/attention_optimization/initial_program.py`
- `examples/signal_processing/evaluator.py`


## Integration

### Unification

#### Api Standardization

Found in files:
- `features/mcp-remote-macos-use-main/src/vnc_client.py`
- `features/mcp-remote-macos-use-main/tests/test_vnc_client.py`

#### Orchestration

Found in files:
- `features/AgentK-master/agents/hermes.py`

### LocalAGI

#### System Integration

Found in files:
- `services/actions/githubprreviewer_test.go`

#### Compatibility

Found in files:
- `pkg/stdio/client.go`


## Nlp

### senpai

#### Entity Recognition

Found in files:
- `senpai/agent.py`
- `senpai/memory.py`

### html-agility-pack

#### Document Processing

Found in files:
- `src/HtmlAgilityPack.Shared/HtmlDocument.cs`
- `src/HtmlAgilityPack.Shared/HtmlElementFlag.cs`
- `src/HtmlAgilityPack.Shared/HtmlParseError.cs`
- `src/HtmlAgilityPack.Shared/HtmlParseErrorCode.cs`
- `src/HtmlAgilityPack.Shared/Metro/HtmlWeb.cs`
- `src/HtmlAgilityPack.Shared/MimeTypeMap.cs`
- `src/Tests/HtmlAgilityPack.Tests.Net45/HtmlDocumentTests.cs`

#### Entity Recognition

Found in files:
- `src/HtmlAgilityPack.Shared/HtmlAttribute.cs`
- `src/HtmlAgilityPack.Shared/HtmlAttributeCollection.cs`
- `src/HtmlAgilityPack.Shared/HtmlCommentNode.cs`
- `src/HtmlAgilityPack.Shared/HtmlConsoleListener.cs`
- `src/HtmlAgilityPack.Shared/HtmlDocument.cs`
- `src/HtmlAgilityPack.Shared/HtmlEntity.cs`
- `src/HtmlAgilityPack.Shared/HtmlNode.Encapsulator.cs`
- `src/HtmlAgilityPack.Shared/HtmlNode.Xpath.cs`
- `src/HtmlAgilityPack.Shared/HtmlNode.cs`
- `src/HtmlAgilityPack.Shared/HtmlNodeCollection.cs`
- ... and 16 more files

#### Sentiment

Found in files:
- `src/HtmlAgilityPack.Shared/HtmlDocument.cs`
- `src/HtmlAgilityPack.Shared/HtmlEntity.cs`

#### Text Understanding

Found in files:
- `src/HtmlAgilityPack.Shared/MimeTypeMap.cs`


## Documentation

### Awesome-AGI

No specific functionality patterns detected.

### awesome-agi-cocosci

No specific functionality patterns detected.


