# Architecture Overview

## System Architecture

PharmaMind is a multi-agent AI system designed for pharmaceutical research and drug discovery. The system follows a modular architecture with specialized agents orchestrated through a central workflow engine.

```
┌─────────────────────────────────────────────────────────────┐
│                    Chainlit Interface                        │
│                  (User Interaction Layer)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Orchestration Layer                             │
│          (SelectorGroupChat - main_chainlit.py)              │
└───────┬───────────────┬───────────────┬─────────────────────┘
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
│ TargetSearch │ │ DrugSearch  │ │ ReportAgent    │
│    Agent     │ │   Agent     │ │                │
└───────┬──────┘ └──────┬──────┘ └─────┬──────────┘
        │               │               │
┌───────▼───────────────▼───────────────▼─────────────────────┐
│                     Tools Layer                              │
│  • ChEMBL API      • OpenTargets API                        │
│  • PubMed Search   • Structure Analysis                     │
│  • PDF Generation  • Data Extraction                        │
└──────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│              External Services & Storage                      │
│  • LLM Providers (OpenAI/DeepSeek)                          │
│  • PostgreSQL Database (Prisma)                             │
│  • Arize Phoenix (Observability & Tracing)                   │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agents (`agents/`)

Specialized AI agents that handle different aspects of drug discovery:

- **TargetSearch Agent**: Identifies and analyzes protein targets
  - Uses ChEMBL and OpenTargets databases
  - Searches scientific literature
  - Analyzes disease-target relationships

- **DrugSearch Agent**: Discovers and evaluates drug candidates
  - Searches chemical compound databases
  - Analyzes molecular properties
  - Evaluates drug-target interactions

- **Report Agent**: Generates comprehensive research reports
  - Synthesizes findings from other agents
  - Creates structured PDF reports
  - Includes visualizations and references

- **Critique Agent**: Quality assurance and validation
  - Reviews agent outputs for accuracy
  - Suggests improvements
  - Ensures scientific rigor

### 2. Orchestration Layer (`orcastration/`)

- **main_chainlit.py**: Central orchestration hub
  - Manages multi-agent conversations
  - Handles user interactions through Chainlit
  - Maintains conversation state
  - Routes queries to appropriate agents

### 3. Configuration (`config/`)

- **llm_client.py**: LLM provider configuration
  - Supports multiple providers (OpenAI, DeepSeek)
  - Configurable model selection
  - Retry logic and timeout handling

- **system_prompts.py**: Agent system prompts
  - Defines agent behaviors and expertise
  - Task-specific instructions
  - Output formatting guidelines

### 4. Tools (`tools/`)

Reusable tools that agents can invoke:
- Database query tools (ChEMBL, OpenTargets)
- Literature search (PubMed)
- Structure analysis
- Report generation
- Data extraction and transformation

### 5. Data Layer

- **Prisma ORM**: Database abstraction
  - Schema-first approach
  - Type-safe database access
  - Migration management

- **Session Management**: Persistent conversation state
  - Saves team state to disk
  - Supports conversation recovery
  - Thread-based isolation

## Data Flow

1. **User Query** → Chainlit interface receives input
2. **Orchestration** → SelectorGroupChat determines which agent(s) to invoke
3. **Agent Processing** → Selected agent(s) process query using available tools
4. **Tool Execution** → Agents call external APIs and databases
5. **Response Synthesis** → Agents compile and format results
6. **User Output** → Chainlit streams responses back to user
7. **State Persistence** → System saves conversation state for recovery

## Key Design Patterns

### Agent Pattern
Each agent is a specialized expert with:
- Clear domain of expertise
- Specific toolset
- Defined system prompts
- Streaming capabilities

### Tool Pattern
Tools are reusable, composable functions that:
- Encapsulate external API calls
- Handle error cases
- Return structured data
- Are decorated with metadata for agent discovery

### Orchestration Pattern
SelectorGroupChat implements:
- Dynamic agent selection
- Conversation history management
- Termination conditions
- State persistence

## Technology Stack

- **Framework**: AutoGen (Multi-agent orchestration)
- **UI**: Chainlit (Conversational interface)
- **LLM**: OpenAI GPT-4 / DeepSeek
- **Database**: PostgreSQL + Prisma ORM
- **Tracking**: Arize Phoenix (observability & tracing)
- **APIs**: ChEMBL, OpenTargets, PubMed
- **Deployment**: Docker + Docker Compose

## Scalability Considerations

- **Horizontal Scaling**: Multiple agent instances can run in parallel
- **Caching**: Implement caching for frequently accessed data
- **Rate Limiting**: Built-in retry logic for API calls
- **Async Operations**: Asynchronous processing where possible
- **State Management**: File-based state for simple deployments, can be migrated to Redis/similar for production

## Security

- **API Key Management**: Environment variable based
- **Authentication**: Chainlit auth integration
- **Database Security**: Prisma parameterized queries prevent SQL injection
- **Session Isolation**: Per-user/thread state separation

## Future Enhancements

- [ ] Add more specialized agents (e.g., Toxicology, Clinical Trials)
- [ ] Implement caching layer for API responses
- [ ] Add support for document upload and analysis
- [ ] Enhance visualization capabilities
- [ ] Implement collaborative features for team research
- [ ] Add REST API for programmatic access
