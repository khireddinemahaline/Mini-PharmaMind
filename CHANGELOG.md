# Changelog

All notable changes to Mini-PharmaMind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-mini] - 2026-01-28

### Added
- Initial mini version release (proof of concept)
- Functional tools implementation (not using MCP servers for simplicity)
- Multi-agent system for pharmaceutical research
- Target search agent with ChEMBL and OpenTargets integration
- Drug search agent with compound analysis capabilities
- Report generation agent for creating comprehensive PDF reports
- Critique agent for quality assurance
- Chainlit-based conversational interface
- Arize Phoenix integration for observability and tracing
- Session state persistence
- Real-time streaming responses
- Docker support for containerized deployment
- Comprehensive documentation (README, ARCHITECTURE, CONTRIBUTING)
- MIT License
- Setup automation script
- Development tooling (black, flake8, mypy, pylint, pytest)

### Features
- Support for OpenAI and DeepSeek LLM providers
- PostgreSQL database with Prisma ORM
- Multi-agent orchestration with AutoGen
- Scientific literature search via PubMed
- Chemical structure analysis
- Drug-target interaction queries
- ADMET property evaluation
- Batch compound lookups

## [Unreleased]

### Fixed
- **Arize tracing: unified trace hierarchy** — replaced `Sampler` (`Decision.DROP`) with a `FilteringSpanProcessor` approach. Dropping spans at the sampler level broke parent-child relationships, causing each agent creation to appear as a separate trace. The new processor-level filtering keeps all spans recorded (preserving OpenTelemetry context propagation) but hides noisy autogen spans from the Arize UI, resulting in a single clean trace per workflow.

### Mini Version Improvements
- Improve test coverage (comprehensive testing needed)
- Optimize latency and performance
- Add response caching layer
- Expand tool library
- Better error handling

### Full Version (Planned)
- MCP server architecture implementation
- RAG system for pharmaceutical literature
- Molecular dynamics simulations
- Drug-target docking capabilities
- Physicochemical property prediction (ADMET)
- Laboratory equipment integration
- Advanced visualization tools
- Multi-modal AI capabilities
- Support for additional databases (PubChem, DrugBank)
- Collaborative research features
- REST API for programmatic access
- Enterprise-grade features
