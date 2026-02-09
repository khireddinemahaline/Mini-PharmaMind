# Mini-PharmaMind 🧬💊

> **⚠️ Mini Version Notice:** This is a **proof-of-concept** limited version of PharmaMind for educational and research purposes.

An intelligent multi-agent system for pharmaceutical research and drug discovery, powered by AI. Mini-PharmaMind orchestrates specialized agents to analyze targets, search for drugs, and generate comprehensive research reports.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0--mini-orange)](https://github.com/khireddinemahaline/Mini-PharmaMind)

## 🌟 Features

- **Multi-Agent Orchestration**: Specialized agents working together for drug discovery workflows
- **Target Search Agent**: Analyzes biological targets using ChEMBL and OpenTargets databases
- **Drug Search Agent**: Discovers and evaluates potential drug candidates
- **Report Generation**: Creates comprehensive PDF reports with findings
- **Interactive Chat Interface**: User-friendly Chainlit interface for conversational AI
- **Real-time Streaming**: Stream responses and updates in real-time
- **Session Management**: Persistent state across conversations
- **Arize Phoenix Observability**: Full multi-agent tracing with OpenTelemetry and OpenInference instrumentation

## ⚠️ Current Limitations (Mini Version)

This mini version has the following limitations:

### Architecture
- ✅ **Functional Tools Implementation** - Uses direct API calls instead of MCP servers for simplicity
- ⚠️ **Limited Test Coverage** - Test functions are basic and need expansion
- ⚠️ **Performance Issues** - Higher latency, optimization needed
- ⚠️ **Basic Tool Set** - Limited tools compared to full version

### Missing Features (Available in Full Version)
- ❌ **Advanced RAG System** - Needed for real pharmaceutical laboratory analysis
- ❌ **Drug-Target Simulation** - Molecular dynamics and docking simulations
- ❌ **Physicochemical Prediction** - AI-powered ADMET and property predictions
- ❌ **MCP Server Architecture** - Scalable Model Context Protocol integration
- ❌ **Real-time Lab Integration** - Connection with laboratory equipment
- ❌ **Advanced Caching** - Performance optimization layer

## 🚀 Future Enhancements (Full Version Roadmap)

We welcome collaboration on developing the full PharmaMind version with:

### 🧪 Advanced Computational Chemistry
- **Molecular Dynamics Simulations** - Drug-target interaction modeling
- **Binding Affinity Prediction** - AI-powered binding score calculations
- **ADMET Property Prediction** - Absorption, Distribution, Metabolism, Excretion, Toxicity
- **Physicochemical Parameter Prediction** - LogP, solubility, permeability, etc.
- **Molecular Docking** - Automated docking with multiple scoring functions

### 🤖 Enhanced AI Capabilities
- **RAG System** - Retrieval-Augmented Generation for pharmaceutical literature
- **Multi-Modal Analysis** - Process scientific papers, patents, and molecular structures
- **Knowledge Graph Integration** - Drug-disease-target relationship mapping
- **Predictive Models** - Machine learning for drug efficacy and safety

### ⚡ Performance & Scalability
- **MCP Server Architecture** - Distributed agent system
- **Response Caching** - Reduce API calls and improve latency
- **Parallel Processing** - Concurrent agent execution
- **Optimized Database** - Advanced querying and indexing

### 🔬 Laboratory Integration
- **LIMS Integration** - Laboratory Information Management Systems
- **Instrument Data** - Real-time data from analytical instruments
- **Electronic Lab Notebooks** - Seamless documentation
- **Workflow Automation** - End-to-end laboratory process automation

### 🤝 Collaboration Opportunities

We're looking for collaborators in:
- **Computational Chemistry** - Molecular simulation experts
- **Machine Learning** - ML model development for drug discovery
- **Pharmaceutical Research** - Domain experts for validation
- **Software Engineering** - Scalability and performance optimization
- **Data Science** - Large-scale data processing and analysis

**Interested in contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📋 Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (for Prisma database)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/khireddinemahaline/Mini-PharmaMind.git
   cd Mini-PharmaMind/pharma-core
   ```

2. **Run the setup script** (recommended)
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   Or manually:

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

   Required environment variables:
   ```env
   # LLM Provider (openai or deepseek)
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Database
   DATABASE_URL=postgresql://user:password@localhost:5432/pharmadb
   
   # Chainlit Configuration
   CHAINLIT_AUTH_SECRET=your_secret_key_here

   # Arize Phoenix Observability
   ARIZE_SPACE_ID=your_arize_space_id
   ARIZE_API_KEY=your_arize_api_key
   ```

4. **Install dependencies**
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   pip install -e .
   ```

5. **Set up the database**
   ```bash
   npx prisma generate
   npx prisma db push
   ```

6. **Run the application**
   ```bash
   chainlit run orcastration/main_chainlit.py -w --host 0.0.0.0 --port 8000
   ```

   Access the interface at `http://localhost:8000`

### 🔐 Default Test Credentials

For testing purposes, you can use these default credentials:
- **Username**: `researcher`
- **Password**: `easydiscovery##1`

> **⚠️ Security Note**: These are default credentials for testing only. In production, change these immediately and use strong passwords.

## 🐳 Docker Deployment

### Quick Start with Docker Compose
```bash
docker compose up --build
```

### Development Mode
```bash
docker compose -f docker-compose_dev.yaml up --build
```

## 📁 Project Structure

```
pharma-core/
├── agents/                 # Agent implementations
│   ├── target_search.py   # Target search agent
│   ├── drug_search.py     # Drug search agent
│   ├── report.py          # Report generation agent
│   └── critique.py        # Critique agent
├── config/                # Configuration files
│   ├── llm_client.py      # LLM client setup
│   └── sytem_prompts.py   # System prompts
├── orcastration/          # Main orchestration
│   └── main_chainlit.py   # Chainlit interface
├── tools/                 # Agent tools and utilities
├── utilities/             # Helper functions
├── tests/                 # Test suite
├── prisma/                # Database schema
│   └── schema.prisma
├── pyproject.toml         # Project configuration
├── Dockerfile             # Docker configuration
├── setup.sh               # Automated setup script
└── README.md              # This file
```

## 🔧 Configuration

### LLM Models

Configure your preferred models in `.env`:

```env
# Use OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4-turbo-preview

# Or use DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API=your_key_here
LLM_MODEL=deepseek-chat
```

### Agent Behavior

Customize agent prompts and behavior in `config/sytem_prompts.py`.

## 🤖 Agents

### Target Search Agent
Analyzes biological targets, diseases, and pathways using:
- ChEMBL database queries
- OpenTargets platform integration
- Literature search capabilities

### Drug Search Agent
Discovers drug candidates through:
- Chemical structure search
- Similarity analysis
- Drug-target interaction queries
- ADMET property evaluation

### Report Generation Agent
Creates comprehensive reports including:
- Executive summaries
- Detailed findings
- Visualizations
- References and citations

### Critique Agent
Reviews and improves outputs by:
- Validating scientific accuracy
- Checking completeness
- Suggesting improvements


## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=. --cov-report=html tests/
```

## 🔐 Authentication

PharmaMind uses Chainlit's built-in authentication. Configure authentication settings in your Chainlit configuration.

## 📝 Development

### Install development dependencies

```bash
uv sync --dev
```



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

- [Architecture Overview](ARCHITECTURE.md) - System design and components
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute
- [Changelog](CHANGELOG.md) - Version history

## 🙏 Acknowledgments

- **ChEMBL** - Chemical database of bioactive molecules
- **OpenTargets** - Target validation platform
- **AutoGen** - Multi-agent conversation framework
- **Chainlit** - Conversational AI interface
- **Arize Phoenix** - Multi-agent observability and tracing
- **OpenInference** - OpenTelemetry-based instrumentation for AI systems

## 📧 Contact

MHLAINE Khireddine - mhalaine.khireddine.chimie@gmail.com

Project Link: [https://github.com/khireddinemahaline/Mini-PharmaMind](https://github.com/khireddinemahaline/Mini-PharmaMind)

**Full Version Development:** We're actively working on the full PharmaMind version with advanced features. Interested in early access or collaboration? Reach out!

## 🗺️ Roadmap

### Near-term (Mini Version Improvements)
- [ ] Improve test coverage (increase from basic to comprehensive)
- [ ] Optimize latency and performance
- [ ] Add response caching layer
- [ ] Expand tool library
- [ ] Add more databases (PubChem, DrugBank)
- [ ] Enhance error handling and logging

### Mid-term (Transition to Full Version)
- [ ] Implement RAG system for pharmaceutical literature
- [ ] Add basic physicochemical property prediction
- [ ] Integrate MCP server architecture
- [ ] Implement advanced visualization tools
- [ ] Add collaborative features
- [ ] Create REST API for programmatic access

### Long-term (Full PharmaMind)
- [ ] Molecular dynamics simulation engine
- [ ] Drug-target docking and binding affinity prediction
- [ ] Complete ADMET prediction suite
- [ ] Laboratory equipment integration
- [ ] Multi-modal AI (text, structure, images)
- [ ] Real-time collaboration platform
- [ ] Enterprise-grade security and compliance

**Want to help?** Join us in building the future of AI-powered drug discovery! 🚀

## ⚠️ Disclaimer

This software is for research purposes only. Always consult with qualified pharmaceutical professionals for drug discovery and development decisions.

---

Made with ❤️ for the pharmaceutical research community
