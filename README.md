# P003-LLM: Multi-Service LLM Inference System

## Overview
This project implements an AI-powered configuration system consisting of **3 microservices** and an **intelligent agent** that orchestrates them.

## Architecture

### Services
1. **Data Assets Service** (`data_assets/`)
   - Manages configuration data and knowledge base assets
   - Provides data retrieval for RAG context
   - Stores reference configurations and templates

2. **LLM Inference API** (`02_LLM_INFERENCE_API/`)
   - Hosts the Zephyr configuration agent model
   - Provides RAG-enhanced inference capabilities
   - Configured for precise, non-creative JSON output (temperature: 0.0)

3. **Verification Service** (`agent_validation/`)
   - Validates generated configurations
   - Ensures output compliance and correctness
   - Performs quality assurance checks

### Agent
The **Configuration Agent** serves as the orchestration layer:
- Consumes all three services
- Performs RAG (Retrieval-Augmented Generation) with contextual knowledge injection
- Generates network configurations in strict JSON format
- Template-based prompt engineering for consistent outputs

## Key Features
- **Zero-temperature inference** for deterministic configuration generation
- **Custom prompt templating** with RAG context injection
- **Structured output enforcement** (JSON-only responses)
- **Multi-stop token handling** for clean response termination

## Model Details
- Base Model: `hf.co/ccaug/zephyr-config-agent-3:F16`
- Framework: Ollama-compatible Modelfile
- Output Format: JSON arrays only

## Getting Started
[Installation and usage instructions to be added]

---
*Project: LLM-based Network Configuration System*
