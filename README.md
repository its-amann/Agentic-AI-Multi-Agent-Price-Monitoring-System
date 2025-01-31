<div align="center">

# 🤖 Agentic AI: Multi-Agent Price Intelligence System

<p align="center">
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Framework-Modal-green.svg" alt="Framework">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
<img src="https://img.shields.io/badge/LLM-GPT--4-purple.svg" alt="LLM">
<img src="https://img.shields.io/badge/ML-Llama--3.1-orange.svg" alt="ML">
</p>

<p align="center">
<b>A sophisticated multi-agent system leveraging serverless architecture and ensemble learning for intelligent price predictions and deal discovery.</b>
</p>

<p align="center">
<img src="photos/agent workflow.png" alt="Agent Workflow" width="800">
</p>

</div>

## 📚 Table of Contents

- [Introduction](#-introduction)
- [System Architecture](#-system-architecture)
  - [Agent Framework](#agent-framework)
  - [Data Flow](#data-flow)
  - [RAG Pipeline](#rag-pipeline)
- [Core Components](#-core-components)
  - [Specialist Agent](#specialist-agent)
  - [Frontier Agent](#frontier-agent)
  - [Random Forest Agent](#random-forest-agent)
  - [Ensemble Agent](#ensemble-agent)
  - [Scanner Agent](#scanner-agent)
  - [Planning Agent](#planning-agent)
- [Technical Implementation](#-technical-implementation)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)

## 🌟 Introduction

Agentic AI is a cutting-edge price intelligence system that combines multiple specialized AI agents working in harmony. The system employs:

- 🧠 Fine-tuned LLaMA 3.1 model using QLoRA
- 🔍 RAG (Retrieval Augmented Generation) with ChromaDB
- 🌲 Random Forest ML with vector embeddings
- 🤝 Weighted ensemble learning
- 🔄 Real-time deal scanning and analysis
- 📊 Beautiful Gradio-based UI

<p align="center">
<img src="photos/gradio app ui.png" alt="Gradio UI" width="800">
</p>

## 🏛️ System Architecture

### Agent Framework

<p align="center">
<img src="photos/agent workflow.png" alt="Agent Framework" width="800">
<br>
<em>Multi-Agent System Architecture</em>
</p>

The system is built on a modular agent framework where each agent specializes in specific tasks:

1. **Base Agent Class**
   ```python
   class Agent:
       def __init__(self):
           self.name = self.__class__.__name__
           self.logger = setup_logger(self.name)
           
       async def process(self, message):
           try:
               return await self._handle_message(message)
           except Exception as e:
               self.logger.error(f"Error processing message: {e}")
               raise
   ```

### Data Flow

<p align="center">
<img src="photos/command line working.png" alt="Command Line Interface" width="800">
</p>

The system follows a sophisticated data flow pattern:

1. **Input Processing**
   ```mermaid
   graph LR
   A[Product Description] --> B[Scanner Agent]
   B --> C[Vector Embedding]
   C --> D[Price Analysis]
   D --> E[Deal Detection]
   ```

### RAG Pipeline

<p align="center">
<img src="photos/4o_mini rag 2 .png" alt="RAG Pipeline" width="800">
</p>

The RAG pipeline enhances price predictions with contextual information:

1. **Vector Database**
   - ChromaDB for similarity search
   - 384-dimensional embeddings
   - Price and metadata storage

2. **Retrieval Process**
   ```python
   def retrieve_similar(description: str, k: int = 5):
       embedding = encode_text(description)
       results = vector_db.query(
           embedding,
           n_results=k,
           include=['metadata', 'distance']
       )
       return process_results(results)
   ```

## 🔧 Core Components & Theory

### Specialist Agent: LLaMA 3.1 Fine-tuning

<p align="center">
<img src="photos/train prompt.png" alt="Training Prompt" width="800">
<br>
<em>LLaMA 3.1 Training Process with QLoRA</em>
</p>

The Specialist Agent is built on a fine-tuned LLaMA 3.1 model, optimized specifically for price prediction tasks. The fine-tuning process employs QLoRA (Quantized Low-Rank Adaptation), a technique that enables efficient model adaptation while maintaining high performance.

**Theory & Implementation:**
1. **QLoRA Fine-tuning**
   - Uses 4-bit quantization to reduce memory footprint
   - Applies low-rank adaptation matrices to key model layers
   - Maintains model quality while reducing training resources
   - Enables efficient deployment on consumer GPUs

2. **Training Process**
   - Custom prompt engineering for price prediction
   - Structured output format for consistent pricing
   - Context window optimization for product descriptions
   - Training on curated e-commerce dataset

```python
class SpecialistAgent(Agent):
    def __init__(self):
        self.model = load_quantized_model(
            "llama-3.1",
            quantization="4bit",
            device_map="auto"
        )
        
    def predict_price(self, description: str) -> float:
        prompt = self.create_price_prompt(description)
        response = self.model.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.7
        )
        return self.extract_price(response)
```

### Frontier Agent: RAG-Enhanced GPT-4

<p align="center">
<img src="photos/4o_mini rag 2 .png" alt="RAG Pipeline" width="800">
<br>
<em>Retrieval Augmented Generation Pipeline</em>
</p>

The Frontier Agent implements a sophisticated RAG (Retrieval Augmented Generation) pipeline that combines GPT-4's reasoning capabilities with a vector database of historical product data.

**Theory & Implementation:**
1. **Vector Similarity Search**
   - Uses Sentence Transformers for embedding generation
   - ChromaDB for efficient similarity matching
   - Cosine similarity for semantic matching
   - Top-k retrieval with dynamic filtering

2. **Context Engineering**
   - Intelligent prompt construction with similar products
   - Price-aware context formatting
   - Confidence scoring for retrieved matches
   - Dynamic context window optimization

```python
def create_context(self, similar_products):
    context = []
    for product in similar_products:
        context.append({
            "description": product.description,
            "price": product.price,
            "similarity": product.score
        })
    return format_context(context)

def predict_with_context(self, description: str, context: str):
    prompt = f"""Given the following similar products and their prices:
    {context}
    
    Estimate the price for: {description}
    Consider the similarities and differences between the products."""
    
    return self.gpt4_client.complete(prompt)
```

### Random Forest Agent: Vector-Based ML

<p align="center">
<img src="photos/sentence transformer array.png" alt="Vector Embeddings" width="800">
<br>
<em>Vector Space Representation of Products</em>
</p>

The Random Forest Agent employs traditional machine learning techniques enhanced with modern NLP embeddings to provide robust price predictions based on historical data.

**Theory & Implementation:**
1. **Feature Engineering**
   - Sentence transformer embeddings (384 dimensions)
   - Product metadata incorporation
   - Categorical feature encoding
   - Numerical feature normalization

2. **Model Architecture**
   - Random Forest with 100 estimators
   - Optimized tree depth for generalization
   - Feature importance analysis
   - Cross-validation for robustness

```python
class RandomForestAgent(Agent):
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2
        )
        
    def create_features(self, description: str):
        # Generate embeddings
        embedding = self.embedder.encode(description)
        
        # Add additional features
        features = np.concatenate([
            embedding,
            self.extract_metadata_features(description)
        ])
        return features
```

### Ensemble Agent: Weighted Prediction Combination

<p align="center">
<img src="photos/ensemble model output.png" alt="Ensemble Architecture" width="800">
<br>
<em>Multi-Model Ensemble Architecture</em>
</p>

The Ensemble Agent combines predictions from all three specialized agents using a sophisticated weighting mechanism that accounts for each model's strengths and confidence levels.

**Theory & Implementation:**
1. **Model Weighting**
   - Dynamic weight assignment
   - Confidence-based adjustment
   - Historical performance tracking
   - Specialized domain expertise

2. **Aggregation Strategy**
   - Weighted average computation
   - Outlier detection and handling
   - Confidence threshold filtering
   - Error analysis and correction

```python
def ensemble_predict(self, description: str):
    predictions = {
        'specialist': self.specialist.predict(description),
        'frontier': self.frontier.predict(description),
        'random_forest': self.rf.predict(description)
    }
    
    confidences = self.calculate_confidences(description)
    return self.weighted_combine(predictions, confidences)
```

### Scanner Agent: Deal Discovery

<p align="center">
<img src="photos/scan agent result.png" alt="Scanner Results" width="800">
<br>
<em>Deal Detection and Analysis Pipeline</em>
</p>

The Scanner Agent continuously monitors various data sources for potential deals, implementing a sophisticated pipeline for deal validation and analysis.

**Theory & Implementation:**
1. **Data Collection**
   - RSS feed monitoring
   - Web scraping integration
   - Real-time price tracking
   - Data normalization

2. **Deal Analysis**
   - Price comparison logic
   - Historical price tracking
   - Discount calculation
   - Deal validation rules

```python
async def process_deals(self):
    deals = await self.fetch_deals()
    for deal in deals:
        if self.is_valid_deal(deal):
            price_estimate = await self.get_price_estimate(deal)
            if self.calculate_discount(deal, price_estimate) > 40:
                await self.notify_deal(deal)
                
def calculate_discount(self, deal, estimated_price):
    return max(0, estimated_price - deal.price)
```

### Planning Agent: Workflow Orchestration

<p align="center">
<img src="photos/planner output.png" alt="Planner Output" width="800">
<br>
<em>Multi-Agent Workflow Orchestration</em>
</p>

The Planning Agent acts as the system's orchestrator, coordinating the activities of all other agents and managing the overall workflow of price prediction and deal discovery.

**Theory & Implementation:**
1. **Workflow Management**
   - Task scheduling and prioritization
   - Agent coordination
   - State management
   - Error handling and recovery

2. **Decision Making**
   - Task decomposition
   - Resource allocation
   - Pipeline optimization
   - Result aggregation

```python
class PlanningAgent(Agent):
    async def execute_workflow(self, task):
        # Create execution plan
        plan = self.create_execution_plan(task)
        
        # Execute steps in sequence
        for step in plan:
            result = await self.execute_step(step)
            self.update_state(result)
            
        return self.summarize_results()
        
    def create_execution_plan(self, task):
        return [
            Step("scan", self.scanner_agent),
            Step("price", self.ensemble_agent),
            Step("analyze", self.analyzer_agent),
            Step("notify", self.notification_agent)
        ]
```

## 🔄 System Workflow

The complete system workflow follows these steps:

1. **Input Processing**
   ```mermaid
   graph TD
   A[User Input] --> B[Planning Agent]
   B --> C[Scanner Agent]
   C --> D[Price Analysis]
   D --> E[Deal Detection]
   E --> F[Notification]
   ```

2. **Price Prediction Flow**
   ```mermaid
   graph LR
   A[Product] --> B[Vector Embedding]
   B --> C[Specialist Agent]
   B --> D[Frontier Agent]
   B --> E[Random Forest]
   C & D & E --> F[Ensemble Agent]
   F --> G[Final Price]
   ```

3. **Deal Processing**
   ```mermaid
   graph TD
   A[New Deal] --> B[Scanner]
   B --> C[Price Estimation]
   C --> D[Discount Calculation]
   D --> E{Threshold Check}
   E -->|Above $40| F[Notify]
   E -->|Below $40| G[Skip]
   ```

## 🚀 Setup & Installation

1. **Environment Setup**
```bash
conda create -n agentic-ai python=3.8
conda activate agentic-ai
pip install -r requirements.txt
```

2. **Model Deployment**
```bash
   modal deploy pricer_service.py
   modal deploy agent_framework.py
   ```

## 📊 Usage Guide

1. **Basic Usage**
   ```python
   from agents import EnsembleAgent
   
   agent = EnsembleAgent()
   price = agent.predict("iPhone 13 Pro Max 256GB")
   print(f"Estimated Price: ${price:.2f}")
   ```

2. **Deal Monitoring**
```python
   from agents import ScannerAgent
   
   scanner = ScannerAgent()
   scanner.start_monitoring(
       threshold=40,
       interval="1h"
   )
   ```

---

<div align="center">

### Made with ❤️ by Aman

<p align="center">
<a href="https://github.com/its-amann">GitHub</a> •
<a href="https://linkedin.com/in/aman-agnihotri004">LinkedIn</a>
</p>

</div>
