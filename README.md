<div align="center">

# 🤖 Agentic AI: Multi-Agent Price Retirival System

<p align="center">
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Framework-Modal-green.svg" alt="Framework">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A sophisticated multi-agent system leveraging serverless architecture and ensemble learning for accurate price predictions.

</div>

## 🌟 Introduction

Agentic AI is an advanced price prediction system that combines multiple AI agents working in harmony through a serverless architecture. The system employs various AI techniques including:

- 🤖 Specialized LLM agents
- 🌐 RAG (Retrieval Augmented Generation)
- 🌲 Random Forest ML models
- 🤝 Ensemble learning techniques

The system runs on Modal's serverless platform, allowing for efficient scaling and deployment without managing infrastructure.

## 🏛️ Architecture
![alt text](<photos/agent workflow.png>)

### Data Layer
- **ChromaDB Vector Store**  
  Stores product descriptions as vector embeddings using `all-MiniLM-L6-v2` sentence transformer for semantic similarity searches.  
  Enables real-time retrieval of comparable products during price estimation.

### Agent Framework
# Frontier Agent

The Frontier Agent is an intelligent pricing estimation system leveraging Retrieval-Augmented Generation (RAG) and OpenAI's GPT-4 to predict product prices based on contextual similarities from a dataset.

## Features

- **Retrieval-Augmented Generation (RAG) Pipeline**: Integrates vector similarity search with generative AI for informed price predictions.
- **Semantic Similarity Search**: Uses `all-MiniLM-L6-v2` embeddings to find products with similar descriptions.
- **Contextual Prompting**: Constructs dynamic prompts using similar items and their prices to guide GPT-4's estimation.
- **Deterministic Output**: Ensures reproducible results with fixed random seeds and structured responses.

## How It Works

1. **Similarity Search**  
   When provided with a product description, the agent:
   - Encodes the description into a vector using `SentenceTransformer`.
   - Queries a ChromaDB collection to retrieve the **5 most similar items** and their prices.

2. **Prompt Engineering**  
   Constructs a context-rich prompt containing:
   - Descriptions and prices of similar items.
   - A direct query asking for the estimated price of the target item.

3. **OpenAI Integration**  
   Sends the prompt to `gpt-4o-mini` with constraints:
   - Responses limited to numeric values (e.g., `$12.34`).
   - Parses the output to extract the predicted price.

## Dependencies

- **Libraries**: `openai`, `chromadb`, `sentence-transformers`, `datasets`
- **Services**: OpenAI API key (required for GPT-4 access)
- **Data**: ChromaDB collection with pre-stored item embeddings and metadata (prices).

## Usage Example

```python
# Initialize ChromaDB collection (assumes pre-populated data)
collection = chromadb.get_collection("product_embeddings")

# Create Frontier Agent instance
agent = FrontierAgent(collection)
   
```
> Results 
![alt text](<photos/gpt_4o_mini rag .png>)

# Specialist Agent 

## Overview
The Specialist Agent is a powerful AI-driven pricing system that leverages a fine-tuned Llama 3.1 model to predict product prices based on item descriptions. Built for scalability and efficiency, it utilizes remote inference via Modal to deliver accurate price estimations in production environments.

## Key Features
- **Fine-Tuned LLM**: Uses a specialized 8-parameter "wheeling" architecture variant of Llama 3.1
- **Efficient Quantization**: 4-bit model format optimized for memory efficiency
- **QLoRA Fine-Tuning**: Trained using QLoRA (Quantized Low-Rank Adaptation) technique
- **Scalable Inference**: Cloud-based execution through Modal for enterprise-grade performance
- **Large Training Data**: Fine-tuned on wall* market data records for robust predictions

## Model Specifications
| Attribute          | Details                          |
|---------------------|----------------------------------|
| Base Model          | Llama 3.1                        |
| Architecture Variant| X-8 Wheeling Parameters          |
| Precision           | 4-bit Quantized                  |
| Training Technique  | QLoRA Fine-Tuning                |
| Inference Platform  | Modal                            |

*\*Wall market data - comprehensive product pricing dataset*

## Usage Example
```python
from agents.specialist_agent import SpecialistAgent

agent = SpecialistAgent()
price_prediction = agent.price("Brand new wireless Bluetooth headphones with ANC")
print(f"Predicted price: ${price_prediction:.2f}") 

# Ensemble Pricing Agent

The **Ensemble Agent** is a robust pricing model that combines predictions from three specialized agents to deliver accurate price estimations for products. It leverages a weighted ensemble approach (via linear regression) to aggregate insights from diverse methodologies, ensuring high reliability and adaptability.

## Key Components

### 1. Specialist Agent (`SpecialistAgent`)
- **Architecture**: Fine-tuned LLaMA 3.1 model
- **Training**: Utilizes Q-Lora (Quantized Low-Rank Adaptation) for parameter-efficient tuning
- **Purpose**: Deep semantic understanding of product descriptions

### 2. Random Forest Agent (`RandomForestAgent`)
- **Feature Engineering**: Employs Sentence Transformers to generate `vector-amperex` embeddings
- **Model**: Random Forest regressor trained on product description embeddings
- **Strength**: Captures complex patterns in textual data through tree-based ensemble learning

### 3. Frontier Agent (`FrontierAgent`)
- **Retrieval Pipeline**: GPT-4/Mini-powered similarity search
- **Methodology**:
  1. Finds top 5 most similar products using semantic search
  2. Predicts price based on historical prices of matched items
- **Unique Value**: Context-aware predictions using comparable products

## Ensemble Strategy
- **Aggregation Model**: Linear Regression (`ensemble_model.pkl`)
- **Features Combined**:
  - Raw predictions from all three agents
  - Minimum predicted value across models
  - Maximum predicted value across models
- **Advantage**: Dynamically weights each model's contribution based on historical performance

## Usage Example
```python
from agents.ensemble_agent import EnsembleAgent

# Initialize with product collection data
agent = EnsembleAgent(collection="your_product_dataset")

# Get price prediction
description = "Wireless Bluetooth Headphones with Noise Cancellation"
predicted_price = agent.price(description)
print(f"Estimated price: ${predicted_price:.2f}")

### Specialist Agent Architecture

The Specialist Agent is a core component of the system that leverages a fine-tuned LLM deployed on Modal's serverless infrastructure for precise price predictions. Here's a detailed breakdown of its architecture:

1. **Model Architecture**
   - Uses Meta-Llama-3.1-8B as the base model
   - Implements 4-bit quantization for efficiency
   - Utilizes PEFT (Parameter-Efficient Fine-Tuning)
   - Runs on T4 GPU infrastructure

2. **Key Components**
   ```python
   class SpecialistAgent(Agent):
       name = "Specialist Agent"
       color = Agent.RED

       def price(self, description: str) -> float:
           return self.pricer.price.remote(description)
   ```

3. **Serverless Implementation**
   - Deployed as modal.Cls for serverless execution
   - Uses BitsAndBytes for 4-bit quantization
   - Implements efficient model loading and caching
   ```python
   quant_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_quant_type="nf4"
   )
   ```

4. **Interaction Flow**
   - Receives product descriptions as input
   - Formats prompts with specific question structure
   - Processes through the fine-tuned model
   - Extracts numerical price predictions
   - Returns floating-point price estimates

5. **Integration Points**
   - Primary component of the Ensemble Agent
   - Interfaces with Modal's serverless platform
   - Contributes specialized predictions to ensemble
   - Works alongside Frontier and Random Forest agents

6. **Performance Optimization**
   - Uses efficient model quantization
   - Implements caching for repeated queries
   - Optimizes prompt formatting for price extraction
   - Maintains lightweight agent footprint

7. **Technical Details**
   - Model: Meta-Llama-3.1-8B
   - Quantization: 4-bit with double quantization
   - Infrastructure: Modal with T4 GPU
   - Response Format: Direct price predictions
   - Integration: Remote procedure calls via Modal

### RAG Pipeline Implementation

### Planning Agent Architecture

The Planning Agent serves as the strategic orchestrator of the entire system, coordinating multiple specialized agents to identify and evaluate price opportunities. Here's a detailed examination of its architecture and functionality:

1. **Orchestration Core**
   - Coordinates three critical agents:
     - Scanner Agent: Deal discovery
     - Ensemble Agent: Price estimation
     - Messaging Agent: Notification delivery
   - Implements workflow management
   - Maintains deal memory system

2. **Workflow Architecture**
   - **Phase 1: Deal Discovery**
     - Initiates Scanner Agent
     - Processes RSS feed data
     - Filters duplicate deals
     - Validates deal quality
   
   - **Phase 2: Price Analysis**
     - Triggers Ensemble Agent
     - Obtains market price estimates
     - Calculates discount opportunities
     - Sorts by potential savings
   
   - **Phase 3: Notification**
     - Evaluates deal thresholds
     - Triggers messaging system
     - Tracks processed deals
     - Updates memory state

3. **Opportunity Management**
   - Defines minimum discount threshold ($50)
   - Implements opportunity scoring
   - Maintains historical tracking
   - Prevents duplicate processing
   - Ensures quality control

4. **Memory System**
   - Tracks processed URLs
   - Prevents redundant analysis
   - Maintains opportunity history
   - Enables trend analysis
   - Supports decision making

5. **Integration Workflow**
   ```
   User Request
        ↓
   Planning Agent
        ↓
   1. Scanner Agent (RSS Feeds)
        ↓
   2. Ensemble Agent (Price Analysis)
        ↓
   3. Opportunity Evaluation
        ↓
   4. Messaging Agent (if threshold met)
        ↓
   Memory Update
   ```

6. **Decision Making**
   - Evaluates deal quality
   - Assesses price differentials
   - Applies threshold filtering
   - Prioritizes opportunities
   - Triggers notifications

7. **Optimization Features**
   - Batch processing of deals
   - Efficient memory management
   - Smart notification throttling
   - Resource optimization
   - Performance monitoring

8. **Technical Implementation**
   - Agent initialization:
     ```python
     def __init__(self, collection):
         self.scanner = ScannerAgent()
         self.ensemble = EnsembleAgent(collection)
         self.messenger = MessagingAgent()
     ```
   
   - Opportunity processing:
     ```python
     def run(self, deal: Deal) -> Opportunity:
         estimate = self.ensemble.price(deal.product_description)
         discount = estimate - deal.price
         return Opportunity(deal=deal, estimate=estimate, discount=discount)
     ```

9. **System Integration**
   - Interfaces with all agent types
   - Manages deal pipeline flow
   - Coordinates parallel processing
   - Handles error conditions
   - Ensures system reliability

10. **Performance Monitoring**
    - Deal processing metrics
    - Success rate tracking
    - Response time monitoring
    - Memory usage optimization
    - System health checks

11. **Security & Validation**
    - Input validation
    - Deal authenticity checks
    - Secure API communication
    - Data integrity assurance
    - Error handling protocols

12. **Notification System**
    - Multi-channel support
      - Push notifications
      - SMS messaging
    - Customizable alerts
    - Priority-based delivery
    - Rate limiting
    - Delivery confirmation

This architectural design enables:
- Efficient deal processing
- Accurate price analysis
- Timely notifications
- Scalable operations
- Reliable performance
- System maintainability

### Structured Output Processing

The system implements a sophisticated structured output processing pipeline that transforms raw RSS feed data into structured deal information using Pydantic models and OpenAI's function calling capabilities. Here's a detailed overview:

1. **Data Model Architecture**
   - Utilizes Pydantic for robust data validation
   - Core models include:
     - ScrapedDeal: Raw RSS feed data
     - Deal: Processed product information
     - DealSelection: Curated list of deals
     - Opportunity: Validated price opportunities

2. **Scanner Agent Implementation**
   - Processes RSS feeds for product deals
   - Enforces structured responses via OpenAI
   - Key features:
     - JSON response enforcement
     - Price validation
     - Duplicate detection
     - Deal quality assessment
     - Memory-based filtering

3. **Structured Output Flow**
   - RSS Feed Scraping → Raw Data
   - Data Validation & Cleaning
   - LLM-based Information Extraction
   - Pydantic Model Validation
   - Memory Storage & Deduplication

4. **Memory Management**
   - Persistent JSON storage
   - Historical deal tracking
   - Duplicate prevention
   - Price opportunity tracking
   - Deal progression monitoring

5. **LLM Integration**
   - Strict JSON response formatting
   - Detailed product description extraction
   - Price validation and normalization
   - URL and metadata preservation
   - Quality-based deal filtering

6. **Deal Processing Pipeline**
   - Raw deal collection from RSS
   - Product description enhancement
   - Price extraction and validation
   - Similar product comparison
   - Opportunity assessment
   - Memory state updates

7. **Data Validation Features**
   - Price sanity checks
   - Description quality assessment
   - URL uniqueness verification
   - Category classification
   - Deal relevance scoring

8. **System Integration**
   - Connects with Ensemble Agent
   - Feeds into pricing pipeline
   - Updates memory storage
   - Triggers notifications
   - Maintains audit trail

9. **Quality Assurance**
   - Description completeness checks
   - Price confidence scoring
   - Deal relevance verification
   - Duplicate prevention
   - Data consistency validation

10. **Performance Considerations**
    - Efficient RSS processing
    - Smart memory management
    - Rapid validation checks
    - Optimized storage format
    - Quick retrieval capabilities

11. **Technical Architecture**
    - Models: Pydantic BaseModel
    - Storage: JSON persistence
    - Processing: OpenAI function calling
    - Integration: Agent framework
    - Memory: Local file system

12. **Security & Validation**
    - Input sanitization
    - Price range validation
    - URL verification
    - Data integrity checks
    - Access control

This structured approach ensures:
- Consistent data formatting
- Reliable price extraction
- Quality deal selection
- Efficient processing
- System reliability
- Easy maintenance
- Scalable architecture

The RAG (Retrieval Augmented Generation) component is a sophisticated system that combines vector similarity search with LLM-based price estimation. Here's a detailed breakdown of its architecture:

1. **Vector Database Architecture**
   - Uses ChromaDB as persistent vector store
   - Stores 400,000+ product embeddings
   - Each record contains:
     ```python
     {
       "document": "product description",
       "embedding": "384-dimensional vector",
       "metadata": {
         "category": "product category",
         "price": "actual price"
       }
     }
     ```

2. **Embedding Model**
   - Uses SentenceTransformer's all-MiniLM-L6-v2
   - 384-dimensional dense vector space
   - Local execution for faster processing
   ```python
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   vector = model.encode([product_description])
   ```

3. **RAG Pipeline Flow**
   ```python
   class FrontierAgent(Agent):
       def price(self, description: str) -> float:
           # 1. Find similar products
           documents, prices = self.find_similars(description)
           
           # 2. Create context with similar products
           context = self.make_context(documents, prices)
           
           # 3. Query LLM with context
           response = self.openai.chat.completions.create(
               model="gpt-4o-mini",
               messages=self.messages_for(description, documents, prices)
           )
           
           # 4. Extract price prediction
           return self.get_price(response.choices[0].message.content)
   ```

4. **Vector Search Implementation**
   - Performs nearest neighbor search
   - Returns top 5 similar products
   - Includes prices for context
   ```python
   def find_similars(self, description: str):
       vector = self.model.encode([description])
       results = self.collection.query(
           query_embeddings=vector.astype(float).tolist(),
           n_results=5
       )
       return results['documents'][0][:], [m['price'] for m in results['metadatas'][0][:]]
   ```

5. **Database Population Process**
   - Batch processing of product data
   - Efficient vector encoding
   - Metadata inclusion for each product
   ```python
   for batch in chunks(products, 1000):
       documents = [item.description for item in batch]
       vectors = model.encode(documents).astype(float).tolist()
       metadatas = [{
           "category": item.category,
           "price": item.price
       } for item in batch]
       collection.add(
           documents=documents,
           embeddings=vectors,
           metadatas=metadatas
       )
   ```

6. **Integration Points**
   - Part of the Frontier Agent
   - Contributes to Ensemble predictions
   - Uses GPT-4o-mini for price estimation
   - Interfaces with ChromaDB for vector search

7. **Performance Features**
   - Local embedding generation
   - Efficient batch processing
   - Persistent vector storage
   - Quick similarity search
   - Context-aware price predictions

8. **Technical Details**
   - Database: ChromaDB
   - Embedding Model: all-MiniLM-L6-v2
   - Vector Dimension: 384
   - LLM: GPT-4o-mini
   - Similar Products: Top 5 matches
   - Storage: Persistent disk-based


### Ensemble Learning

```python
class EnsembleAgent(Agent):
    def price(self, description: str) -> float:
        specialist = self.specialist.price(description)
        frontier = self.frontier.price(description)
        random_forest = self.random_forest.price(description)
        
        X = pd.DataFrame({
            'Specialist': [specialist],
            'Frontier': [frontier],
            'RandomForest': [random_forest],
            'Min': [min(specialist, frontier, random_forest)],
            'Max': [max(specialist, frontier, random_forest)],
        })
        
        return self.model.predict(X)[0]
```

## 🚀 Setup

1. **Environment Setup**
```bash
conda create -n agentic-ai python=3.8
conda activate agentic-ai
pip install -r requirements.txt
```

2. **Modal Configuration**
```bash
modal setup
modal token new  # For Windows users
```

3. **Deploy Serverless Components**
```bash
modal deploy pricer_service
modal deploy pricer_service2
```

4. **Configure OpenAI API**
- Set up OpenAI API key in environment variables
- Configure HuggingFace token in Modal secrets

## 📊 Usage

```python
from agents.ensemble_agent import EnsembleAgent

# Initialize the ensemble agent
agent = EnsembleAgent(collection)

# Get price prediction
price = agent.price("iPad Pro 2nd generation")
print(f"Predicted Price: ${price:.2f}")
```

## 🔧 Key Features

- **Serverless Architecture**: Runs on Modal's cloud platform
- **Multi-Agent System**: Combines multiple specialized agents
- **RAG Integration**: Uses ChromaDB for similar product lookup
- **Ensemble Learning**: Weighted combination of predictions
- **Real-time Processing**: Fast and scalable predictions

## 🛠️ Technologies Used

- **Modal**: Serverless deployment platform
- **OpenAI GPT**: Large language model integration
- **ChromaDB**: Vector database for RAG
- **Sentence Transformers**: Text embedding
- **scikit-learn**: Random Forest and ensemble models
- **pandas**: Data manipulation
- **Python**: Core development

## 📈 Performance

The system achieves high accuracy through:
- Ensemble learning combining multiple models
- RAG-enhanced context awareness
- Specialized agent expertise
- ML model robustness

## 🔐 Security

- Secure API key management through Modal secrets
- Serverless deployment reducing attack surface
- Isolated agent execution environments

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
