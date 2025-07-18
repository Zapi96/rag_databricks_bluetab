# RAG Databricks Bluetab - Enhanced Pipeline

## 🚀 Overview

This project implements a complete **Retrieval Augmented Generation (RAG)** pipeline on Databricks, designed specifically for Bluetab's internal knowledge base. The pipeline has been enhanced with comprehensive parameterization, MLflow experiment tracking, and asset bundle support for enterprise deployment.

## ✨ Key Features

### 🔧 Enhanced Capabilities
- **Parameterized Notebooks**: All notebooks now include configurable widgets for easy customization
- **Unified MLflow Experiment**: Single experiment tracking across all pipeline steps
- **Asset Bundle Support**: Ready for deployment using Databricks Asset Bundles
- **Cross-Notebook Variable Sharing**: Centralized configuration management
- **Environment Support**: Dev/Test/Prod environment configurations
- **Comprehensive Logging**: Detailed step-by-step logging and monitoring

### 🏗️ Architecture Components
- **PDF Processing**: Incremental text extraction and chunking
- **Embedding Model**: Custom DistilBERT-based embedding model
- **LLM Integration**: FLAN-T5 model for response generation
- **Vector Search**: Databricks Vector Search for similarity retrieval
- **Serving Endpoints**: Real-time inference capabilities
- **Chat Interface**: LangChain-compatible chat API

## 📁 Project Structure

```
RAG Chatbot/
├── 00 Configuration and Utils.py      # 🔧 Centralized configuration & utilities
├── 01 Create needed tables.py         # 📊 Database table creation
├── 02 Incremental PDF to docs_text.py # 📄 PDF processing pipeline
├── 03 Register embedding model.ipynb  # 🧠 Embedding model registration
├── 04 Create Embedding Serving Endpoint.ipynb # 🌐 Embedding endpoint
├── 05 Register LLM model.ipynb        # 🤖 LLM model registration
├── 06 Create LLM Serving Endpoint.ipynb # 🌐 LLM endpoint creation
├── 07 Crear Embbedings.py             # 📈 Embedding generation
├── 08 Create index.ipynb              # 🔍 Vector search index
├── 09 Chatbot.ipynb                   # 💬 RAG model assembly
├── 10 Create RAG Serving Endpoint.ipynb # 🌐 RAG endpoint
└── 11 Chatbot GUI.py                  # 🖥️ User interface
databricks.yml                         # 📦 Asset bundle configuration
```

## 🚀 Quick Start

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- MLflow Model Registry access
- Vector Search capability
- Appropriate compute resources

## 📦 Dependencies Management

This project uses a sophisticated requirements management system:

### 📋 Requirements Files Structure
- **`requirements.txt`** - General dependencies for all notebooks
- **`requirements_embedding_model.txt`** - Specific dependencies for embedding model registration
- **`requirements_llm_model.txt`** - Specific dependencies for LLM model registration

### 🔧 Installation
```bash
# For general development
%pip install -r requirements.txt

# Model-specific requirements are automatically used during model registration
```

### ✅ Benefits
- **Reproducible deployments** with exact dependency versions
- **Isolated dependencies** for different model types
- **Optimized serving environments** with minimal dependencies
- **Easy maintenance** and version control

📖 **See [REQUIREMENTS_MANAGEMENT.md](REQUIREMENTS_MANAGEMENT.md) for detailed documentation**

### 1. Configuration Setup
All notebooks now use the centralized configuration system. Start by running:

```python
%run "./00 Configuration and Utils"
```

This sets up:
- 📋 Widget-based parameter configuration
- 🧪 MLflow experiment management
- 🔧 Utility functions for logging and error handling
- 🌍 Environment-specific variables

### 2. Pipeline Execution

#### Option A: Manual Step-by-Step
Execute notebooks in sequence:

1. **00 Configuration and Utils** - Set up shared configuration
2. **01 Create needed tables** - Initialize database tables
3. **02 Incremental PDF to docs_text** - Process PDF documents
4. **03 Register embedding model** - Register embedding model
5. **04 Create Embedding Serving Endpoint** - Deploy embedding endpoint
6. **05 Register LLM model** - Register language model
7. **06 Create LLM Serving Endpoint** - Deploy LLM endpoint
8. **07 Crear Embeddings** - Generate embeddings for documents
9. **08 Create index** - Create vector search index
10. **09 Chatbot** - Assemble RAG pipeline
11. **10 Create RAG Serving Endpoint** - Deploy complete RAG system

#### Option B: Asset Bundle Deployment
For production deployment, use Databricks Asset Bundles:

```bash
# Deploy to development environment
databricks bundle deploy --target dev

# Deploy to production environment
databricks bundle deploy --target prod

# Run the complete pipeline
databricks bundle run rag_pipeline_job --target prod
```

## ⚙️ Configuration Parameters

### Core Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `catalog_name` | Unity Catalog name | `bluetab` |
| `schema_name` | Schema name | `rag` |
| `environment` | Environment (dev/test/prod) | `dev` |

### Model Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_model_base` | Base embedding model | `distilbert-base-uncased` |
| `llm_base_model` | Base LLM model | `google/flan-t5-base` |
| `embedding_dim` | Embedding dimensions | `768` |
| `max_length` | Max sequence length | `512` |

### Processing Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Text chunk size | `1000` |
| `chunk_overlap` | Chunk overlap | `200` |
| `batch_size` | Processing batch size | `32` |

### Endpoint Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `workload_size` | Endpoint workload size | `Small` |
| `scale_to_zero` | Enable auto-scaling | `true` |

## 🧪 MLflow Integration

### Unified Experiment Tracking
All pipeline steps are tracked under a single MLflow experiment: `/Shared/RAG_Databricks_Bluetab_Pipeline`

### Run Naming Convention
- **01_Create_Tables_{environment}**
- **02_PDF_Processing_{environment}**
- **03_Register_Embedding_Model_{environment}**
- **04_Create_Embedding_Endpoint_{environment}**
- And so on...

### Tracked Metrics
- Processing statistics (files processed, chunks created)
- Model performance metrics
- Endpoint deployment status
- Error rates and processing times

## 🌍 Environment Management

### Development Environment
```yaml
catalog_name: bluetab
schema_name: rag_dev
environment: dev
workload_size: Small
```

### Production Environment
```yaml
catalog_name: bluetab
schema_name: rag_prod
environment: prod
workload_size: Large
chunk_size: 1500
batch_size: 64
```

## 📊 Monitoring and Logging

### Built-in Logging
Every notebook includes comprehensive logging:
- Step-by-step progress tracking
- Error handling and recovery
- Performance metrics
- MLflow parameter and metric logging

### Example Log Output
```
[2025-01-12 10:30:15] Step: pdf_processing - Status: started - Details: Starting PDF processing pipeline
[2025-01-12 10:30:45] Step: scan_pdf_volume - Status: success - Details: Found 25 PDF files
[2025-01-12 10:32:10] Step: text_extraction - Status: completed - Details: Extracted 150,000 characters
```

## 🔧 Customization Guide

### Adding New Parameters
1. Add widget in `00 Configuration and Utils.py`:
```python
dbutils.widgets.text("new_parameter", "default_value", "Description")
NEW_PARAMETER = dbutils.widgets.get("new_parameter")
```

2. Use in other notebooks:
```python
%run "./00 Configuration and Utils"
# NEW_PARAMETER is now available
```

### Environment-Specific Configuration
Modify `databricks.yml` to add environment-specific variables:

```yaml
targets:
  custom_env:
    variables:
      environment: "custom"
      catalog_name: "custom_catalog"
      chunk_size: 2000
```

### Custom Model Integration
Replace model parameters in configuration:
```python
dbutils.widgets.text("embedding_model_base", "sentence-transformers/all-MiniLM-L6-v2", "Embedding Model")
```

## 🚨 Troubleshooting

### Common Issues

1. **Widget Values Not Persisting**
   - Always run `00 Configuration and Utils` first
   - Check that widgets are properly defined

2. **MLflow Experiment Not Found**
   - Verify experiment name in configuration
   - Check workspace permissions

3. **Endpoint Creation Failures**
   - Verify model is registered
   - Check endpoint quotas
   - Validate permissions

4. **Asset Bundle Deployment Issues**
   - Verify `databricks.yml` syntax
   - Check workspace configuration
   - Validate variable references

### Debugging Tips
- Use the built-in logging system
- Check MLflow runs for detailed metrics
- Verify table and model existence before dependent steps
- Monitor endpoint status in Databricks UI

## 🛠️ Development Workflow

### Local Development
1. Set up development environment in `databricks.yml`
2. Use small datasets for testing
3. Monitor logs and MLflow experiments
4. Test individual notebooks before pipeline execution

### Production Deployment
1. Validate in test environment first
2. Use asset bundles for consistent deployment
3. Monitor production metrics
4. Set up automated testing

## 📚 Additional Resources

- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
- [MLflow Model Registry](https://docs.databricks.com/mlflow/model-registry.html)
- [Databricks Vector Search](https://docs.databricks.com/vector-search/index.html)
- [LangChain Integration](https://docs.databricks.com/large-language-models/langchain.html)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive logging to new features
4. Update configuration parameters
5. Test with multiple environments
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for Bluetab's internal knowledge management system**