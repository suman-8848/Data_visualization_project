# System Architecture: LLM Reasoning Visualization

## Overview

This document describes the complete system architecture, data flow, and component interactions for the Interactive Visualization of LLM Reasoning project.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Web Browser - D3.js)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Attention   │  │  Knowledge   │  │    Token     │         │
│  │   Heatmap    │  │    Graph     │  │   Sequence   │         │
│  │              │  │              │  │              │         │
│  │  (Matrix)    │  │ (Force-Dir)  │  │   (Linear)   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                   ┌────────▼────────┐                          │
│                   │ Sync Controller │                          │
│                   │ (Brushing &     │                          │
│                   │  Linking)       │                          │
│                   └────────┬────────┘                          │
└────────────────────────────┼───────────────────────────────────┘
                             │ HTTP/JSON
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                      REST API LAYER                            │
│                     (FastAPI Backend)                          │
├────────────────────────────────────────────────────────────────┤
│  POST /query                                                   │
│  GET  /health                                                  │
│  GET  /sample-questions                                        │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                   DATA PROCESSING LAYER                        │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │    Model     │  │  Knowledge   │  │    Graph     │        │
│  │   Handler    │  │  Extractor   │  │   Builder    │        │
│  │              │  │              │  │              │        │
│  │ SmolLM2-360M │  │    spaCy     │  │  NetworkX    │        │
│  │ + Attention  │  │  NER + Rel   │  │  + Layout    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. Frontend Layer (JavaScript/D3.js)

#### 1.1 Main Application Controller (`main.js`)
**Responsibilities**:
- Coordinate all frontend components
- Handle user input and API communication
- Manage application state
- Update UI based on API responses

**Key Methods**:
- `handleSubmit()`: Process user queries
- `queryAPI()`: Communicate with backend
- `displayResults()`: Update UI with results
- `initializeVisualizations()`: Create visualization instances

#### 1.2 Attention Visualization (`attention-viz.js`)
**Responsibilities**:
- Render attention weights as heatmap
- Handle zoom/pan interactions
- Highlight token relationships
- Support layer/head selection

**Data Structure**:
```javascript
{
  attention_mean: [[float]],      // Averaged attention matrix
  attention_by_layer: [[[float]]], // Per-layer attention
  attention_by_head: [[[float]]],  // Per-head attention
  tokens: [string],                // Token sequence
  num_layers: int,
  num_heads: int
}
```

**Visual Encoding**:
- X-axis: Target tokens
- Y-axis: Source tokens
- Color: Attention weight (blue scale, 0-1)
- Cell size: Adaptive based on sequence length

#### 1.3 Graph Visualization (`graph-viz.js`)
**Responsibilities**:
- Render knowledge graph with force-directed layout
- Handle node/edge interactions
- Display entity and relationship information
- Compute and visualize graph metrics

**Data Structure**:
```javascript
{
  nodes: [{
    id: string,
    label: string,
    type: string,
    x: float, y: float,
    degree: int,
    centrality: float,
    pagerank: float
  }],
  edges: [{
    source: string,
    target: string,
    relation: string,
    confidence: float
  }]
}
```

**Visual Encoding**:
- Node position: Force-directed layout
- Node size: Degree centrality
- Node color: Entity type
- Edge thickness: Relationship confidence
- Edge direction: Arrows

#### 1.4 Synchronization Controller (`sync-controller.js`)
**Responsibilities**:
- Maintain token-to-entity mappings
- Coordinate highlighting across views
- Implement brushing & linking
- Handle cross-view interactions

**Key Data Structures**:
```javascript
tokenToEntities: {
  [tokenIndex]: [entityId1, entityId2, ...]
}

entityToTokens: {
  [entityId]: {
    start: int,
    end: int,
    tokens: [int]
  }
}
```

**Interaction Flows**:
1. Node click → Highlight tokens in attention view
2. Attention cell click → Highlight entities in graph
3. Token click → Highlight in both views

### 2. Backend Layer (Python/FastAPI)

#### 2.1 API Server (`main.py`)
**Responsibilities**:
- Handle HTTP requests
- Coordinate backend components
- Format responses
- Error handling

**Endpoints**:
```python
POST /query
  Request: { question: str, context?: str, max_length?: int }
  Response: {
    question: str,
    answer: str,
    tokens: [str],
    attention: AttentionData,
    knowledge_graph: GraphData,
    metadata: MetadataDict
  }

GET /health
  Response: { model_loaded: bool, ... }

GET /sample-questions
  Response: { samples: [QuestionDict] }
```

#### 2.2 Model Handler (`model_handler.py`)
**Responsibilities**:
- Load and manage LLM
- Generate text with attention extraction
- Process attention tensors
- Aggregate attention across layers/heads

**Key Methods**:
```python
generate_with_attention(question, context, max_length)
  → { answer, tokens, attention, prompt_length }

_process_attention(attentions, tokens, prompt_length)
  → { attention_mean, attention_by_layer, attention_by_head, ... }
```

**Attention Processing Pipeline**:
1. Extract raw attention tensors from model
2. Convert to numpy arrays
3. Aggregate across layers (mean)
4. Aggregate across heads (mean)
5. Create multiple views for exploration

#### 2.3 Knowledge Extractor (`knowledge_extractor.py`)
**Responsibilities**:
- Extract named entities using spaCy
- Identify relationships via dependency parsing
- Extract noun chunks as concepts
- Map entities to token positions

**Key Methods**:
```python
extract_entities(text)
  → [{ id, text, label, start, end, start_token, end_token }]

extract_relationships(text, entities)
  → [{ id, source, target, relation, confidence }]
```

**Entity Types**:
- PERSON: People, characters
- GPE/LOC: Locations, places
- ORG: Organizations, companies
- CONCEPT: Abstract concepts (noun chunks)
- DATE: Temporal expressions
- PRODUCT: Products, objects

**Relationship Extraction**:
- Verb-based: Subject-Verb-Object patterns
- Co-occurrence: Entities in same sentence
- Dependency-based: Using spaCy's dependency parser

#### 2.4 Graph Builder (`graph_builder.py`)
**Responsibilities**:
- Construct NetworkX graph from entities/relationships
- Compute graph metrics (centrality, PageRank)
- Calculate force-directed layout
- Prepare visualization-ready data

**Key Methods**:
```python
build_graph(entities, relationships, tokens, attention)
  → { nodes, edges, metrics }

_compute_layout()
  → { [nodeId]: (x, y) }

_compute_metrics()
  → { degree_centrality, betweenness, pagerank, ... }
```

**Graph Metrics**:
- Degree centrality: Node importance by connections
- Betweenness centrality: Node importance in paths
- PageRank: Iterative importance measure
- Graph density: Overall connectivity

## Data Flow Pipeline

### Query Processing Flow

```
1. User Input
   ↓
   Question: "What is the capital of France?"
   
2. API Request
   ↓
   POST /query { question: "..." }
   
3. Model Inference
   ↓
   SmolLM2-360M generates:
   - Answer: "The capital of France is Paris."
   - Tokens: ["What", "is", "the", "capital", ...]
   - Attention: [12 layers × 9 heads × seq × seq]
   
4. Attention Processing
   ↓
   Aggregate attention:
   - Mean across all layers/heads
   - Per-layer aggregation
   - Per-head aggregation
   
5. Entity Extraction
   ↓
   spaCy identifies:
   - "France" → GPE (location)
   - "Paris" → GPE (location)
   - "capital" → CONCEPT
   
6. Relationship Extraction
   ↓
   Dependency parsing finds:
   - (Paris, is, capital)
   - (capital, of, France)
   
7. Graph Construction
   ↓
   NetworkX builds:
   - Nodes: France, Paris, capital
   - Edges: Paris→capital, capital→France
   - Metrics: centrality, PageRank
   - Layout: force-directed positions
   
8. Response Assembly
   ↓
   JSON response with all data
   
9. Frontend Rendering
   ↓
   - Attention heatmap displays token-to-token weights
   - Knowledge graph shows entities and relationships
   - Token sequence enables selection
   - Synchronization controller links views
   
10. User Interaction
    ↓
    - Click "Paris" node → highlights tokens in attention
    - Click attention cell → highlights related entities
    - Explore reasoning patterns
```

## Synchronization Mechanism

### Token-Entity Mapping

The synchronization system maintains bidirectional mappings:

```javascript
// Forward mapping: token index → entity IDs
tokenToEntities[5] = ["entity_0", "entity_1"]

// Reverse mapping: entity ID → token range
entityToTokens["entity_0"] = {
  start: 5,
  end: 6,
  tokens: [5]
}
```

### Interaction Scenarios

#### Scenario 1: Node Click in Graph
```
User clicks "Paris" node
  ↓
Graph highlights node
  ↓
Sync controller finds tokens [8, 9]
  ↓
Attention view highlights row/column 8-9
  ↓
Token view highlights tokens 8-9
```

#### Scenario 2: Attention Cell Click
```
User clicks cell (5, 8)
  ↓
Attention view highlights cell
  ↓
Sync controller finds entities for tokens 5 and 8
  ↓
Graph highlights "France" and "Paris" nodes
  ↓
Token view highlights tokens 5 and 8
```

#### Scenario 3: Token Click
```
User clicks token 5
  ↓
Token view highlights token
  ↓
Sync controller finds entity "France"
  ↓
Graph highlights "France" node
  ↓
Attention view highlights row/column 5
```

## Performance Considerations

### Backend Optimization
- **Model Loading**: Load once on startup, reuse for all requests
- **Attention Processing**: Use numpy for efficient array operations
- **Graph Computation**: Cache layout for repeated queries
- **API Response**: Compress large attention matrices

### Frontend Optimization
- **Rendering**: Use D3 data joins for efficient updates
- **Zoom/Pan**: Hardware-accelerated transforms
- **Large Sequences**: Virtual scrolling for token view
- **Transitions**: Smooth animations with requestAnimationFrame

### Scalability Limits
- **Max Sequence Length**: ~512 tokens (model limit)
- **Attention Matrix Size**: 512×512 = 262K cells
- **Graph Size**: ~50 nodes, ~100 edges (readable)
- **Response Time**: <5s for typical queries

## Security Considerations

- **Input Validation**: Sanitize user questions
- **Rate Limiting**: Prevent API abuse
- **CORS**: Restrict to known origins in production
- **Model Safety**: Use instruction-tuned model to reduce harmful outputs

## Deployment Architecture

### Development
```
localhost:8001 (Backend)
localhost:8000 (Frontend)
```

### Production (Potential)
```
Cloud VM (AWS/GCP/Azure)
  ├── Backend: Gunicorn + FastAPI
  ├── Frontend: Nginx static serving
  └── Model: Cached on disk
```

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | D3.js v7 | Visualization |
| Frontend | Vanilla JS | Application logic |
| Frontend | HTML5/CSS3 | UI structure/styling |
| Backend | FastAPI | REST API |
| Backend | PyTorch | Deep learning |
| Backend | Transformers | LLM inference |
| Backend | spaCy | NLP/NER |
| Backend | NetworkX | Graph algorithms |
| Model | SmolLM2-360M | Text generation |
| Data | Dolly-15K | Question dataset |

## Future Enhancements

1. **Multi-Model Support**: Compare attention across different LLMs
2. **Attention Flow Animation**: Visualize attention over generation steps
3. **Wikidata Integration**: Ground entities with external knowledge
4. **Export Functionality**: Save visualizations as images/PDFs
5. **Collaborative Features**: Share and annotate visualizations
6. **Performance Profiling**: Identify bottlenecks in reasoning
7. **Attention Patterns Library**: Catalog common attention patterns
8. **Interactive Tutorials**: Guide users through interpretation

## Conclusion

This architecture provides a robust, scalable foundation for exploring LLM reasoning through interactive visualization. The modular design enables easy extension and modification, while the synchronization mechanism creates a cohesive multi-view experience that reveals insights at multiple levels of abstraction.
