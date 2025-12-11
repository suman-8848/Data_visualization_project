"""
FastAPI Backend for LLM Reasoning Visualization
Handles inference, attention extraction, and knowledge graph construction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from model_handler import ModelHandler
from knowledge_extractor import KnowledgeExtractor
from graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Reasoning Visualization API")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_handler = None
knowledge_extractor = None
graph_builder = None


class QueryRequest(BaseModel):
    """Request model for Q&A queries"""
    question: str
    context: Optional[str] = None
    max_length: int = 150
    temperature: float = 0.8
    top_p: float = 0.95
    layer_aggregation: str = "mean"  # mean, max, or specific layer index


class QueryResponse(BaseModel):
    """Response model containing all visualization data"""
    question: str
    answer: str
    tokens: List[str]
    question_tokens: List[str]  # Clean question tokens for visualization
    answer_tokens: List[str]  # Clean answer tokens for visualization
    answer_indices: Optional[List[int]] = []  # Indices of clean answer tokens in the full token list
    attention: Dict[str, Any]  # Attention matrices and metadata
    knowledge_graph: Dict[str, Any]  # Nodes and edges
    metadata: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize models and extractors on startup"""
    global knowledge_extractor, graph_builder
    
    knowledge_extractor = KnowledgeExtractor()
    graph_builder = GraphBuilder()
    
    # Load model in background thread to avoid blocking startup
    import threading
    def load_model_background():
        global model_handler
        logger.info("Initializing models in background...")
        try:
            model_handler = ModelHandler(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    threading.Thread(target=load_model_background, daemon=True).start()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "SmolLM2-360M-Instruct",
        "endpoints": ["/query", "/health", "/sample-questions"]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "model_loaded": model_handler is not None,
        "extractor_loaded": knowledge_extractor is not None,
        "graph_builder_loaded": graph_builder is not None
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint: Process question and return visualization data
    
    Pipeline:
    1. Generate answer with attention extraction
    2. Extract entities and relationships
    3. Build knowledge graph
    4. Return synchronized data
    """
    try:
        logger.info(f"Processing query: {request.question[:50]}...")
        
        # Step 1: Generate answer with attention weights
        result = model_handler.generate_answer(
            question=request.question,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        answer = result["answer"]
        tokens = result["tokens"]
        attention_data = result["attention"]
        
        logger.info(f"Generated answer: {answer[:50]}...")
        
        # Step 2: Extract entities and relationships
        combined_text = f"{request.question} {answer}"
        entities = knowledge_extractor.extract_entities(combined_text)
        relationships = knowledge_extractor.extract_relationships(combined_text, entities)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        
        # Step 3: Build knowledge graph
        graph_data = graph_builder.build_graph(
            entities=entities,
            relationships=relationships,
            tokens=tokens,
            attention=attention_data
        )
        
        # Step 4: Prepare response
        response = QueryResponse(
            question=request.question,
            answer=answer,
            tokens=tokens,
            question_tokens=result.get("question_tokens", []),
            answer_tokens=result.get("answer_tokens", []),
            answer_indices=result.get("answer_indices", []),
            attention=attention_data,
            knowledge_graph=graph_data,
            metadata={
                "num_tokens": len(tokens),
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "num_layers": attention_data["num_layers"],
                "num_heads": attention_data["num_heads"]
            }
        )
        
        logger.info("Query processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample-questions")
async def get_sample_questions():
    """Return sample questions from Dolly-15K dataset"""
    samples = [
        {
            "question": "What is the capital of France?",
            "category": "closed_qa"
        },
        {
            "question": "Explain how photosynthesis works in plants.",
            "category": "open_qa"
        },
        {
            "question": "Who wrote the novel '1984'?",
            "category": "closed_qa"
        },
        {
            "question": "What are the main causes of climate change?",
            "category": "open_qa"
        },
        {
            "question": "Describe the process of DNA replication.",
            "category": "summarization"
        }
    ]
    return {"samples": samples}


@app.get("/attention-patterns")
async def get_attention_patterns():
    """Return descriptions of common attention patterns"""
    return {
        "patterns": [
            {
                "name": "Self-Attention",
                "description": "Token attends primarily to itself",
                "interpretation": "Often seen in content words that carry independent meaning"
            },
            {
                "name": "Previous Token",
                "description": "Strong attention to immediately preceding token",
                "interpretation": "Common in sequential/grammatical relationships"
            },
            {
                "name": "Local Context",
                "description": "Attention spread across nearby tokens (±3 positions)",
                "interpretation": "Captures local syntactic and semantic context"
            },
            {
                "name": "Global Spread",
                "description": "Attention distributed broadly across all tokens",
                "interpretation": "Aggregating information from entire context"
            },
            {
                "name": "Sparse Focus",
                "description": "Concentrated attention on few key tokens",
                "interpretation": "Identifying most relevant information for current prediction"
            },
            {
                "name": "Positional Bias",
                "description": "Strong attention to beginning of sequence",
                "interpretation": "Often captures task instructions or key context"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
