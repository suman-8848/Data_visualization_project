"""
Knowledge Extractor: Entity and relationship extraction using spaCy
Constructs semantic knowledge from LLM-generated text
"""

import spacy
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extracts entities and relationships from text"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy NLP pipeline
        
        Args:
            model_name: spaCy model to use
        """
        logger.info(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        logger.info("spaCy model loaded successfully")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with id, text, label, start, end
        """
        doc = self.nlp(text)
        
        entities = []
        for idx, ent in enumerate(doc.ents):
            entities.append({
                "id": f"entity_{idx}",
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "start_token": ent.start,
                "end_token": ent.end
            })
        
        # Also extract noun chunks as potential entities
        for idx, chunk in enumerate(doc.noun_chunks):
            # Avoid duplicates with named entities
            if not any(e["text"].lower() == chunk.text.lower() for e in entities):
                entities.append({
                    "id": f"chunk_{idx}",
                    "text": chunk.text,
                    "label": "CONCEPT",
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                    "start_token": chunk.start,
                    "end_token": chunk.end
                })
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using dependency parsing
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of relationship dictionaries with source, target, relation
        """
        doc = self.nlp(text)
        relationships = []
        
        # Create entity lookup by token position
        entity_by_token = {}
        for entity in entities:
            for token_idx in range(entity["start_token"], entity["end_token"]):
                entity_by_token[token_idx] = entity
        
        # Extract relationships using dependency parsing
        for token in doc:
            # Look for verb-based relationships
            if token.pos_ == "VERB":
                # Find subject and object
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                
                for subj in subjects:
                    for obj in objects:
                        # Check if subject and object are entities
                        subj_entity = self._find_entity_for_token(subj, entity_by_token)
                        obj_entity = self._find_entity_for_token(obj, entity_by_token)
                        
                        if subj_entity and obj_entity and subj_entity["id"] != obj_entity["id"]:
                            relationships.append({
                                "id": f"rel_{len(relationships)}",
                                "source": subj_entity["id"],
                                "target": obj_entity["id"],
                                "relation": token.lemma_,
                                "relation_text": token.text,
                                "confidence": 0.8
                            })
        
        # Add co-occurrence relationships for entities in same sentence
        for sent in doc.sents:
            sent_entities = [
                e for e in entities
                if e["start_token"] >= sent.start and e["end_token"] <= sent.end
            ]
            
            # Connect entities that appear together
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    # Avoid duplicate relationships
                    if not any(
                        (r["source"] == e1["id"] and r["target"] == e2["id"]) or
                        (r["source"] == e2["id"] and r["target"] == e1["id"])
                        for r in relationships
                    ):
                        relationships.append({
                            "id": f"rel_{len(relationships)}",
                            "source": e1["id"],
                            "target": e2["id"],
                            "relation": "co-occurs",
                            "relation_text": "appears with",
                            "confidence": 0.5
                        })
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def _find_entity_for_token(
        self,
        token,
        entity_by_token: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find entity that contains the given token
        
        Args:
            token: spaCy token
            entity_by_token: Mapping from token index to entity
            
        Returns:
            Entity dictionary or None
        """
        # Check if token itself is in an entity
        if token.i in entity_by_token:
            return entity_by_token[token.i]
        
        # Check if any child token is in an entity
        for child in token.children:
            if child.i in entity_by_token:
                return entity_by_token[child.i]
        
        return None
    
    def extract_with_context(
        self,
        question: str,
        answer: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and relationships with question-answer context
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Tuple of (entities, relationships)
        """
        # Process question and answer separately to maintain context
        q_doc = self.nlp(question)
        a_doc = self.nlp(answer)
        
        entities = []
        
        # Extract from question
        for idx, ent in enumerate(q_doc.ents):
            entities.append({
                "id": f"q_entity_{idx}",
                "text": ent.text,
                "label": ent.label_,
                "source": "question",
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Extract from answer
        for idx, ent in enumerate(a_doc.ents):
            entities.append({
                "id": f"a_entity_{idx}",
                "text": ent.text,
                "label": ent.label_,
                "source": "answer",
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Extract relationships from combined text
        combined_text = f"{question} {answer}"
        relationships = self.extract_relationships(combined_text, entities)
        
        return entities, relationships
