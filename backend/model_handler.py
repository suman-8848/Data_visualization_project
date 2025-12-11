import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-360M-Instruct"):
        """Initialize the model handler with SmolLM2-360M-Instruct"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            output_attentions=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate_answer(self, question: str, max_new_tokens: int = 50, temperature: float = 0.8, top_p: float = 0.95):
        """Generate answer and extract attention patterns"""
        try:
            # Tokenize just the question to get clean question tokens
            question_only_inputs = self.tokenizer(question, return_tensors="pt")
            question_token_ids = question_only_inputs["input_ids"][0]
            question_raw_tokens = self.tokenizer.convert_ids_to_tokens(question_token_ids)
            # Clean question tokens for display
            clean_question_tokens = [t.replace('Ġ', ' ').replace('▁', ' ').strip() or t for t in question_raw_tokens]
            
            # Format as chat message for instruction model
            messages = [{"role": "user", "content": question}]
            
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            # Tokenize full prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            full_prompt_length = input_ids.shape[1]
            
            logger.info(f"Question: {question}")
            logger.info(f"Clean question tokens: {clean_question_tokens}")
            
            # Generate answer with user-controlled parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=10,  # Force at least 10 new tokens
                    do_sample=True,
                    temperature=temperature,  # User-controlled
                    top_p=top_p,  # User-controlled
                    top_k=50,
                    repetition_penalty=1.1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the full output
            generated_ids = outputs.sequences[0]
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Decode only the new tokens (answer part)
            prompt_length = input_ids.shape[1]
            answer_ids = generated_ids[prompt_length:]
            answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            
            # If answer is empty, use a fallback
            if not answer:
                answer = full_text.strip()
            
            logger.info(f"Generated {len(answer_ids)} new tokens")
            logger.info(f"Full text: {full_text}")
            logger.info(f"Answer: {answer}")
            
            # Get attention from a forward pass for visualization
            with torch.no_grad():
                forward_outputs = self.model(
                    generated_ids.unsqueeze(0),
                    output_attentions=True
                )
            
            # Extract attention weights
            attentions = forward_outputs.attentions
            raw_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
            
            # Clean up tokens - remove the Ġ prefix (space marker) for display
            # If a token is just 'Ġ', replacement makes it ' ', strip makes it '', so we use '_' to show it's a special space
            tokens = []
            for t in raw_tokens:
                clean = t.replace('Ġ', ' ').replace('▁', ' ').strip()
                if not clean:
                     tokens.append('_' if 'Ġ' in t or '▁' in t else t)
                else:
                    tokens.append(clean)
            
            # Process attention for visualization
            attention_data = self._process_attention(attentions, tokens, prompt_length)
            
            # Find where question tokens appear in the full token sequence
            # Search for the question tokens in the prompt portion
            question_start_in_prompt = self._find_question_position(tokens[:prompt_length], clean_question_tokens)
            
            # Add question-specific info for cleaner visualization
            attention_data["question_token_count"] = len(clean_question_tokens)
            attention_data["question_start_idx"] = question_start_in_prompt
            
            logger.info(f"Question starts at position {question_start_in_prompt} in prompt of length {prompt_length}")
            
            # Create clean token lists for visualization
            # Use the pre-computed clean question tokens
            answer_token_list = tokens[prompt_length:]
            
            # Clean answer tokens (remove special tokens but keep content)
            clean_answer_tokens = []
            answer_indices = []
            for i, t in enumerate(answer_token_list):
                if t and not t.startswith('<|') and not t.endswith('|>') and t not in ['<s>', '</s>']:
                    clean_answer_tokens.append(t)
                    answer_indices.append(prompt_length + i)
            
            # If all tokens were filtered, use original
            if not clean_answer_tokens:
                clean_answer_tokens = [t for t in answer_token_list if t]
                answer_indices = [prompt_length + i for i, t in enumerate(answer_token_list) if t]
            
            logger.info(f"Question tokens: {clean_question_tokens}")
            logger.info(f"Answer tokens (first 10): {clean_answer_tokens[:10]}")
            
            return {
                "answer": answer,
                "tokens": tokens,
                "question_tokens": clean_question_tokens,  # Use clean question tokens
                "answer_tokens": clean_answer_tokens,
                "answer_indices": answer_indices,
                "attention": attention_data,
                "full_text": full_text,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def _process_attention(self, attentions, tokens, prompt_length):
        """Process attention weights for visualization"""
        # Use the last layer's attention
        last_layer_attention = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
        
        # Average across attention heads
        avg_attention = last_layer_attention.mean(dim=0).cpu().numpy()  # Shape: [seq_len, seq_len]
        
        return {
            "attention_mean": avg_attention.tolist(),
            "matrix": avg_attention.tolist(),
            "num_heads": int(last_layer_attention.shape[0]),
            "num_layers": len(attentions),
            "seq_length": len(tokens),
            "prompt_length": prompt_length  # Actual prompt length
        }
    
    def _find_question_position(self, prompt_tokens, question_tokens):
        """Find where the question tokens start in the full prompt"""
        if not question_tokens or not prompt_tokens:
            return 0
        
        # Clean tokens for comparison
        prompt_clean = [t.lower().strip() for t in prompt_tokens]
        question_clean = [t.lower().strip() for t in question_tokens]
        
        # Search for the first question token
        first_q_token = question_clean[0]
        
        for i in range(len(prompt_clean) - len(question_clean) + 1):
            # Check if this position matches the start of question
            if first_q_token in prompt_clean[i] or prompt_clean[i] in first_q_token:
                # Verify by checking a few more tokens
                match_count = 0
                for j in range(min(5, len(question_clean))):
                    if i + j < len(prompt_clean):
                        if (question_clean[j] in prompt_clean[i + j] or 
                            prompt_clean[i + j] in question_clean[j] or
                            question_clean[j] == prompt_clean[i + j]):
                            match_count += 1
                
                if match_count >= min(3, len(question_clean)):
                    return i
        
        # Fallback: assume question is after initial template tokens
        # Look for common patterns like "user" or the actual question start
        for i, token in enumerate(prompt_clean):
            if 'user' in token or '\\n' in token:
                # Question likely starts after this
                if i + 1 < len(prompt_clean):
                    return i + 1
        
        return max(0, len(prompt_tokens) - len(question_tokens) - 5)
