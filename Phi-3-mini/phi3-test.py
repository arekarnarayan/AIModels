import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import time
import gradio as gr

class Phi3MiniChat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = Path("models/phi-3-mini")
        self.setup_model()

    def setup_model(self):
        if not self.model_path.exists():
            raise Exception(f"Model directory not found at {self.model_path.absolute()}")

        try:
            import bitsandbytes as bnb
        except ImportError:
            import subprocess
            subprocess.check_call(["pip", "install", "bitsandbytes"])
            import bitsandbytes as bnb

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response(self, message, history):
        try:
            # Format with Phi-3's chat template
            formatted_prompt = f"<|user|>\n{message}\n<|assistant|>\n"
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract only the assistant's response
            assistant_response = full_response.split("<|assistant|>")[-1].strip()
            
            # Remove any remaining tokens
            for token in ["<|user|>", "<|system|>", "<|assistant|>"]:
                assistant_response = assistant_response.replace(token, "").strip()
            
            generation_time = time.time() - start_time
            return f"{assistant_response}\n\n[Generated in {generation_time:.2f} seconds]"

        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # Create chat interface
    chat_interface = Phi3MiniChat()
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        chat_interface.generate_response,
        title="Phi-3-Mini Chat Interface (Quantized Version)",
        description="Chat with the Phi-3-Mini model. This is running a 4-bit quantized version for efficient memory usage.",
        examples=[
            "What is the capital of France?",
            "Write a haiku about artificial intelligence.",
            "Explain how transformers work in machine learning."
        ],
        theme="soft"
    )
    
    # Launch the interface
    iface.launch(share=False)

if __name__ == "__main__":
    main()
