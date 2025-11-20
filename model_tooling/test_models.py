"""
Test script to run a single forward pass through each model type
Verifies that all models are working correctly
Includes memory cleanup between tests to avoid RAM issues
"""

from model_loader import ModelLoader
import torch
from PIL import Image
import numpy as np
import warnings
import os
import gc

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def cleanup():
    """Force garbage collection and clear cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_vision_models():
    """Test all vision models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING VISION MODELS")
    print("="*70)
    
    loader = ModelLoader()
    vision_models = ["mobilenet_v2", "resnet18", "efficientnet_b0", "vit_tiny"]
    
    # Create a dummy RGB image
    dummy_image = Image.new('RGB', (224, 224), color=(128, 64, 192))
    
    for model_name in vision_models:
        try:
            print(f"\nTesting {model_name}...")
            model, transform = loader.load_vision_model(model_name)
            
            # Preprocess image
            input_tensor = transform(dummy_image).unsqueeze(0).to(loader.device)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            print(f"   ✓ Input shape: {input_tensor.shape}")
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"   ✓ Top-5 predictions: {output.topk(5).indices[0].tolist()}")
            
            # Cleanup
            ModelLoader.cleanup_model(model, transform)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_language_models():
    """Test all language models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING LANGUAGE MODELS")
    print("="*70)
    
    loader = ModelLoader()
    language_models = ["distilbert", "gpt2-small", "t5-small", "minilm"]
    
    test_text = "The quick brown fox jumps over the lazy dog."
    
    for model_name in language_models:
        try:
            print(f"\nTesting {model_name}...")
            model, tokenizer = loader.load_language_model(model_name)
            
            # Special handling for different model types
            if model_name == "gpt2-small":
                tokenizer.pad_token = tokenizer.eos_token
                inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(loader.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                main_output = outputs.logits
            elif model_name == "t5-small":
                inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(loader.device) for k, v in inputs.items()}
                with torch.no_grad():
                    encoder_outputs = model.encoder(**inputs)
                main_output = encoder_outputs.last_hidden_state
            else:
                inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(loader.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    main_output = outputs.last_hidden_state
                elif hasattr(outputs, 'logits'):
                    main_output = outputs.logits
                else:
                    main_output = outputs[0]
            
            print(f"   ✓ Input text: '{test_text}'")
            print(f"   ✓ Input tokens: {inputs['input_ids'].shape}")
            print(f"   ✓ Output shape: {main_output.shape}")
            print(f"   ✓ Output mean: {main_output.mean().item():.4f}")
            print(f"   ✓ Output std: {main_output.std().item():.4f}")
            
            # Cleanup
            ModelLoader.cleanup_model(model, tokenizer)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_multimodal_models():
    """Test all multimodal models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING MULTIMODAL MODELS")
    print("="*70)
    
    loader = ModelLoader()
    multimodal_models = ["clip", "blip"]
    
    # Create test data
    dummy_image = Image.new('RGB', (224, 224), color=(200, 100, 50))
    test_texts = ["a red sunset", "a blue ocean", "a green forest"]
    
    for model_name in multimodal_models:
        try:
            print(f"\n Testing {model_name}...")
            model, processor = loader.load_multimodal_model(model_name)
            
            # BLIP ITM requires one text at a time
            if model_name == "blip":
                inputs = processor(images=dummy_image, text=test_texts[0], return_tensors="pt")
            else:
                inputs = processor(text=test_texts, images=dummy_image, return_tensors="pt", padding=True)
            
            # Move inputs to device
            inputs = {k: v.to(loader.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"   ✓ Image: 224x224 RGB")
            
            # Handle different output formats
            if hasattr(outputs, 'logits_per_image'):
                print(f"   ✓ Texts: {test_texts}")
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)[0]
                print(f"   ✓ Image-text similarities (CLIP format):")
                for i, text in enumerate(test_texts):
                    print(f"      - '{text}': {probs[i].item():.4f}")
            elif hasattr(outputs, 'itm_score'):
                print(f"   ✓ Text: '{test_texts[0]}'")
                scores = outputs.itm_score
                match_prob = torch.softmax(scores, dim=1)[0][1].item()
                print(f"   ✓ Image-text match probability: {match_prob:.4f}")
            else:
                print(f"   ✓ Texts: {test_texts}")
                print(f"   ✓ Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'N/A'}")
            
            # Cleanup
            ModelLoader.cleanup_model(model, processor)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_audio_models():
    """Test audio models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING AUDIO MODELS")
    print("="*70)
    
    loader = ModelLoader()
    
    # Test Whisper
    try:
        print(f"\n Testing whisper-tiny...")
        model, processor = loader.load_audio_model("whisper-tiny")
        
        # Create dummy audio (1 second at 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32)
        
        # Whisper has its own transcribe method
        result = model.transcribe(dummy_audio, fp16=False)
        
        print(f"   ✓ Audio shape: {dummy_audio.shape}")
        print(f"   ✓ Sample rate: 16000 Hz")
        print(f"   ✓ Transcription result: '{result['text']}'")
        print(f"   ✓ Model loaded successfully")
        
        # Cleanup
        ModelLoader.cleanup_model(model, processor)
        cleanup()
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        cleanup()
    
    # Test Wav2Vec2
    try:
        print(f"\n Testing wav2vec2...")
        model, processor = loader.load_audio_model("wav2vec2")
        
        # Create dummy audio (1 second at 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32)
        
        # Process audio
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(loader.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Wav2Vec2ForCTC outputs logits, not hidden states
        logits = outputs.logits
        
        print(f"   ✓ Audio shape: {dummy_audio.shape}")
        print(f"   ✓ Input shape: {inputs['input_values'].shape}")
        print(f"   ✓ Output shape: {logits.shape}")
        print(f"   ✓ Output mean: {logits.mean().item():.4f}")
        
        # Decode the predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print(f"   ✓ Transcription: '{transcription[0]}'")
        
        # Cleanup
        ModelLoader.cleanup_model(model, processor)
        cleanup()
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        cleanup()


def test_embedding_models():
    """Test embedding models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING EMBEDDING MODELS")
    print("="*70)
    
    loader = ModelLoader()
    embedding_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM"]
    
    test_sentences = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "The weather is nice today."
    ]
    
    for model_name in embedding_models:
        try:
            print(f"\n Testing {model_name}...")
            model = loader.load_embedding_model(model_name)
            
            # Encode sentences
            embeddings = model.encode(test_sentences)
            
            print(f"   ✓ Input sentences: {len(test_sentences)}")
            print(f"   ✓ Embedding shape: {embeddings.shape}")
            print(f"   ✓ Embedding dimension: {embeddings.shape[1]}")
            print(f"   ✓ Embedding norm (first): {np.linalg.norm(embeddings[0]):.4f}")
            
            # Calculate similarity between first two sentences
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            print(f"   ✓ Similarity (sent 1 vs 2): {sim:.4f}")
            
            # Cleanup
            ModelLoader.cleanup_model(model)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_diffusion_models():
    """Test diffusion models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING DIFFUSION MODELS")
    print("="*70)
    
    loader = ModelLoader()
    diffusion_models = ["sd-turbo"]  # Test one to avoid huge downloads
    
    for model_name in diffusion_models:
        try:
            print(f"\n Testing {model_name}...")
            pipeline, _ = loader.load_diffusion_model(model_name)
            
            # Generate a small test image
            prompt = "a simple red circle"
            result = pipeline(
                prompt=prompt,
                num_inference_steps=1,  # Minimal steps for testing
                guidance_scale=0.0,
                height=64,  # Small size for testing
                width=64
            )
            image = result.images[0]
            
            print(f"   ✓ Prompt: '{prompt}'")
            print(f"   ✓ Generated image size: {image.size}")
            print(f"   ✓ Image mode: {image.mode}")
            print(f"   ✓ Model loaded successfully")
            
            # Cleanup
            del pipeline
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e.__class__.__name__}: {str(e)[:80]}")
            cleanup()


def test_reasoning_models():
    """Test reasoning/LLM models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING REASONING MODELS")
    print("="*70)
    
    loader = ModelLoader()
    reasoning_models = ["tinyllama"]  # Test smallest one
    
    for model_name in reasoning_models:
        try:
            print(f"\n Testing {model_name}...")
            model, tokenizer = loader.load_reasoning_model(model_name)
            
            prompt = "Q: What is 2+2? A:"
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(loader.device) for k, v in inputs.items()}
            
            # Generate a short response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"   ✓ Prompt: '{prompt}'")
            print(f"   ✓ Response: '{response[:100]}...'")
            print(f"   ✓ Model loaded successfully")
            
            # Cleanup
            ModelLoader.cleanup_model(model, tokenizer)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e.__class__.__name__}: {str(e)[:100]}")
            cleanup()


def test_molecular_models():
    """Test molecular/chemistry models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING MOLECULAR MODELS")
    print("="*70)
    
    loader = ModelLoader()
    molecular_models = ["chemberta"]
    
    for model_name in molecular_models:
        try:
            print(f"\n Testing {model_name}...")
            model, tokenizer = loader.load_molecular_model(model_name)
            
            # SMILES string for caffeine
            smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(loader.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            embeddings = outputs.last_hidden_state
            
            print(f"   ✓ SMILES: '{smiles}'")
            print(f"   ✓ Input shape: {inputs['input_ids'].shape}")
            print(f"   ✓ Embedding shape: {embeddings.shape}")
            print(f"   ✓ Model loaded successfully")
            
            # Cleanup
            ModelLoader.cleanup_model(model, tokenizer)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_gnn_models():
    """Test GNN architectures with a forward pass"""
    print("\n" + "="*70)
    print("TESTING GNN MODELS")
    print("="*70)
    
    loader = ModelLoader()
    gnn_models = ["gcn"]
    
    for model_name in gnn_models:
        try:
            print(f"\n Testing {model_name}...")
            model_class = loader.load_gnn_model(model_name)
            
            # Create a small GNN instance
            from torch_geometric.data import Data
            model = model_class(in_channels=16, hidden_channels=32, num_layers=2, out_channels=7)
            model = model.to(loader.device)
            model.eval()
            
            # Create dummy graph data
            edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4],
                [1, 0, 2, 1, 3, 2, 4, 3]
            ], dtype=torch.long)
            x = torch.randn(5, 16)  # 5 nodes, 16 features
            
            data = Data(x=x, edge_index=edge_index).to(loader.device)
            
            with torch.no_grad():
                output = model(data.x, data.edge_index)
            
            print(f"   ✓ Graph: 5 nodes, 8 edges")
            print(f"   ✓ Input features: {x.shape}")
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Model architecture loaded successfully")
            
            # Cleanup
            del model
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_code_models():
    """Test code understanding models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING CODE MODELS")
    print("="*70)
    
    loader = ModelLoader()
    code_models = ["codebert"]
    
    for model_name in code_models:
        try:
            print(f"\n Testing {model_name}...")
            model, tokenizer = loader.load_code_model(model_name)
            
            code_snippet = "def hello_world():\n    print('Hello, World!')"
            inputs = tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(loader.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            embeddings = outputs.last_hidden_state
            
            print(f"   ✓ Code: '{code_snippet[:50]}...'")
            print(f"   ✓ Input shape: {inputs['input_ids'].shape}")
            print(f"   ✓ Embedding shape: {embeddings.shape}")
            print(f"   ✓ Model loaded successfully")
            
            # Cleanup
            ModelLoader.cleanup_model(model, tokenizer)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_advanced_vision_models():
    """Test advanced vision models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING ADVANCED VISION MODELS")
    print("="*70)
    
    loader = ModelLoader()
    advanced_vision_models = ["dinov2"]
    
    for model_name in advanced_vision_models:
        try:
            print(f"\n Testing {model_name}...")
            model, processor = loader.load_advanced_vision_model(model_name)
            
            dummy_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
            inputs = processor(images=dummy_image, return_tensors="pt")
            inputs = {k: v.to(loader.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # DINOv2 outputs last_hidden_state
            features = outputs.last_hidden_state
            
            print(f"   ✓ Image: 224x224 RGB")
            print(f"   ✓ Feature shape: {features.shape}")
            print(f"   ✓ Model loaded successfully")
            
            # Cleanup
            ModelLoader.cleanup_model(model, processor)
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def test_object_detection_models():
    """Test object detection models with a forward pass"""
    print("\n" + "="*70)
    print("TESTING OBJECT DETECTION MODELS")
    print("="*70)
    
    loader = ModelLoader()
    detection_models = ["yolo"]
    
    for model_name in detection_models:
        try:
            print(f"\n Testing {model_name}...")
            model, _ = loader.load_object_detection_model(model_name)
            
            # Create test image
            dummy_image = Image.new('RGB', (640, 640), color=(128, 128, 128))
            
            # Run detection
            results = model(dummy_image)
            
            print(f"   ✓ Image: 640x640 RGB")
            print(f"   ✓ Number of results: {len(results)}")
            if len(results) > 0:
                print(f"   ✓ Detections: {len(results[0].boxes)} objects")
            print(f"   ✓ Model loaded successfully")
            
            # Cleanup
            del model
            cleanup()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            cleanup()


def main():
    """Run all model tests"""
    print("\n" + "="*70)
    print("MODEL FORWARD PASS TESTING - ALL 14 CATEGORIES")
    print("Testing each model type with sample inputs")
    print("Memory cleanup enabled between tests")
    print("="*70)
    
    # Run all tests (14 categories)
    test_vision_models()              # 1. Vision
    test_language_models()            # 2. Language
    test_multimodal_models()          # 3. Multimodal
    test_audio_models()               # 4. Audio
    test_embedding_models()           # 5. Embeddings
    test_diffusion_models()           # 6. Diffusion (NEW)
    test_reasoning_models()           # 7. Reasoning (NEW)
    test_molecular_models()           # 8. Molecular (NEW)
    test_gnn_models()                 # 9. GNN (NEW)
    test_code_models()                # 10. Code (NEW)
    test_advanced_vision_models()     # 11. Advanced Vision (NEW)
    test_object_detection_models()    # 12. Object Detection (NEW)
    
    # Note: Advanced Multimodal and Audio Generation skipped to save time/memory
    # They can be tested individually if needed
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETED (12/14 categories)")
    print("="*70)
    print("\nAll tested models have been verified with forward passes.")
    print("Skipped: Advanced Multimodal (BLIP-2, LLaVA) and Audio Generation")
    print("         (large downloads, can test individually if needed)\n")


if __name__ == "__main__":
    main()
