"""
TraceTranslator Model Loader
A clean interface for loading various pre-trained models for vision, language, and multimodal tasks.
"""

from typing import Optional, Dict, Any, Tuple
import torch
from pathlib import Path


class ModelLoader:
    """Central hub for loading and managing different types of pre-trained models."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the ModelLoader.
        
        Args:
            device: Device to load models on ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        if device is None:
            # Default to CPU for better compatibility
            # MPS has known issues with some operations (sparse tensors, certain matmul ops)
            self.device = "cpu"
        else:
            self.device = device
        print(f"ModelLoader initialized on device: {self.device}")
    
    @staticmethod
    def cleanup_model(model, processor=None):
        """
        Clean up model from memory.
        
        Args:
            model: Model to clean up
            processor: Optional processor to clean up
        """
        import gc
        del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def load_vision_model(self, model_name: str = "mobilenet_v2") -> Tuple[Any, Any]:
        """
        Load a vision model for image classification/feature extraction.
        
        Args:
            model_name: Name of the model to load
                Options: 'mobilenet_v2', 'resnet18', 'efficientnet_b0', 'vit_tiny'
        
        Returns:
            Tuple of (model, preprocessing_transform)
        """
        from torchvision import models, transforms
        import timm
        
        print(f"Loading vision model: {model_name}")
        
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            transform = models.MobileNet_V2_Weights.DEFAULT.transforms()
        
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            transform = models.ResNet18_Weights.DEFAULT.transforms()
        
        elif model_name == "efficientnet_b0":
            model = timm.create_model('efficientnet_b0', pretrained=True)
            transform = timm.data.create_transform(
                input_size=(3, 224, 224),
                is_training=False
            )
        
        elif model_name == "vit_tiny":
            model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
            transform = timm.data.create_transform(
                input_size=(3, 224, 224),
                is_training=False
            )
        
        else:
            raise ValueError(f"Unknown vision model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model, transform
    
    def load_language_model(self, model_name: str = "distilbert") -> Tuple[Any, Any]:
        """
        Load a language model for text processing.
        
        Args:
            model_name: Name of the model to load
                Options: 'distilbert', 'gpt2-small', 't5-small', 'minilm'
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading language model: {model_name}")
        
        model_map = {
            "distilbert": "distilbert-base-uncased",
            "gpt2-small": "gpt2",
            "t5-small": "t5-small",
            "minilm": "microsoft/MiniLM-L12-H384-uncased"
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown language model: {model_name}")
        
        hf_model_name = model_map[model_name]
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        
        # Use causal LM for GPT-2, prefer safetensors
        if model_name == "gpt2-small":
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, use_safetensors=True)
        else:
            model = AutoModel.from_pretrained(hf_model_name, use_safetensors=True)
        
        model = model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_multimodal_model(self, model_name: str = "clip") -> Tuple[Any, Any]:
        """
        Load a multimodal model for vision-language tasks.
        
        Args:
            model_name: Name of the model to load
                Options: 'clip', 'blip'
        
        Returns:
            Tuple of (model, processor)
        """
        from transformers import CLIPModel, CLIPProcessor, BlipForImageTextRetrieval, BlipProcessor
        
        print(f"Loading multimodal model: {model_name}")
        
        if model_name == "clip":
            # Use safetensors format to avoid torch.load security issues
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        elif model_name == "blip":
            # Use BlipForImageTextRetrieval instead of deprecated BlipModel
            model = BlipForImageTextRetrieval.from_pretrained(
                "Salesforce/blip-itm-base-coco", 
                use_safetensors=True
            )
            processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        
        else:
            raise ValueError(f"Unknown multimodal model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model, processor
    
    def load_audio_model(self, model_name: str = "whisper-tiny") -> Tuple[Any, Any]:
        """
        Load an audio processing model.
        
        Args:
            model_name: Name of the model to load
                Options: 'whisper-tiny', 'whisper-base', 'wav2vec2'
        
        Returns:
            Tuple of (model, processor)
        """
        print(f"Loading audio model: {model_name}")
        
        if model_name.startswith("whisper"):
            import whisper
            size = model_name.split("-")[1]  # tiny, base, small, etc.
            model = whisper.load_model(size, device=self.device)
            return model, None  # Whisper handles preprocessing internally
        
        elif model_name == "wav2vec2":
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            # Use Wav2Vec2ForCTC for speech recognition (includes CTC head)
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", use_safetensors=True)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = model.to(self.device)
            model.eval()
            return model, processor
        
        else:
            raise ValueError(f"Unknown audio model: {model_name}")
    
    def load_embedding_model(self, model_name: str = "all-MiniLM-L6-v2") -> Any:
        """
        Load a sentence embedding model.
        
        Args:
            model_name: Name of the model to load
                Options: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-MiniLM'
        
        Returns:
            SentenceTransformer model
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {model_name}")
        
        model_map = {
            "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "all-mpnet-base-v2",
            "paraphrase-MiniLM": "paraphrase-MiniLM-L6-v2"
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown embedding model: {model_name}")
        
        model = SentenceTransformer(model_map[model_name], device=self.device)
        return model
    
    def load_diffusion_model(self, model_name: str = "sd-turbo") -> Tuple[Any, Any]:
        """
        Load a diffusion model for image generation.
        
        Args:
            model_name: Name of the model to load
                Options: 'sd-turbo', 'lcm', 'sdxl-turbo'
        
        Returns:
            Tuple of (pipeline, None)
        """
        from diffusers import AutoPipelineForText2Image, LCMScheduler
        
        print(f"Loading diffusion model: {model_name}")
        
        if model_name == "sd-turbo":
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo",
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None
            )
        elif model_name == "sdxl-turbo":
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None
            )
        elif model_name == "lcm":
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        else:
            raise ValueError(f"Unknown diffusion model: {model_name}")
        
        pipeline = pipeline.to(self.device)
        return pipeline, None
    
    def load_reasoning_model(self, model_name: str = "phi-2") -> Tuple[Any, Any]:
        """
        Load a small reasoning/LLM model.
        
        Args:
            model_name: Name of the model to load
                Options: 'phi-2', 'tinyllama', 'phi-3-mini'
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading reasoning model: {model_name}")
        
        model_map = {
            "phi-2": "microsoft/phi-2",
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct"
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown reasoning model: {model_name}")
        
        hf_model_name = model_map[model_name]
        
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        model = model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_molecular_model(self, model_name: str = "chemberta") -> Tuple[Any, Any]:
        """
        Load a molecular/chemistry model.
        
        Args:
            model_name: Name of the model to load
                Options: 'chemberta', 'molformer'
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading molecular model: {model_name}")
        
        model_map = {
            "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
            "molformer": "ibm/MoLFormer-XL-both-10pct"
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown molecular model: {model_name}")
        
        hf_model_name = model_map[model_name]
        
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True, use_safetensors=True)
        
        model = model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_gnn_model(self, model_name: str = "gcn") -> Any:
        """
        Load a Graph Neural Network model architecture.
        
        Args:
            model_name: Name of the model to load
                Options: 'gcn', 'gat', 'graphsage'
        
        Returns:
            Model class (not instantiated - requires graph data)
        """
        from torch_geometric.nn import GCN, GAT, GraphSAGE
        
        print(f"Loading GNN architecture: {model_name}")
        
        model_map = {
            "gcn": GCN,
            "gat": GAT,
            "graphsage": GraphSAGE
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown GNN model: {model_name}")
        
        return model_map[model_name]
    
    def load_advanced_multimodal(self, model_name: str = "blip2") -> Tuple[Any, Any]:
        """
        Load advanced multimodal models.
        
        Args:
            model_name: Name of the model to load
                Options: 'blip2', 'llava-tiny'
        
        Returns:
            Tuple of (model, processor)
        """
        from transformers import Blip2ForConditionalGeneration, Blip2Processor, AutoProcessor, LlavaForConditionalGeneration
        
        print(f"Loading advanced multimodal model: {model_name}")
        
        if model_name == "blip2":
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                use_safetensors=True
            )
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        elif model_name == "llava-tiny":
            model = LlavaForConditionalGeneration.from_pretrained(
                "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained("tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown advanced multimodal model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model, processor
    
    def load_audio_generation_model(self, model_name: str = "musicgen") -> Tuple[Any, Any]:
        """
        Load audio generation models.
        
        Args:
            model_name: Name of the model to load
                Options: 'musicgen', 'audioldm'
        
        Returns:
            Tuple of (model, processor)
        """
        print(f"Loading audio generation model: {model_name}")
        
        if model_name == "musicgen":
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small",
                dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        elif model_name == "audioldm":
            from diffusers import AudioLDMPipeline
            model = AudioLDMPipeline.from_pretrained(
                "cvssp/audioldm-s-full-v2",
                dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            processor = None
        else:
            raise ValueError(f"Unknown audio generation model: {model_name}")
        
        model = model.to(self.device)
        return model, processor
    
    def load_code_model(self, model_name: str = "codebert") -> Tuple[Any, Any]:
        """
        Load code understanding models.
        
        Args:
            model_name: Name of the model to load
                Options: 'codebert', 'codet5-small'
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
        
        print(f"Loading code model: {model_name}")
        
        if model_name == "codebert":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base", use_safetensors=True)
        elif model_name == "codet5-small":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
            model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small", use_safetensors=True)
        else:
            raise ValueError(f"Unknown code model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_advanced_vision_model(self, model_name: str = "dinov2") -> Tuple[Any, Any]:
        """
        Load advanced vision models.
        
        Args:
            model_name: Name of the model to load
                Options: 'dinov2', 'sam', 'depth-anything'
        
        Returns:
            Tuple of (model, processor)
        """
        from transformers import AutoModel, AutoImageProcessor, SamModel, SamProcessor, AutoModelForDepthEstimation
        
        print(f"Loading advanced vision model: {model_name}")
        
        if model_name == "dinov2":
            model = AutoModel.from_pretrained("facebook/dinov2-small", use_safetensors=True)
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        elif model_name == "sam":
            model = SamModel.from_pretrained("facebook/sam-vit-base", use_safetensors=True)
            processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        elif model_name == "depth-anything":
            model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf", use_safetensors=True)
            processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        else:
            raise ValueError(f"Unknown advanced vision model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model, processor
    
    def load_object_detection_model(self, model_name: str = "yolo") -> Tuple[Any, None]:
        """
        Load object detection models.
        
        Args:
            model_name: Name of the model to load
                Options: 'yolo', 'detr'
        
        Returns:
            Tuple of (model, processor/None)
        """
        print(f"Loading object detection model: {model_name}")
        
        if model_name == "yolo":
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # nano version
            return model, None
        elif model_name == "detr":
            from transformers import DetrForObjectDetection, DetrImageProcessor
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", use_safetensors=True)
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = model.to(self.device)
            model.eval()
            return model, processor
        else:
            raise ValueError(f"Unknown object detection model: {model_name}")
    
    def get_available_models(self) -> Dict[str, list]:
        """
        Get a dictionary of all available model types and their options.
        
        Returns:
            Dictionary mapping model types to available model names
        """
        return {
            "vision": ["mobilenet_v2", "resnet18", "efficientnet_b0", "vit_tiny"],
            "language": ["distilbert", "gpt2-small", "t5-small", "minilm"],
            "multimodal": ["clip", "blip"],
            "audio": ["whisper-tiny", "whisper-base", "wav2vec2"],
            "embedding": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM"],
            "diffusion": ["sd-turbo", "sdxl-turbo", "lcm"],
            "reasoning": ["phi-2", "tinyllama", "phi-3-mini"],
            "molecular": ["chemberta", "molformer"],
            "gnn": ["gcn", "gat", "graphsage"],
            "advanced_multimodal": ["blip2", "llava-tiny"],
            "audio_generation": ["musicgen", "audioldm"],
            "code": ["codebert", "codet5-small"],
            "advanced_vision": ["dinov2", "sam", "depth-anything"],
            "object_detection": ["yolo", "detr"]
        }
    
    def print_available_models(self):
        """Print all available models in a formatted way."""
        models = self.get_available_models()
        print("\n" + "="*60)
        print("Available Models in TraceTranslator")
        print("="*60)
        for category, model_list in models.items():
            print(f"\n{category.upper()}:")
            for model in model_list:
                print(f"  - {model}")
        print("\n" + "="*60)


def main():
    """Example usage of the ModelLoader."""
    print("TraceTranslator Model Loader Demo\n")
    
    # Initialize loader
    loader = ModelLoader()
    
    # Show available models
    loader.print_available_models()
    
    print("\n" + "="*60)
    print("LOADING EXAMPLE MODELS FROM EACH CATEGORY")
    print("="*60)
    
    # Example: Load a vision model
    print("\n--- Loading Vision Model ---")
    vision_model, vision_transform = loader.load_vision_model("mobilenet_v2")
    print(f"✓ Vision model loaded: {type(vision_model).__name__}")
    ModelLoader.cleanup_model(vision_model, vision_transform)
    
    # Example: Load a language model
    print("\n--- Loading Language Model ---")
    lang_model, tokenizer = loader.load_language_model("distilbert")
    print(f"✓ Language model loaded: {type(lang_model).__name__}")
    ModelLoader.cleanup_model(lang_model, tokenizer)
    
    # Example: Load a multimodal model
    print("\n--- Loading Multimodal Model ---")
    mm_model, mm_processor = loader.load_multimodal_model("clip")
    print(f"✓ Multimodal model loaded: {type(mm_model).__name__}")
    ModelLoader.cleanup_model(mm_model, mm_processor)
    
    # Example: Load an embedding model
    print("\n--- Loading Embedding Model ---")
    embed_model = loader.load_embedding_model()
    print(f"✓ Embedding model loaded: {type(embed_model).__name__}")
    ModelLoader.cleanup_model(embed_model)
    
    # Example: Load a reasoning model
    print("\n--- Loading Reasoning Model (NEW) ---")
    try:
        reasoning_model, reasoning_tokenizer = loader.load_reasoning_model("phi-2")
        print(f"✓ Reasoning model loaded: {type(reasoning_model).__name__}")
        ModelLoader.cleanup_model(reasoning_model, reasoning_tokenizer)
    except Exception as e:
        print(f"⚠ Skipped (large download): {e.__class__.__name__}")
    
    # Example: Load a code model
    print("\n--- Loading Code Model (NEW) ---")
    try:
        code_model, code_tokenizer = loader.load_code_model("codebert")
        print(f"✓ Code model loaded: {type(code_model).__name__}")
        ModelLoader.cleanup_model(code_model, code_tokenizer)
    except Exception as e:
        print(f"⚠ Skipped: {e.__class__.__name__}")
    
    # Example: Load an advanced vision model
    print("\n--- Loading Advanced Vision Model (NEW) ---")
    try:
        adv_vision_model, adv_vision_processor = loader.load_advanced_vision_model("dinov2")
        print(f"✓ Advanced vision model loaded: {type(adv_vision_model).__name__}")
        ModelLoader.cleanup_model(adv_vision_model, adv_vision_processor)
    except Exception as e:
        print(f"⚠ Skipped: {e.__class__.__name__}")
    
    # Example: Load an object detection model
    print("\n--- Loading Object Detection Model (NEW) ---")
    try:
        detection_model, detection_processor = loader.load_object_detection_model("yolo")
        print(f"✓ Object detection model loaded: {type(detection_model).__name__}")
        ModelLoader.cleanup_model(detection_model, detection_processor)
    except Exception as e:
        print(f"⚠ Skipped: {e.__class__.__name__}")
    
    # Example: Load a GNN architecture
    print("\n--- Loading GNN Architecture (NEW) ---")
    try:
        gnn_class = loader.load_gnn_model("gcn")
        print(f"✓ GNN architecture loaded: {gnn_class.__name__}")
    except Exception as e:
        print(f"⚠ Skipped: {e.__class__.__name__}")
    
    print("\n" + "="*60)
    print("✓ Demo completed! All 14 model categories available.")
    print("  Use loader.load_<category>_model() to load any model.")
    print("="*60)


if __name__ == "__main__":
    main()

