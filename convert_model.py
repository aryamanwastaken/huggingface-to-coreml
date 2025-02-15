import torch
from transformers import MarianMTModel, MarianTokenizer
import coremltools as ct

def download_and_convert(model_name: str, sample_text: str = "Hello, how are you?"):
    """
    Downloads a translation model from Hugging Face, converts it to TorchScript,
    and then converts it to Core ML format.
    
    Parameters:
        model_name (str): The Hugging Face model identifier (e.g., "Helsinki-NLP/opus-mt-en-es").
        sample_text (str): A sample text for tracing the model.
    
    Returns:
        mlmodel (coremltools.models.MLModel): The converted Core ML model.
    """
    print(f"Downloading model '{model_name}'...")
    # Load the pre-trained MarianMT model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Tokenize a sample input to prepare for TorchScript conversion.
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("Converting model to TorchScript via tracing...")
    # Trace the model with sample inputs.
    # Note: MarianMTModel expects input_ids and attention_mask as inputs.
    # Adjust the tracing process if your model uses a different interface.
    try:
        scripted_model = torch.jit.trace(model, (input_ids, attention_mask))
    except Exception as e:
        print(f"Error during TorchScript conversion: {e}")
        raise
    
    print("Converting TorchScript model to Core ML format...")
    # Convert the TorchScript model to Core ML.
    # Specify the input tensor types. The shapes are taken from the sample input.
    try:
        mlmodel = ct.convert(
            scripted_model,
            inputs=[
                ct.TensorType(shape=input_ids.shape, name="input_ids"),
                ct.TensorType(shape=attention_mask.shape, name="attention_mask")
            ],
            source="torchscript"
        )
    except Exception as e:
        print(f"Error during Core ML conversion: {e}")
        raise
    
    print("Conversion successful!")
    return mlmodel

def save_coreml_model(mlmodel, output_path: str):
    """
    Saves the converted Core ML model to the specified output path.
    
    Parameters:
        mlmodel (coremltools.models.MLModel): The Core ML model to save.
        output_path (str): File path to save the .mlmodel file.
    """
    try:
        mlmodel.save(output_path)
        print(f"Core ML model saved to {output_path}")
    except Exception as e:
        print(f"Error saving Core ML model: {e}")
        raise

def main():
    # Define the Hugging Face model to convert
    model_name = "Helsinki-NLP/opus-mt-en-es"
    output_path = "MarianMT_en_es.mlmodel"
    
    # Download, convert, and save the model
    mlmodel = download_and_convert(model_name)
    save_coreml_model(mlmodel, output_path)

if __name__ == "__main__":
    main()
