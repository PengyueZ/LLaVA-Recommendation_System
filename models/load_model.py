from llava.model.builder import load_pretrained_model

model_path = "llava-hf/llava-1.5-7b-hf"
model, vis_processor, tokenizer = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava"
)