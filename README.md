# infer_huggingface_semantic_segmentation

This plugin proposes inference for semantic segmentation using transformer models from Hugging Face. 

Models can be loaded from:

- Your fine-tuned model (local path folder containing model weights and config file)

- The [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=image-segmentation) using the model name, e.g.:  
    - [nvidia/segformer-b5-finetuned-ade-640-640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) 
    - [microsoft/beit-base-finetuned-ade-640-640](https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640)




The following models can be use with this plugin: 

-	BEiT 

-	Data2VecVision 

-	DPT 

-	MobileNetV2 

-	MobileViT 

-	SegFormer 
