from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from . import AiInterface
from PIL import Image 
from utils import FileUtils

class UnichartModel(AiInterface):
    def __init__(self, ai_name = "ahmed-masry/unichart-chart2text-pew-960"):
        self.model, self.processor, self.device = self.__load_ai_model(ai_name)


    def __load_ai_model(self, ai_name: str):
        """Load ai model"""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = VisionEncoderDecoderModel.from_pretrained(ai_name)
        processor = DonutProcessor.from_pretrained(ai_name)
        model.to(device)
        return model, processor, device

    def model_inference(self, datapath: str, chart_obj: dict, prompt: str  = "<summarize_chart> Summarize the chart. <s_answer>"):
        """
        Interference ai model
        """
        error = None
        try:
            decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            image = Image.open(FileUtils.get_image_path(datapath, chart_obj['id'], chart_obj['type'])).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            sequence = sequence.split("<s_answer>")[1].strip()
        except Exception as e:
            print(e)
            error = {"id": chart_obj['id'], "error": e}
        return sequence, error 