from ai_module import AiInterface
from utils import FileUtils
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration


class MatchaQa(AiInterface):
    def __init__(self, ai_name = "google/matcha-chart2text-pew"):
        self.model, self.processor = self.__load_ai_model(ai_name)


    def __load_ai_model(self, ai_name: str):
        """Load ai model"""
        processor = Pix2StructProcessor.from_pretrained(ai_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(ai_name)
        return model, processor

    def model_inference(self, datapath: str, chart_obj: dict, prompt: str = "Generate underlying summarization of the figure below:"):
        """
        Interference ai model
        """
        result = None
        error = None
        try:
            image = Image.open(FileUtils.get_image_path(datapath, chart_obj['id'], chart_obj['type'])).convert("RGB")
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            predictions = self.model.generate(**inputs, max_new_tokens=1024)
            result = self.processor.decode(predictions[0], skip_special_tokens=True)
        except Exception as e:
            error = {"id": chart_obj['id'], "error": e}
        return result, error