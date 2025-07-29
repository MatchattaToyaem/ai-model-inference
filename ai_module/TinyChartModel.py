from ai_module import AiInterface
from utils import FileUtils
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds


class TinyChartModel(AiInterface):
    def __init__(self, ai_name = "mPLUG/TinyChart-3B-768"):
        self.tokenizer, self.model, self.processor, self.context_len = self.__load_ai_model(ai_name)


    def __load_ai_model(self, ai_name: str):
        """Load ai model"""

        tokenizer, model, processor, context_len = load_pretrained_model( 
            ai_name,  
            model_base=None, 
            model_name=get_model_name_from_path(ai_name), 
            device="cpu" # device="cpu" if running on cpu 
        )
        return tokenizer, model, processor, context_len

    def model_inference(self, datapath: str, chart_obj: dict, prompt: str = "Create a brief summarization or extract key insights based on the chart image."):
        """
        Interference ai model
        """
        error = None
        try:
            image_path = FileUtils.get_image_path(datapath, chart_obj['id'], chart_obj['type'])
            response = inference_model([image_path], prompt, self.model, self.tokenizer, self.processor, self.context_len, conv_mode="phi", max_new_tokens=1024)
        except Exception as e:
            error = {"id": chart_obj['id'], "error": e}
            raise e
        return response, error