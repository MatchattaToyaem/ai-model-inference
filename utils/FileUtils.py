import json
import os


class FileUtils:
    @staticmethod
    def get_image_path(data_path: str, chart_id: str, chart_type: str):
        type, image_id = chart_id.split("-")
        if type == "two_col":
            image_data = "t_"
        elif type == "multi_col":
            image_data = "m_"
        else:
            image_data = ""
        image_data += image_id
        return f"{data_path}/{chart_type}/{image_data}.png"
    
    @staticmethod
    def get_chart_objs(key_file_name: str):
        chart_objs = []
        with open(key_file_name, "r") as file:
            
            for line in  file.readlines():
                chart_obj = json.loads(line)
                chart_objs.append(chart_obj)
        return chart_objs
    
    @staticmethod
    def create_output_file(file_name: str, results: list):
        result_directory = os.path.join(os.getcwd(), 'inference_result')
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        with open(f"{result_directory}/{file_name}", "w+") as file:
            file.write(json.dumps(results))