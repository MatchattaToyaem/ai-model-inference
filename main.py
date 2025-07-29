from ai_module import UnichartModel, TinyChartModel, MatchaQa
import argparse
from utils import FileUtils


def main():
    parser = argparse.ArgumentParser(description="Process user input.")
    parser.add_argument("--model_name", help="Name of the model")
    parser.add_argument("--datapath", help="Datapath of the testing data")
    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.datapath
    if "tiny" in model_name:
        ai_agent = TinyChartModel(model_name)
    elif "matcha" in model_name:
        ai_agent = MatchaQa(model_name)
    elif "uni" in model_name:
        ai_agent = UnichartModel(model_name)
    else:
        print(f"This model name is not support \'{model_name}\'")
        return -1
    
    key_file_name = f"{data_path}/sorted_charts_output_pew_test.txt"
    chart_objs = FileUtils.get_chart_objs(key_file_name)
    # chart_obj = chart_ob
    result_objs = []
    execution_errors = []
    print("------------start------------")
    for chart_obj in chart_objs:
        print("====================Process "+chart_obj["id"]+" ===========================")
        response, error = ai_agent.model_inference(data_path, chart_obj)
        result_objs.append({"id": chart_obj['id'],"type": chart_obj["type"], "actual": response, "expected": chart_obj['caption']})
        if error:
            execution_errors.append(error)
            print(error)
        print("====================Done "+chart_obj["id"]+" ===========================")
    FileUtils.create_output_file("output_unichart.json", result_objs)
    FileUtils.create_output_file("error_unichart.txt", execution_errors)
    print("====================Completed===========================")

if __name__ == "__main__":
    main()