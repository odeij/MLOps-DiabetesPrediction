import os
import json
import yaml
import joblib
import numpy as np


params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

# Make custom Exceptions

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception): #So that anyone sending a request, doesn't send invalid column names
    def __init__(self, message="Invalid Feature name(s)"):
        self.message = message
        super().__init__(self.message)

#Method to return prediction
def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    
    try:
        if 0 <= prediction <= 1:   #As the Target column can take values only in that range
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"

# Method to validate inputs
def validate_input(dict_request):

    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True

# Return prediction for form response
def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response

# Return prediction for api response
def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response": response}
            return response

    # except NotInCols as e:
    #     response = {"response": str(e), "Expected feature names": get_schema().keys()}
    #     return response

    except NotInRange as e:
        response = {"response": str(e), "Expected range": get_schema()}
        return response

    except Exception as e:
        response = {"response": str(e) }
        return response