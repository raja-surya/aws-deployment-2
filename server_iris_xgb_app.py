# Fastapi server App for Iris flower prediction. 
# This receives an user input from a client, calls XGBoost model to predict,
#   and returns the prediction result back to the client.

from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse, Response
import xgboost as xgb
import joblib
import json
import numpy as np

app = FastAPI()

# This "/" route is just to test the fast api server is up and running.
@app.get("/")
def read_root():
#    return "Hello World"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI HTML Response</title>
    </head>
    <body>
        <h1>Hello !</h1>
        <p>This is a sample HTML response from Iris XGB FastAPI server.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Load the model
model_file_name = "DEMO-local-xgboost-model"
model = joblib.load(model_file_name)

# Classes of Iris flower to decode the prediction
classes = ["Setosa", "Versicolor", "Virginica"]


# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html
# To be compatible with SageMaker, your container must have the following characteristics:
# Your container must have a web server listing on port 8080.
# Your container must accept POST requests to the /invocations and /ping real-time endpoints. 

# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb
# Amazon SageMaker uses two URLs in the container:
# /ping will receive GET requests from the infrastructure. Your program returns 200 if the container is up and accepting requests.
# /invocations is the endpoint that receives client inference POST requests. The format of the request and the response is up to the
#    algorithm. If the client supplied ContentType and Accept headers, these will be passed in as well.

@app.get("/ping")
async def ping():
    return Response(content="OK", status_code=200)

# Main method to receive the post request from the client, classify the data received.
@app.post("/invocations")
async def read_predict(request: Request):
    data = await request.json()

    # The data received would be in the form {"data": [6, 3, 4.8, 1.8]}
    if 'data' in data:
        user_input = np.array(data['data'])

    # XGBoost prediction requires a 2-d numpy array like [[6, 3, 4.8, 1.8]]
    user_input = np.array([user_input])
    # Call the model for classification.
    result = model.predict(user_input)

    # We have to typecast numpy.int32 to int, as  fastapi cannot support numpy.int32
    # https://stackoverflow.com/questions/74005747/valueerror-typeerrornumpy-int32-object-is-not-iterable-typeerrorvars
    return classes[int(result[0])]
        
    
if __name__ == "__main__":
    uvicorn.run("server_iris_xgb_app:app", host = '0.0.0.0', port = 8080, \
                reload = True, reload_includes = ["server_iris_xgb_app.py"], reload_excludes = "*.py")



