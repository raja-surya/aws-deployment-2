FROM python:3.8.19

# Setting this path was required for error-free AWS execution.
ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program

COPY ./requirements.txt /opt/program/requirements.txt
COPY ./server_iris_xgb_app.py /opt/program/server_iris_xgb_app.py
COPY ./DEMO-local-xgboost-model /opt/program/DEMO-local-xgboost-model

RUN pip install -r /opt/program/requirements.txt 
    
EXPOSE 8080

#CMD python server_iris_xgb_app.py
ENTRYPOINT ["python", "/opt/program/server_iris_xgb_app.py"]
