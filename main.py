

from src.caasr import CAASRArgumentStructure
from transformers import GPT2Tokenizer,pipeline, AutoModelForSequenceClassification

from amf_fast_inference import model

from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import logging
import torch

logging.basicConfig(datefmt='%H:%M:%S', level=logging.DEBUG)

app = Flask(__name__)
metrics = PrometheusMetrics(app)
model_name = "debela-arg/dialogtp-am-medium"
loader = model.ModelLoader(model_name)
model = loader.load_model()   
#model =     AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


@metrics.summary('requests_by_status', 'Request latencies by status',
                 labels={'status': lambda r: r.status_code})
@metrics.histogram('requests_by_status_and_path', 'Request latencies by status and path',
                   labels={'status': lambda r: r.status_code, 'path': lambda: request.path})

@app.route('/caasra', methods = ['GET', 'POST'])
def caasra():
	if request.method == 'POST':
		file_obj = request.files['file']
		strcture_generator  = CAASRArgumentStructure(file_obj,pipe,tokenizer)
		structure = strcture_generator.get_argument_structure()

		return structure
	
	if request.method == 'GET':
		info = """The Inference Identifier is a component of AMF that detects argument relations between propositions. 
		This implementation utilises the Hugging Face implementation of BERT for textual entailment. 
		The model is fine-tuned to recognize inferences, conflicts, and non-relations. 
		It accepts xIAF as input and returns xIAF as output. 
		This component can be integrated into the argument mining pipeline alongside a segmenter."""
		return info	
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5015"), debug=False)	  

