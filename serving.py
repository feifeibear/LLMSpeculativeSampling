from flask import Flask, request, jsonify
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2

app = Flask(__name__)
pipeline = None

GLOBAL_SERVER = None

class Server:
    def __init__(self, approx_model_name, target_model_name) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logging.info("begin load models")
        self._small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, trust_remote_code=True).to(self._device)
        self._large_model = AutoModelForCausalLM.from_pretrained(target_model_name, trust_remote_code=True).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
        logging.info("fininsh load models")
          
        self.num_tokens = 40
        self.top_k = 10
        self.top_p = 0.9
        
    def process_request(self, request : str) -> torch.Tensor:
        input_str = request['prompt']
        logging.info(f"recieve request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        output = speculative_sampling(input_ids, 
                                      self._small_model, 
                                      self._large_model, self.num_tokens, 
                                      top_k = self.top_k, 
                                      top_p = self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Set up a route to listen for inference requests
@app.route('/predict', methods=['POST'])
def predict():
    # Check the content type of the request
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Invalid content type'})

    # Get the request data
    request_data = request.json

    # Perform inference
    result = GLOBAL_SERVER.process_request(request_data)

    # Return the inference results
    return jsonify(result)

if __name__ == '__main__':
    GLOBAL_SERVER = Server(approx_model_name="/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
           target_model_name="/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1")
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000)
