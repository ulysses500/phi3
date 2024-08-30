# Inference du model phi3
This page will summarize the problems we encountered during the inference of the model phi3 and solutions we found.

First it should be noted that we have used different medium to serve the model in a server. We have used the following medium:
- Ollama : https://ollama.com/   https://github.com/ollama/ollama
- DeepInfra : https://deepinfra.com/microsoft/Phi-3-medium-4k-instruct/api

## Options Explanation

`/set parameter seed <int>`
**Random number seed**: This parameter sets a specific seed value for the random number generator. By setting a seed value, the sequence of random numbers generated will be deterministic. This ensures that the output will be the same each time the model is run with the same seed, providing reproducibility.

`/set parameter num_predict <int>`
**Max number of tokens to predict**: This parameter defines the maximum number of tokens that the model will generate. When generating text, this parameter limits the length of the output. For example, if you set `num_predict` to 50, the model will generate up to 50 tokens in response to the input.

`/set parameter top_k <int>`
**Pick from top k number of tokens**: This parameter restricts the model to selecting the next token from the top `k` most probable tokens. When the model predicts the next token, it considers a range of possibilities. The `top_k` parameter limits these possibilities to the `k` highest probability tokens, which can help to produce more coherent and relevant output.

`/set parameter top_p <float>`
**Pick token based on sum of probabilities**: This parameter implements a method known as nucleus sampling or top-p sampling. Instead of picking from a fixed number of top tokens (as in `top_k`), `top_p` considers the smallest number of tokens whose cumulative probability adds up to `p`. This allows for more dynamic token selection based on the actual distribution of probabilities.

`/set parameter min_p <float>`
**Pick token based on top token probability * min_p**: This parameter sets a minimum threshold for token selection based on probability. When generating a token, the model will only consider tokens whose probability is at least `min_p` times the probability of the most likely token. This prevents the selection of very unlikely tokens, which can reduce nonsensical or irrelevant outputs.

`/set parameter num_ctx <int>`
**Set the context size**: This parameter defines the context window size or the number of tokens the model considers when generating the next token. The larger the context, the more history the model takes into account, which can improve the coherence of the generated text.

`/set parameter temperature <float>`
**Set creativity level**: This parameter controls the randomness of the model’s predictions. A higher temperature value (e.g., 1.5) results in more random, creative, and diverse outputs, while a lower temperature value (e.g., 0.7) produces more focused and deterministic text. A temperature of 1.0 typically represents the default randomness.

`/set parameter repeat_penalty <float>`
**How strongly to penalize repetitions**: This parameter penalizes the model for generating repeated tokens. When generating text, the model may sometimes fall into loops, repeating the same words or phrases. The `repeat_penalty` parameter discourages such behavior by reducing the probability of selecting tokens that have already been used.

`/set parameter repeat_last_n <int>`
**Set how far back to look for repetitions**: This parameter determines how many tokens back the model should look to identify repetitions. If you set this to a high value, the model will be more sensitive to repeated patterns over a longer sequence. Lower values make the model focus on more recent repetitions.

`/set parameter num_gpu <int>`
**The number of layers to send to the GPU**: This parameter determines how much of the model is offloaded to the GPU. By offloading more layers of the model to the GPU, you can speed up computation, but it requires more GPU memory. Setting this parameter appropriately can optimize performance based on your hardware capabilities.

`/set parameter stop <string> <string> ...`
**Set the stop parameters**: This parameter specifies the conditions under which the model should stop generating text. The model will stop generating further tokens if it encounters any of the specified stop sequences. For example, if you set `/set parameter stop "."`, the model would stop after generating a period, assuming the end of a sentence. Multiple stop sequences can be provided.

## Problem 1: Streaming
Ollama is an interface that will serve the different models locally.
Inference using this medium is done by sending a request to the server.
Here is the documentation for the API of Ollama: https://github.com/ollama/ollama/blob/main/docs/api.md

To serve response in streaming, the request should contain the parameters "stream" set to true.
One token doesnt correspond to a word.
```py
response = ollama.chat(model=default_model, stream=default_stream_option, messages=messages, options=default_options)
buffer = ""
for partial_resp in response:
    token = partial_resp["message"]["content"]
    buffer += token
    if buffer.endswith((" ", ".", ",", "!", "?", ";", ":", "\n")):
        words = buffer.split()
        yield "\n".join(words) + "\n"
        buffer = ""
```
This way each token sent is a word.

For the second medium, lf the lib_llm_infra.py file
You can create an API around this code.
To permit deployed app to have streaming, change nginx configuration and add in location:
location /{}: proxy_buffering off;

We have then focused on the inference of the model phi3 using Ollama because of limitations in conducting chat with DeepInfra.


## Problem 2 : Hallucinations

By using ollama we have noticed that the model phi3 generates hallucinations.
The model generates text that is not coherent with the prompt. To alleviate this problem we reset the ollama server after each request.
The problem with this approach is that it disables possibility of parallel request handling.

We have then implemented a solution where the reset of the server is done when at the start of a new line we have certain caracteristics.
In the different hallucinations we can see that the model is generating a conversation with itself.
We search for words such as : "system", "user", "assistant", "bot", "Query :", "Response :", "Prompt :", "Answer :", "Question :", "Chat :", "Conversation :"


## Problem 3 : 