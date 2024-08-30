import requests
import json

URL = 'http://url/v1/completions'
HEADERS = {'Content-Type': 'application/json'}


def generate_response_chat(input_text):
    if isinstance(input_text, str):
        messages = [{"role": "user", "content": input_text}]
    else:
        messages = [{"role": item["role"], "content": item["content"]} for item in input_text]

    prompt = ""
    user_role = "user"
    for message in messages:
        if message["role"] == "system":
            prompt += f"System: {message['content']}\n"
        elif message["role"] == "user":
            prompt += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            prompt += f"Assistant: {message['content']}\n"
            user_role = "assistant"

    params = {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "prompt": prompt,
        "user": user_role,  # dynamically set the user role
        "max_tokens":1000,
        "frequency_penalty": 0.5,  # Adjust this value as needed
        "presence_penalty": 0.5,   # Adjust this value as needed
        "top_p": 0.9,              # Use nucleus sampling
        "temperature": 1,
        "stream": True
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=params, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        if response.status_code == 200:
            role_counters = {"Assistant:": 0, "System:": 0, "User:": 0}
            stop_signal_received = False
            buffer = ""

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        decoded_line = line.strip()
                        if decoded_line.startswith("data: "):
                            decoded_line = decoded_line[len("data: "):]  # Remove the prefix
                        if decoded_line:  # Ensure the line is not empty after stripping the prefix
                            if "DONE" in decoded_line:
                                # Handle the "DONE" line here
                                print("Processing completed.")
                                break
                            else:
                                data = json.loads(decoded_line)
                                text_content = data['choices'][0]['text']
                                buffer += text_content

                                for role in role_counters.keys():
                                    role_counters[role] += buffer.count(role)
                                    buffer = buffer.replace(role, "", buffer.count(role))

                                    if role_counters[role] >= 2:
                                        stop_signal_received = True
                                        break

                                yield f"{text_content}\n"

                                if stop_signal_received:
                                    print("Stop condition met.")
                                    break

                    except json.JSONDecodeError as e:
                        yield f"Failed to decode JSON: {e}\n"

    except requests.exceptions.RequestException as e:
        yield f"Error: {str(e)}\n"
