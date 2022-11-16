import os
from flask import Flask, request, jsonify
from src.sampling_methods import sample_tokens
from src.sample_shake_speare_model import my_gpt, tokenizer
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>" 

@app.route('/check_health/', methods=['GET'])
def health():
    # Return the response in json format
    return jsonify({'message': 'App is running!'})

@app.route('/shakespeare/', methods=['GET'])
def shakespeare():
    text = request.args.get("text", None)

    # For debugging
    print(f"Received: {text}")

    response = {}

    # Check if the user sent a name at all
    if not text:
        response["ERROR"] = "No name found. Please send a name."
    else:
        response["MESSAGE"] = f"Recieved prompt '{text}'"

        text_output = sample_tokens(my_gpt,
                                    tokenizer,
                                    text,
                                    max_tokens_generated=100,
                                    temperature=1.0,
                                    top_p=0.3,
                                    freq_penalty=2)

        #print(text_output)

        response["Response"] = f"Recieved prompt '{text_output}'"

    # Return the response in json format
    return jsonify(response)

@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")

    response = {}

    # Check if the user sent a name at all
    if not name:
        response["ERROR"] = "No name found. Please send a name."
    # Check if the user entered a number
    elif str(name).isdigit():
        response["ERROR"] = "The name can't be numeric. Please send a string."
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome API!"

    # Return the response in json format
    return jsonify(response)


@app.route('/post/', methods=['POST'])
def post_something():
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": f"Welcome {name} to our awesome API!",
            # Add this option to distinct the POST request
            "METHOD": "POST"
        })
    else:
        return jsonify({
            "ERROR": "No name found. Please send a name."
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)