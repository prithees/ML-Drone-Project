from flask import Flask, render_template, request, jsonify
import openai
import os

app = Flask(__name__)
openai.api_key = ""
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def uploading():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return "File uploaded successfully"

@app.route("/chat", methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=user_input,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    bot_response = response.choices[0].text.strip()
    return jsonify({"response": bot_response})

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")
    app.run(debug=True)
