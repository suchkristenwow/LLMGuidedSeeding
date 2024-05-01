from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app)

messages = []

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@socketio.on('send_message')
def handle_send_message(data):
    message = data['message']
    print(f"Received message: {message}")
    if message.startswith("Robot:"):
        plain_message = message
        messages.append(('robot',"Robot: " +  plain_message))
        print(f"Broadcasting robot message: {plain_message}")
        emit('new_message', {'message': plain_message}, broadcast=True, include_self=False)

    else:
        plain_message = message
        messages.append(('user',"User: " + plain_message))
        print(f"Broadcasting user message: {plain_message}")
        emit('new_message', {'message': plain_message}, broadcast=True, include_self=False)


@socketio.on('get_messages')
def handle_get_messages():
    plain_messages = [(sender, message) if sender == 'user' else ('robot', "Robot: " + message) for sender, message in messages]
    print(f"Sending messages: {plain_messages}")
    emit('messages', {'messages': plain_messages})

@socketio.on('download_conversation')
def handle_download_conversation():
    conversation = '\n'.join([message for _, message in messages])
    print(f"Sending conversation data: {conversation}")
    emit('conversation_data', {'conversation': conversation})

if __name__ == '__main__':
    print("Starting Flask app")
    socketio.run(app, debug=True, port=7000)