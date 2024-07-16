// Assuming you have the Socket.IO client library loaded
const socket = io();

// When a message is received from the server
socket.on('incoming', function(msg) {
    const messages = document.getElementById('chat-messages');
    const newMessage = document.createElement('li');
    newMessage.textContent = msg;
    newMessage.classList.add('incoming-msg');
    messages.appendChild(newMessage);
    messages.scrollTop = messages.scrollHeight; // Auto-scroll to the bottom
});

// When the form is submitted
document.getElementById('message-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const input = document.getElementById('input-message');
    const msg = input.value;
    socket.emit('outgoing', msg); // Send the message to the server (conversational interface)
    addOutgoingMessage(msg);
    input.value = ''; // Clear the input field
});

// Add outgoing message to the chatbox
function addOutgoingMessage(msg) {
    const messages = document.getElementById('chat-messages');
    const newMessage = document.createElement('li');
    newMessage.textContent = msg;
    newMessage.classList.add('outgoing-msg'); // Add outgoing message class
    messages.appendChild(newMessage);
    messages.scrollTop = messages.scrollHeight; // Auto-scroll to the bottom
}

// If you have a send button image instead of a submit button
document.querySelector('.send-button-container a').addEventListener('click', function(e) {
    e.preventDefault();
    const input = document.getElementById('input-message');
    const msg = input.value;
    socket.emit('message', msg); // Send the message to the server
    input.value = ''; // Clear the input field
});