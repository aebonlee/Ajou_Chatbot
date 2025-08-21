document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Function to add a new message to the chatbox
    function addMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add(sender + '-message');
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
    }

    // Function to handle sending messages
    function sendMessage() {
        const message = userInput.value.trim();
        if (message !== '') {
            addMessage(message, 'user');
            userInput.value = ''; // Clear the input field

            // Simulate a bot response after a short delay
            setTimeout(() => {
                const botMessage = "안녕하세요! 무엇을 도와드릴까요?"; // Simple example bot response
                addMessage(botMessage, 'bot');
            }, 500);
        }
    }

    // Event listeners
    sendBtn.addEventListener('click', sendMessage);

    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial bot message
    addMessage("안녕하세요! 챗봇에 오신 것을 환영합니다.", 'bot');
});