// script.js
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function sendMessage() {
    const message = userInput.value.trim();
    if (message === "") return;

    // Display user message
    const userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = message;
    chatBox.appendChild(userMessage);

    // Send the message to the server
    fetch("/get_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: message }),
    })
        .then((response) => response.json())
        .then((data) => {
            // Display bot response
            const botMessage = document.createElement("div");
            botMessage.className = "bot-message";
            botMessage.textContent = data.response;
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
        });

    userInput.value = ""; // Clear the input field
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);

// Listen for Enter key press
userInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent default form submission
        sendMessage();
    }
});