// Chat.js
import React, { useState } from 'react';
import './home.css';
import BotTyping from './botTypeing';

// Import icons
import bot from '../assets/bot.svg';
import user from '../assets/user.svg';
import send from '../assets/send.svg';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const genUId = () => {
    const timestamp = Date.now();
    const randomNumber = Math.random();
    const hexadecimalString = randomNumber.toString(16);
    return `id-${timestamp}-${hexadecimalString}`;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!userInput.trim()) return;

    const newMessage = {
      AI: false,
      text: userInput,
      uid: genUId(),
    };

    setMessages([...messages, newMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:3001/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: userInput }),
      });

      if (response.ok) {
        const data = await response.json();
        // Check the structure of the response
        const parsedData = data.response ? data.response.trim() : 'No response from server';

        setMessages((prevMessages) => [
          ...prevMessages,
          { AI: true, text: parsedData, uid: genUId() },
        ]);
      } else {
        const err = await response.text();
        console.error('Error:', err);
        setMessages((prevMessages) => [
          ...prevMessages,
          { AI: true, text: "Sorry, I couldn't process your request. Please try again.", uid: genUId() },
        ]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { AI: true, text: "Sorry, there was an error connecting to the server. Please try again later.", uid: genUId() },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="chat_container">
      {messages.map((message) => (
        <div key={message.uid} id={message.uid} className={`wrapper ${message.AI ? 'ai' : ''}`}>
          <div className="chat">
            <div className="profile">
              <img 
                src={message.AI ? bot : user}
                alt={message.AI ? 'bot' : 'user'} />
            </div>
            {message.AI ? <BotTyping message={message.text}/> : <div className="message">{message.text}</div>}
          </div>
        </div>
      ))}
      
      {isLoading && (
        <div className="wrapper ai">
          <div className="chat">
            <div className="profile">
              <img src={bot} alt="bot" />
            </div>
            <div className="message">Thinking...</div>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <textarea 
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSubmit(e)}
          name="prompt" 
          rows="1" 
          cols="1" 
          placeholder="Ask me anything!"
        ></textarea>
        <button type="submit"><img src={send} alt="send"/></button>
      </form>
    </div>
  );
};

export default Chat;
