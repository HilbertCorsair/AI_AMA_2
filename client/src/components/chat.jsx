// Chat.js
import React, { useState } from 'react';
import bot from './../icons/bot.svg';
import user from './../icons/user.svg';
import send from './../icons/icons8-send-button-50.png'
import './home.css'
import BotTyping from './botTypeing';




const Chat = () => {

  /*

  let loadInterval

  function loader(element) {
      element.textContent = ''

      loadInterval = setInterval(() => {
          // Update the text content of the loading indicator
          element.textContent += '.';

          // If the loading indicator has reached three dots, reset it
          if (element.textContent === '....') {
              element.textContent = '';
          }
      }, 300);
  }

  function typeText(element, text) {
      let index = 0

      let interval = setInterval(() => {
          if (index < text.length) {
              element.innerHTML += text.charAt(index)
              index++
          } else {
              clearInterval(interval)
          }
      }, 20)
  }
  */
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');

  const genUId = () => {
    const timestamp = Date.now();
    const randomNumber = Math.random();
    const hexadecimalString = randomNumber.toString(16);
    return `id-${timestamp}-${hexadecimalString}`;
  };
  

  const handleSubmit = async (e) => {
    e.preventDefault();

    const newMessage = {
      AI: false,
      text: userInput,
      uid: genUId(),
    };

    setMessages([...messages, newMessage]);
    setUserInput('');

    const response = await fetch('http://localhost:3001/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt: userInput }),
    });

    if (response.ok) {
      const data = await response.json();
      const parsedData = data.bot.content.trim();

      setMessages([
        ...messages,
        newMessage,
        { AI: true, text: parsedData, uid: genUId() },
      ]);
        
    } else {
      const err = await response.text();
      alert(err);
    }

   

    // specific message div 
    //const messageDiv = document.getElementById(uid)

    // messageDiv.innerHTML = "..."
    //loader(messageDiv)
  };

  return (
    <div id="chat_container">
      
      {messages.map((message) => (
        <div id={message.uid} className={`wrapper ${message.AI ? 'ai' : ''}`}>
          <div className="chat">
            <div className="profile">
              <img 
                src={message.AI ? bot : user}
                alt={message.AI ? 'bot' : 'user'} />
            </div>
            {message.AI ? <BotTyping message = {message.text}/> : <div className="message">{message.text}</div>}
          </div>
        </div>
      ))}
      <form onSubmit={handleSubmit} onKeyUp={(e)=>{e.keyCode === 13 && handleSubmit(e)}}>
      <textarea type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)} 
          name = "prompt" rows="1" cols="1" placeholder="Ask me anything!">
      </textarea>
        <button type="submit"><img src={send} alt="send"/></button>
      </form>
    </div>
  );
};

export default Chat;