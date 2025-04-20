import React, { useState, useEffect } from 'react';

const BotTyping = ({ message }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < message.length) {
      const timer = setTimeout(() => {
        setDisplayedText(prev => prev + message[currentIndex]);
        setCurrentIndex(prevIndex => prevIndex + 1);
      }, 20); // Adjust typing speed here
      
      return () => clearTimeout(timer);
    }
  }, [currentIndex, message]);

  useEffect(() => {
    // Reset when message changes
    setDisplayedText('');
    setCurrentIndex(0);
  }, [message]);

  return (
    <div className="message">
      {displayedText}
      {currentIndex < message.length && <span className="cursor">|</span>}
    </div>
  );
};

export default BotTyping;
