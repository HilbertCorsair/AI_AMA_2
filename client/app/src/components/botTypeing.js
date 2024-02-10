import React, { useEffect, useRef } from 'react';

function useTypeText(text, delay = 20) {
  const ref = useRef(null);

  useEffect(() => {
    const element = ref.current;
    let index = 0;

    if (element) {
      const interval = setInterval(() => {
        if (index < text.length) {
          element.innerHTML += text.charAt(index);
          index++;
        } else {
          clearInterval(interval);
        }
      }, delay);
      return () => clearInterval(interval); // Cleanup function to clear the interval if the component unmounts
    }
  }, [text, delay]); // Dependency array. The effect will only run on the first render since text and delay are not supposed to change.

  return ref;
}


function BotTyping(props) {
  const txt = props.message;
  const ref = useTypeText(txt);

  return <div className="message" ref={ref}></div>;
}

export default BotTyping;
