import React, {useState} from 'react';
import Greeting from './placeholder.jsx';
import Chat from './chat.jsx';
import {SignupForm, AuthenticationButtons, LoginForm} from './signin.jsx';
import NodeCount from './nodeCount.jsx';


export default function HomePage(){
  const [view, setView] = useState('none'); // none, signup, or login
  const handleSignupClick = () => setView(view === 'none'? 'signup' : 'none');
  const handleLoginClick = () => setView(view === 'none'? 'login' : 'none');

  return(
    <>
      <AuthenticationButtons onSignupClick={handleSignupClick} onLoginClick={handleLoginClick} />
      {view === 'signup' && <SignupForm />}
      {view === 'login' && <LoginForm />}
      <NodeCount></NodeCount>
      <Greeting></Greeting>
      <Chat></Chat>
    </>
  )
}
