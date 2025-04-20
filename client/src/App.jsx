import React from 'react';
import { Routes, Route } from 'react-router-dom';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI AMA App</h1>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<div>Home Page</div>} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
