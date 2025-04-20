import React from 'react';
import { Routes, Route } from 'react-router-dom';
import './App.css';
import GenerateBanner from './components/banner.jsx';
import HomePage from './components/home.jsx';
import Links from './components/links.jsx';
import Thoughts from './components/thoughts.jsx';
import About from './components/about.jsx';
import Act from './components/act.jsx';

function App() {
  return (
    <div className="App">
      <GenerateBanner />
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/links" element={<Links />} />
          <Route path="/thoughts" element={<Thoughts />} />
          <Route path="/about" element={<About />} />
          <Route path="/act" element={<Act />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
