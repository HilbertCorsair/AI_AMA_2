const express = require('express');
const path = require('path');
const app = express();
const PORT = 5173;

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Simple test route
app.get('/test', (req, res) => {
  res.send('Server is working!');
});

// Serve the home page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'home.html'));
});

// Add routes for other pages
app.get('/links', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Links - Dan's Labyrinth</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background-color: #343541;
          color: white;
          margin: 0;
          padding: 20px;
          text-align: center;
        }
        h1 {
          color: #10a37f;
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
        .banner {
          background-color: black;
          padding: 20px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .banner h1 {
          margin: 0;
          color: white;
        }
        .banner-links {
          display: flex;
          gap: 20px;
        }
        .banner-links a {
          color: white;
          text-decoration: none;
        }
        .banner-links a:hover {
          text-decoration: underline;
        }
        .content {
          margin: 40px 0;
        }
      </style>
    </head>
    <body>
      <div class="banner">
        <h1>Dan's Labyrinth</h1>
        <div class="banner-links">
          <a href="/">Home</a>
          <a href="/links">Links</a>
          <a href="/thoughts">Thoughts</a>
          <a href="/about">About</a>
          <a href="/act">Act</a>
        </div>
      </div>
      <div class="container">
        <div class="content">
          <h2>Links</h2>
          <p>This page will contain useful links and resources.</p>
          <p>Coming soon...</p>
        </div>
      </div>
    </body>
    </html>
  `);
});

app.get('/thoughts', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Thoughts - Dan's Labyrinth</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background-color: #343541;
          color: white;
          margin: 0;
          padding: 20px;
          text-align: center;
        }
        h1 {
          color: #10a37f;
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
        .banner {
          background-color: black;
          padding: 20px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .banner h1 {
          margin: 0;
          color: white;
        }
        .banner-links {
          display: flex;
          gap: 20px;
        }
        .banner-links a {
          color: white;
          text-decoration: none;
        }
        .banner-links a:hover {
          text-decoration: underline;
        }
        .content {
          margin: 40px 0;
        }
      </style>
    </head>
    <body>
      <div class="banner">
        <h1>Dan's Labyrinth</h1>
        <div class="banner-links">
          <a href="/">Home</a>
          <a href="/links">Links</a>
          <a href="/thoughts">Thoughts</a>
          <a href="/about">About</a>
          <a href="/act">Act</a>
        </div>
      </div>
      <div class="container">
        <div class="content">
          <h2>Thoughts</h2>
          <p>This page will contain thoughts and reflections.</p>
          <p>Coming soon...</p>
        </div>
      </div>
    </body>
    </html>
  `);
});

app.get('/about', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>About - Dan's Labyrinth</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background-color: #343541;
          color: white;
          margin: 0;
          padding: 20px;
          text-align: center;
        }
        h1 {
          color: #10a37f;
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
        .banner {
          background-color: black;
          padding: 20px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .banner h1 {
          margin: 0;
          color: white;
        }
        .banner-links {
          display: flex;
          gap: 20px;
        }
        .banner-links a {
          color: white;
          text-decoration: none;
        }
        .banner-links a:hover {
          text-decoration: underline;
        }
        .content {
          margin: 40px 0;
        }
      </style>
    </head>
    <body>
      <div class="banner">
        <h1>Dan's Labyrinth</h1>
        <div class="banner-links">
          <a href="/">Home</a>
          <a href="/links">Links</a>
          <a href="/thoughts">Thoughts</a>
          <a href="/about">About</a>
          <a href="/act">Act</a>
        </div>
      </div>
      <div class="container">
        <div class="content">
          <h2>About</h2>
          <p>This page will contain information about the project.</p>
          <p>Coming soon...</p>
        </div>
      </div>
    </body>
    </html>
  `);
});

app.get('/act', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Act - Dan's Labyrinth</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background-color: #343541;
          color: white;
          margin: 0;
          padding: 20px;
          text-align: center;
        }
        h1 {
          color: #10a37f;
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
        .banner {
          background-color: black;
          padding: 20px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .banner h1 {
          margin: 0;
          color: white;
        }
        .banner-links {
          display: flex;
          gap: 20px;
        }
        .banner-links a {
          color: white;
          text-decoration: none;
        }
        .banner-links a:hover {
          text-decoration: underline;
        }
        .content {
          margin: 40px 0;
        }
      </style>
    </head>
    <body>
      <div class="banner">
        <h1>Dan's Labyrinth</h1>
        <div class="banner-links">
          <a href="/">Home</a>
          <a href="/links">Links</a>
          <a href="/thoughts">Thoughts</a>
          <a href="/about">About</a>
          <a href="/act">Act</a>
        </div>
      </div>
      <div class="container">
        <div class="content">
          <h2>Act</h2>
          <p>This page will contain actions and activities.</p>
          <p>Coming soon...</p>
        </div>
      </div>
    </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
