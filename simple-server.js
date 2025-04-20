const express = require('express');
const path = require('path');
const app = express();
const PORT = 5173;

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve the direct.html file for the root path
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'direct.html'));
});

// Serve the direct.html file for any other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'direct.html'));
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
