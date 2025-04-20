const express = require('express');
const path = require('path');
const app = express();
const PORT = 5173;

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve the React app for any other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
