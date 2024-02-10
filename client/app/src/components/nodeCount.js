import React, { useState, useEffect } from 'react';
import axios from 'axios';  // If using axios

function NodeCount() {
  const [nodeCount, setNodeCount] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Using axios
    axios.get('http://localhost:3001/node-count')
      .then(response => {
        setNodeCount(response.data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });

    // If using fetch
    /*
    fetch('http://localhost:3000/node-count')
      .then(response => response.text())
      .then(data => {
        setNodeCount(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
    */

  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      Total number of nodes: {nodeCount}
    </div>
  );
}

export default NodeCount;
