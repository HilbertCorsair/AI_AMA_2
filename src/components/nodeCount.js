import React, { useState, useEffect } from 'react';

export default function NodeCount() {
  const [nodeCount, setNodeCount] = useState(0);
  
  useEffect(() => {
    // Simulate fetching node count from server
    const fetchNodeCount = async () => {
      try {
        // This is a placeholder - in a real app, you would fetch from your API
        // const response = await fetch('http://localhost:3001/node-count');
        // const data = await response.json();
        // setNodeCount(data.count);
        
        // For now, just set a random number
        setNodeCount(Math.floor(Math.random() * 1000) + 500);
      } catch (error) {
        console.error('Error fetching node count:', error);
      }
    };
    
    fetchNodeCount();
    
    // Refresh count every 30 seconds
    const interval = setInterval(fetchNodeCount, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div style={{ textAlign: 'center', margin: '1rem 0', color: '#10a37f', fontFamily: 'Arsenal, sans-serif' }}>
      <p>Knowledge Graph: {nodeCount.toLocaleString()} nodes</p>
    </div>
  );
}
