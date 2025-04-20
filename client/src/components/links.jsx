import React, { useEffect, useState } from 'react';
import YouTubeEmbed from './youtubeConstent';
import './links.css'

const FedLinks = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('https://api.rss2json.com/v1/api.json?rss_url=https%3A%2F%2Fwww.federalreserve.gov%2Ffeeds%2Fpress_all.xml')
      .then(response => response.json())
      .then(data => setData(data))
      .catch(error => setError(error));
  }, []); 

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (!data) {
    return <div>Loading...</div>;
  }

  return (
    <div id='newsContainer'>
      <YouTubeEmbed></YouTubeEmbed>
      <h1>FED News</h1>
      <img id = "fed_image" src="/assets/fed.jpg" alt="FED press conference"></img>
      {data.items.map((item, index) => (
        <div key={index}>
          <a href={item.link}>{item.title}</a>
          <br />
        </div>
      ))}
    </div>
  );
};

export default FedLinks;
