import React from 'react';
import { Link,  useMatch, useResolvedPath } from 'react-router-dom';
import "./banner.css"

export default function GenerateBanner() {
  return (
    <nav className="nav" id="banner">
    <img src="./assets/sd2.webp" alt="hilbertspace" id = 'h_space'/>
            
    <Link to="/" className="site-title">
    <h1>Dan's labyrinth</h1>
    </Link>
    <ul className = "banner-menu" id = "bm1" >
        <CustomLink to = '/'><li>Home</li> </CustomLink>
        <CustomLink to = '/links'><li>Links</li> </CustomLink>
        <CustomLink to = '/thoughts'><li>Thoughts</li> </CustomLink>
        <CustomLink to = '/about'><li>About</li> </CustomLink>
        <CustomLink to = '/act'><li>Act</li> </CustomLink>

    </ul>
    </nav>
  )
}

function CustomLink({ to, children, ...props }) {
  const resolvedPath = useResolvedPath(to)
  const isActive = useMatch({ path: resolvedPath.pathname, end: true })

  return (
    <li className={isActive ? "active" : ""}>
      <Link to={to} {...props}>
        {children}
      </Link>
    </li>
  )
}


