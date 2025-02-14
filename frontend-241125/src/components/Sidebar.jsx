import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
  const location = useLocation();
  const links = [
    { to: "/search", text: "Similarity Search" },
    { to: "/evaluation", text: "Search Evaluation" }
  ];

  return (
    <div className="w-64 bg-gray-800 h-screen fixed left-0 top-0">
      <div className="p-4">
        <img 
          src="https://www.a-star.edu.sg/images/librariesprovider15/default-album/18.-institute-of-high-performance-computing-(ihpc).png?sfvrsn=ff199933_14" 
          alt="IHPC Logo" 
          className="w-full h-auto mb-6"
        />
      </div>

      <nav className="p-4">
        {links.map(link => (
          <Link
            key={link.to}
            to={link.to}
            className={`block px-4 py-3 text-gray-300 hover:bg-gray-700 ${
              location.pathname === link.to ? 'bg-gray-700' : ''
            }`}
          >
            {link.text}
          </Link>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;