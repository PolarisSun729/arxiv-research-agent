// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Search from './pages/Search';
import Evaluation from './pages/Evaluation';

const App = () => {
  return (
    <Router>
      <div className="flex">
        <Sidebar />
        <main className="ml-64 flex-1 min-h-screen bg-gray-100">
          <Routes>
            <Route path="/search" element={<Search />} />
            <Route path="/" element={<Search />} />
            <Route path="/evaluation" element={<Evaluation />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
