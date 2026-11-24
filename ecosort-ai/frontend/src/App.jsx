import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header.jsx';
import KioskHome from './pages/KioskHome.jsx';
import WasteGuide from './pages/WasteGuide.jsx';

export default function App() {
  const [language, setLanguage] = useState('en');

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans w-full selection:bg-emerald-500 selection:text-slate-950">
        <Header language={language} setLanguage={setLanguage} />

        <main className="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6">
          <Routes>
            <Route path="/" element={<KioskHome language={language} />} />
            <Route path="/guide" element={<WasteGuide language={language} />} />
          </Routes>
        </main>

        <footer className="border-t border-slate-900 bg-slate-950/80 py-4 px-6 text-center text-xs text-slate-500">
          EcoSort AI &bull; Problem Statement 2: Smart Recycling and Waste Segregation &bull; Bharat AI-SoC Challenge 2026–27
        </footer>
      </div>
    </BrowserRouter>
  );
}
