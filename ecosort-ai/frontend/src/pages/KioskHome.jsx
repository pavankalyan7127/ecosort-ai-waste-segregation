import React, { useState } from 'react';
import CameraView from '../components/CameraView.jsx';
import DetectionCard from '../components/DetectionCard.jsx';
import AssistantChat from '../components/AssistantChat.jsx';
import CategoryLegend from '../components/CategoryLegend.jsx';

export default function KioskHome({ language }) {
  const [detections, setDetections] = useState([]);

  const handleDetectionsUpdated = (rawDetections) => {
    // Ensure detections is ALWAYS a flat array of detection items
    if (Array.isArray(rawDetections)) {
      setDetections(rawDetections);
    } else if (rawDetections && Array.isArray(rawDetections.detected_items)) {
      setDetections(rawDetections.detected_items);
    } else {
      setDetections([]);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Vision & Detections (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <CameraView
            language={language}
            onDetectionsUpdated={handleDetectionsUpdated}
          />
          <DetectionCard
            detections={detections}
            language={language}
          />
        </div>

        {/* Right Column: AI Assistant (5 cols) */}
        <div className="lg:col-span-5">
          <AssistantChat
            language={language}
            detectedItems={detections}
          />
        </div>
      </div>

      {/* Bottom Category Highlights */}
      <CategoryLegend language={language} />
    </div>
  );
}
