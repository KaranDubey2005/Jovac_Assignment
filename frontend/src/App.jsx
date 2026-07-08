import React, { useState, useEffect } from 'react';
import './App.css';

const MACRO_MAPPING = {
  "T-shirt/top": "Tops & Outerwear",
  "Pullover": "Tops & Outerwear",
  "Dress": "Tops & Outerwear",
  "Coat": "Tops & Outerwear",
  "Shirt": "Tops & Outerwear",
  "Trouser": "Trousers & Bottoms",
  "Sandal": "Footwear",
  "Sneaker": "Footwear",
  "Ankle boot": "Footwear",
  "Bag": "Accessories & Bags"
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [activeSampleId, setActiveSampleId] = useState(null);
  const [showFineGrained, setShowFineGrained] = useState(false);

  const API_BASE = window.location.origin === 'http://localhost:5173' 
    ? 'http://localhost:8000' 
    : window.location.origin;

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (!res.ok) throw new Error("Failed to load sample images");
      const data = await res.json();
      setSamples(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchSamples();
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleImageSelection(file);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      handleImageSelection(file);
    }
  };

  const handleImageSelection = (file) => {
    if (!file.type.startsWith('image/')) {
      setError("Please select a valid image file.");
      return;
    }
    setError(null);
    setSelectedFile(file);
    setActiveSampleId(null);
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
    
    triggerPrediction(file);
  };

  const handleSelectSample = async (sample) => {
    setError(null);
    setActiveSampleId(sample.id);
    setImagePreview(sample.image);
    setSelectedFile(null);
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(sample.image);
      const blob = await response.blob();
      const file = new File([blob], `sample_${sample.id}.png`, { type: 'image/png' });
      
      triggerPrediction(file);
    } catch (err) {
      setError("Failed to process sample image.");
      setLoading(false);
    }
  };

  const triggerPrediction = async (fileObj) => {
    setLoading(true);
    setResult(null);
    setError(null);
    setShowFineGrained(false);

    const formData = new FormData();
    formData.append("file", fileObj);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Server error during prediction");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to contact prediction server.");
    } finally {
      setLoading(false);
    }
  };

  const activeSample = samples.find(s => s.id === activeSampleId);

  return (
    <div className="container">
      <header className="app-header">
        <h1 className="title">E-Commerce Product Classifier</h1>
        <p className="subtitle">Instant product image categorizer powered by Transfer Learning (MobileNetV3-Large)</p>
      </header>

      <main className="app-main">
        <section className="panel upload-panel">
          <div className="panel-header">
            <h2>Select Product Image</h2>
          </div>
          
          <div 
            className="drop-zone"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {imagePreview ? (
              <div className="preview-container">
                <img src={imagePreview} className="image-preview animate-fade" alt="Preview" />
                <button className="btn-clear" onClick={() => {
                  setImagePreview(null);
                  setSelectedFile(null);
                  setResult(null);
                  setActiveSampleId(null);
                }}>✕ Clear</button>
              </div>
            ) : (
              <div className="drop-zone-prompt">
                <div className="upload-icon">✦</div>
                <p>Drag and drop a product image here, or</p>
                <label className="btn-upload">
                  Browse File
                  <input type="file" onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
                </label>
              </div>
            )}
          </div>

          <div className="samples-section">
            <div className="samples-header">
              <h3>Or use a Test Sample from Fashion-MNIST:</h3>
              <button className="btn-refresh" onClick={fetchSamples} title="Load different samples">
                ↻ Refresh
              </button>
            </div>
            <div className="samples-grid">
              {samples.map((sample) => (
                <button
                  key={sample.id}
                  className={`sample-card ${activeSampleId === sample.id ? 'active' : ''}`}
                  onClick={() => handleSelectSample(sample)}
                >
                  <img src={sample.image} alt={sample.true_label} />
                  <span className="sample-label">{sample.true_label}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="panel results-panel">
          <div className="panel-header">
            <h2>Classification Results</h2>
          </div>

          <div className="results-content">
            {loading && (
              <div className="status-container loading-state">
                <div className="spinner"></div>
                <p>Running neural network inference...</p>
              </div>
            )}

            {error && (
              <div className="status-container error-state animate-fade">
                <div className="error-icon">⚠</div>
                <p>{error}</p>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="status-container empty-state">
                <p>Select or upload an image to view predictions and probability scores.</p>
              </div>
            )}

            {result && !loading && (
              <div className="prediction-report animate-fade">
                <div className="best-match-card">
                  <span className="badge-match">Best Prediction</span>
                  <h3 className="predicted-label">{result.predicted_macro}</h3>
                  <div className="confidence-meter-container">
                    <div className="confidence-meter-fill" style={{ width: `${result.macro_confidence * 100}%` }}></div>
                  </div>
                  <p className="confidence-value">
                    Confidence: <strong>{(result.macro_confidence * 100).toFixed(2)}%</strong>
                  </p>
                  
                  <div className="sub-prediction-details">
                    <p>
                      Estimated Silhouette: <strong>{result.predicted_fine}</strong> (Confidence: {(result.fine_confidence * 100).toFixed(1)}%)
                    </p>
                  </div>
                  
                  {activeSample && (
                    <div className={`verification-badge ${MACRO_MAPPING[activeSample.true_label] === result.predicted_macro ? 'correct' : 'incorrect'}`}>
                      {MACRO_MAPPING[activeSample.true_label] === result.predicted_macro 
                        ? `✓ Matches Dataset Category! (${result.predicted_macro})` 
                        : `✗ Ground Truth is: ${activeSample.true_label} (${MACRO_MAPPING[activeSample.true_label]})`}
                    </div>
                  )}
                </div>

                <div className="probabilities-chart">
                  <h4>Category Probability Distribution</h4>
                  <div className="probabilities-list">
                    {Object.entries(result.macro_probabilities).map(([className, score]) => (
                      <div key={className} className="probability-row">
                        <div className="prob-label-container">
                          <span className="prob-name">{className}</span>
                          <span className="prob-percent">{(score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="prob-bar-bg">
                          <div 
                            className={`prob-bar-fill ${className === result.predicted_macro ? 'highlighted' : ''}`} 
                            style={{ width: `${score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="fine-grained-accordion">
                  <button 
                    className="btn-accordion-toggle" 
                    onClick={() => setShowFineGrained(!showFineGrained)}
                  >
                    {showFineGrained ? "Hide Fine-grained Silhouette Breakdown ▴" : "Show Fine-grained Silhouette Breakdown ▾"}
                  </button>
                  
                  {showFineGrained && (
                    <div className="probabilities-list fine-grained-list animate-fade">
                      {Object.entries(result.fine_probabilities).map(([className, score]) => (
                        <div key={className} className="probability-row sub-row">
                          <div className="prob-label-container">
                            <span className="prob-name sub-name">{className} <small style={{ opacity: 0.6, fontSize: '0.8em' }}>({MACRO_MAPPING[className]})</small></span>
                            <span className="prob-percent">{(score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="prob-bar-bg sub-bg">
                            <div 
                              className={`prob-bar-fill sub-fill ${className === result.predicted_fine ? 'highlighted-sub' : ''}`} 
                              style={{ width: `${score * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>Built with FastAPI (Python) & React (Vite). Single-process unified web server deployment.</p>
      </footer>
    </div>
  );
}

export default App;
