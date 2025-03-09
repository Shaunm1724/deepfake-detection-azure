import { useState, ChangeEvent, JSX } from 'react';
import { FiUpload, FiAlertTriangle, FiCheck, FiFilm, FiLoader } from 'react-icons/fi';
import './App.css';

// Define types for our application state
interface ContentSafety {
  is_safe: boolean;
  reason?: string;
}

interface AnalysisResult {
  is_fake: boolean;
  confidence: number;
  prediction: 'fake' | 'real';
  content_safety?: ContentSafety;
}

function App(): JSX.Element {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string>('');

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError('');
      setResult(null);
    }
  };

  const handleUpload = async (): Promise<void> => {
    if (!selectedFile) {
      setError('Please select a video file first.');
      return;
    }

    if (!selectedFile.type.includes('video/')) {
      setError('Please select a valid video file.');
      return;
    }

    setIsLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      // Replace the URL with your Azure Function endpoint or API endpoint
      const response = await fetch('https://your-function-app.azurewebsites.net/api/ProcessVideo', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Failed to analyze video: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Deepfake Detection System</h1>
        <p>Upload a video to detect if it's real or manipulated</p>
      </header>

      <main className="main">
        <div className="upload-container">
          <div 
            className="upload-area" 
            onClick={() => document.getElementById('file-input')?.click()}
          >
            {previewUrl ? (
              <video 
                src={previewUrl} 
                controls 
                className="video-preview"
              />
            ) : (
              <div className="upload-placeholder">
                <FiUpload size={48} />
                <p>Drag & drop your video here or click to browse</p>
                <p className="upload-note">Supported formats: MP4, AVI, MOV</p>
              </div>
            )}
            <input
              id="file-input"
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>

          <button 
            className="upload-button" 
            onClick={handleUpload}
            disabled={!selectedFile || isLoading}
          >
            {isLoading ? <FiLoader className="spinner" /> : <FiFilm />} 
            {isLoading ? 'Analyzing...' : 'Analyze Video'}
          </button>

          {error && (
            <div className="error-message">
              <FiAlertTriangle /> {error}
            </div>
          )}
        </div>

        {result && (
          <div className={`result-container ${result.is_fake ? 'fake' : 'real'}`}>
            <h2>Analysis Results</h2>
            
            <div className="result-card">
              <div className="result-header">
                {result.is_fake ? (
                  <div className="result-fake">
                    <FiAlertTriangle size={32} />
                    <h3>Deepfake Detected</h3>
                  </div>
                ) : (
                  <div className="result-real">
                    <FiCheck size={32} />
                    <h3>No Manipulation Detected</h3>
                  </div>
                )}
              </div>
              
              <div className="result-details">
                <div className="detail-item">
                  <span className="detail-label">Confidence:</span>
                  <span className="detail-value">{result.confidence}%</span>
                </div>
                
                <div className="detail-item">
                  <span className="detail-label">Prediction:</span>
                  <span className="detail-value">
                    {result.prediction === 'fake' ? 'Manipulated Video' : 'Authentic Video'}
                  </span>
                </div>
                
                {result.content_safety && (
                  <div className="detail-item">
                    <span className="detail-label">Content Safety:</span>
                    <span className="detail-value">
                      {result.content_safety.is_safe ? 'No harmful content detected' : result.content_safety.reason}
                    </span>
                  </div>
                )}
              </div>
              
              {result.is_fake && (
                <div className="warning-note">
                  <FiAlertTriangle />
                  <p>This video appears to be manipulated with deepfake technology.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>© 2025 Deepfake Detection System | Powered by Azure Cloud & AI</p>
      </footer>
    </div>
  );
}

export default App;