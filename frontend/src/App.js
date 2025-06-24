import React, { useState, useRef } from "react";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const resultRef = useRef(null);
  const [excelUrl, setExcelUrl] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      if (selectedFile.type.startsWith('image/')) {
        setImageUrl(URL.createObjectURL(selectedFile));
      } else {
        setImageUrl(""); // Clear image preview for PDFs
      }
      setAnalysis(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setAnalysis(null);

    const formData = new FormData();
    formData.append("file", file);
    if (question) {
      formData.append("question", question);
    }

    try {
      const response = await fetch("http://localhost:8000/analyze/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setAnalysis(result);
    } catch (error) {
      console.error("Error during analysis:", error);
      setAnalysis({
        error: `Failed to analyze document. Error: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleExcelDownload = async () => {
    if (!file) return;
    setLoading(true);
    setExcelUrl(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("http://localhost:8000/extract_sanction_excel/", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Failed to generate Excel file");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setExcelUrl(url);
      // Auto-download
      const a = document.createElement('a');
      a.href = url;
      a.download = "sanction_letter.xlsx";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (error) {
      alert("Error downloading Excel: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const renderClauseSection = (title, clauses) => {
    if (!clauses || clauses.length === 0) return null;
    return (
      <div className="clause-section">
        <h3>{title}</h3>
        <ul>
          {clauses.map((clause, index) => (
            <li key={index}>{clause}</li>
          ))}
        </ul>
      </div>
    );
  };

  const renderRisks = (risks) => {
    if (!risks || risks.length === 0) return null;
    return (
      <div className="risks-section">
        <h3>⚠️ Potential Risks</h3>
        <ul>
          {risks.map((risk, index) => (
            <li key={index}>
              <strong>{risk.description}</strong>
              <p>{risk.clause}</p>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Contract Analysis System</h1>
        <p>Upload contracts or legal documents for analysis</p>
      </header>
      <main className="App-main">
        <div className="controls">
          <input 
            type="file" 
            accept="image/*,.pdf" 
            onChange={handleFileChange} 
          />
          <div className="question-input">
            <input
              type="text"
              placeholder="Ask a question about the contract..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          <button 
            onClick={handleUpload} 
            disabled={!file || loading}
          >
            {loading ? "Analyzing..." : "Analyze Document"}
          </button>
          <button
            onClick={handleExcelDownload}
            disabled={!file || loading}
            style={{ marginLeft: 10 }}
          >
            Download Excel
          </button>
          {excelUrl && (
            <a href={excelUrl} download="sanction_letter.xlsx" style={{ marginLeft: 10 }}>
              Click here if download does not start
            </a>
          )}
        </div>
        
        <div className="content">
          <div className="document-view">
            <h2>Document Preview</h2>
            {imageUrl ? (
              <img src={imageUrl} alt="Document preview" />
            ) : (
              <div className="placeholder">
                {file ? "PDF preview not available" : "Upload a document to begin"}
              </div>
            )}
          </div>
          
          <div className="analysis-view">
            <h2>Analysis Results</h2>
            {loading ? (
              <div className="loading">Analyzing document...</div>
            ) : analysis ? (
              <div className="analysis-content">
                {analysis.error ? (
                  <div className="error">{analysis.error}</div>
                ) : (
                  <>
                    {analysis.answer && (
                      <div className="answer-section">
                        <h3>Answer to Your Question</h3>
                        <p>{analysis.answer}</p>
                      </div>
                    )}
                    
                    {renderRisks(analysis.risks)}
                    
                    <div className="clauses-container">
                      <h3>Important Clauses</h3>
                      {renderClauseSection("Penalty Clauses", analysis.clauses?.penalty)}
                      {renderClauseSection("Termination Clauses", analysis.clauses?.termination)}
                      {renderClauseSection("Payment Terms", analysis.clauses?.payment)}
                      {renderClauseSection("Confidentiality", analysis.clauses?.confidentiality)}
                      {renderClauseSection("Liability", analysis.clauses?.liability)}
                    </div>
                    
                    <div className="full-text">
                      <h3>Full Extracted Text</h3>
                      <pre ref={resultRef}>{analysis.full_text}</pre>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="placeholder">
                Upload a document and click 'Analyze Document' to begin
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
