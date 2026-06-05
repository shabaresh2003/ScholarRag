import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [messages, setMessages] = useState([]);
  const [queryText, setQueryText] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const API_BASE = 'http://localhost:8000';

  // Load documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  };

  const handleUpload = async (file) => {
    if (!file) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        await fetchDocuments();
        setSelectedPaper(data.filename); // Select uploaded paper by default
      } else {
        const errData = await res.json();
        alert(`Upload failed: ${errData.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error uploading file:', err);
      alert('Network error uploading file.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (e, filename) => {
    e.stopPropagation(); // Avoid selecting the deleted item
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      const res = await fetch(`${API_BASE}/documents/${filename}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        if (selectedPaper === filename) {
          setSelectedPaper(null);
        }
        fetchDocuments();
      }
    } catch (err) {
      console.error('Error deleting document:', err);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!queryText.trim() || isGenerating) return;

    const userQuery = queryText;
    setQueryText('');

    // Append user message
    const userMsgId = crypto.randomUUID();
    const newUserMessage = {
      id: userMsgId,
      sender: 'user',
      text: userQuery,
    };
    
    // Append blank assistant message to stream into
    const assistantMsgId = crypto.randomUUID();
    const newAssistantMessage = {
      id: assistantMsgId,
      sender: 'assistant',
      text: '',
      isStreaming: true,
      sources: [],
      citations: [],
    };

    setMessages((prev) => [...prev, newUserMessage, newAssistantMessage]);
    setIsGenerating(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userQuery,
          source_filter: selectedPaper || null,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json(); // returns CitationResponse schema
      
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { 
                ...msg, 
                text: data.answer || '', 
                citations: data.citations || [], 
                sources: data.sources || [], 
                image_url: data.image_url || null,
                isStreaming: false 
              }
            : msg
        )
      );

    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, text: `Error connecting to assistant: ${err.message}`, isStreaming: false }
            : msg
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  // Helper to format text with styled citation badge links
  const renderTextWithCitations = (text) => {
    if (!text) return null;
    
    // Regular expression to match brackets like [1], [2]
    const parts = text.split(/(\[\d+\])/g);
    
    return parts.map((part, idx) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const citationIndex = parseInt(match[1]) - 1;
        return (
          <span 
            key={idx} 
            className="citation-badge"
            title="Click to view citation details"
            onClick={() => {
              const element = document.getElementById(`citation-card-${citationIndex}`);
              if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.style.borderColor = '#6366f1';
                element.style.boxShadow = '0 0 12px rgba(99, 102, 241, 0.4)';
                setTimeout(() => {
                  element.style.borderColor = '';
                  element.style.boxShadow = '';
                }, 2000);
              }
            }}
          >
            {match[1]}
          </span>
        );
      }
      return part;
    });
  };

  return (
    <div className="app-container">
      {/* Sidebar - Document upload and list */}
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">S</div>
          <h1 className="logo-title">ScholarRAG</h1>
        </div>

        {/* Drag and drop upload */}
        <div 
          className={`upload-card ${dragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="file-input" 
            accept=".pdf"
            onChange={(e) => handleUpload(e.target.files?.[0])}
          />
          <div className="upload-icon">
            {isUploading ? '⚡' : '📂'}
          </div>
          <div className="upload-title">
            {isUploading ? 'Indexing paper...' : 'Upload PDF'}
          </div>
          <div className="upload-subtitle">
            {isUploading ? 'Reading, chunking, and embedding' : 'Drag & drop or click to browse'}
          </div>
        </div>

        <div className="doc-section-title">Research Papers</div>
        <div className="doc-list">
          <div 
            className={`doc-item ${selectedPaper === null ? 'selected' : ''}`}
            onClick={() => setSelectedPaper(null)}
          >
            <div className="doc-info">
              <span className="doc-icon">📚</span>
              <span className="doc-name">All Uploaded Papers</span>
            </div>
          </div>
          
          {documents.map((doc, idx) => (
            <div 
              key={idx}
              className={`doc-item ${selectedPaper === doc ? 'selected' : ''}`}
              onClick={() => setSelectedPaper(doc)}
            >
              <div className="doc-info">
                <span className="doc-icon">📄</span>
                <span className="doc-name" title={doc}>{doc}</span>
              </div>
              <button 
                className="doc-delete-btn" 
                onClick={(e) => handleDelete(e, doc)}
                title="Delete document"
              >
                ✕
              </button>
            </div>
          ))}

          {documents.length === 0 && !isUploading && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '20px' }}>
              No papers uploaded yet. Upload a PDF above to begin!
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Workspace */}
      <main className="chat-space">
        <header className="chat-header">
          <div className="header-title-container">
            <h2 className="header-title">Study Assistant</h2>
            <div className="header-subtitle">
              {selectedPaper ? `Scoped to: ${selectedPaper}` : 'Searching across all research papers'}
            </div>
          </div>
          <div className="status-badge">
            <span className="status-dot"></span>
            Langfuse Monitored
          </div>
        </header>

        {/* Chat Logs */}
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">🎓</div>
              <h3 className="empty-title">Welcome to your ScholarRAG Workspace</h3>
              <p className="empty-desc">
                Upload research papers (PDF) in the sidebar. Ask complex questions about methodology, findings, or literature, and receive streaming answers with strict, verifiable citation citations.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
              <div className="message-bubble">
                {msg.sender === 'user' ? (
                  <p>{msg.text}</p>
                ) : (
                  <>
                    <div className="markdown-content">
                      {msg.image_url && (
                        <div className="message-image-container" style={{ margin: '12px 0', borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0' }}>
                          <img 
                            src={msg.image_url} 
                            alt="Retrieved Multimodal Context" 
                            style={{ maxWidth: '100%', maxHeight: '300px', objectFit: 'contain', display: 'block' }} 
                          />
                        </div>
                      )}
                      <ReactMarkdown
                        components={{
                          // Intercept text nodes to convert citation labels [1], [2], etc., to styled interactive buttons
                          p: ({ children }) => {
                            if (typeof children === 'string') {
                              return <p>{renderTextWithCitations(children)}</p>;
                            }
                            // If children contains elements, process nested string segments
                            const processed = React.Children.map(children, (child) => {
                              if (typeof child === 'string') {
                                return renderTextWithCitations(child);
                              }
                              return child;
                            });
                            return <p>{processed}</p>;
                          },
                          li: ({ children }) => {
                            const processed = React.Children.map(children, (child) => {
                              if (typeof child === 'string') {
                                return renderTextWithCitations(child);
                              }
                              return child;
                            });
                            return <li>{processed}</li>;
                          }
                        }}
                      >
                        {msg.text}
                      </ReactMarkdown>
                    </div>

                    {/* Citations Expandable Section */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-footer">
                        <div className="citations-footer-title">Verified Citations</div>
                        {msg.citations.map((cite, cIdx) => (
                          <div 
                            key={cIdx} 
                            id={`citation-card-${cIdx}`}
                            className="citation-card"
                          >
                            <div className="citation-card-header">
                              <span style={{ fontWeight: 800 }}>[{cIdx + 1}]</span>
                              <span className="citation-card-source" title={cite.source}>
                                {cite.source}
                              </span>
                              <span className="citation-card-page">
                                Page {cite.page}
                              </span>
                            </div>
                            <div className="citation-card-snippet">
                              "{cite.snippet}"
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className="chat-input-container">
          <form className="chat-input-form" onSubmit={handleSend}>
            <input 
              type="text" 
              className="chat-input"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder={
                documents.length === 0 
                  ? "Upload a PDF in the sidebar to start asking questions..." 
                  : `Ask a question about ${selectedPaper ? selectedPaper : 'your research papers'}...`
              }
              disabled={documents.length === 0 || isGenerating}
            />
            <button 
              type="submit" 
              className="send-btn"
              disabled={!queryText.trim() || isGenerating || documents.length === 0}
            >
              {isGenerating ? '⚡' : '➔'}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
