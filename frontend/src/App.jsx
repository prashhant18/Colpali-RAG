import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = "/api";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [uploadProgress, setUploadProgress] = useState(null); // {stage,current,total,filename}
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ------------------------------------------------------------------
  // Upload handling
  // ------------------------------------------------------------------
  const handleFiles = useCallback(async (files) => {
    setUploadError("");

    for (const file of files) {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setUploadError(`${file.name} is not a PDF.`);
        continue;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        setUploadProgress({
          stage: "uploading",
          current: 0,
          total: 0,
          filename: file.name,
        });
        const res = await fetch(`${API_BASE}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          // Validation errors (400/409/413) arrive as plain JSON before
          // streaming starts.
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || "Upload failed");
        }

        // Ingestion progress streams back as SSE events (same format as /ask).
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finished = false;
        let errorMessage = null;

        while (!finished && !errorMessage) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events (separated by blank lines)
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const event of events) {
            const line = event.trim();
            if (!line.startsWith("data:")) continue;
            const payload = JSON.parse(line.slice(5).trim());

            if (payload.type === "parsing") {
              setUploadProgress({
                stage: "parsing",
                current: 0,
                total: 0,
                filename: file.name,
              });
            } else if (payload.type === "progress") {
              setUploadProgress({
                stage: "embedding",
                current: payload.current,
                total: payload.total,
                filename: file.name,
              });
            } else if (payload.type === "done") {
              setUploadStatus(payload.message);
              finished = true;
              break;
            } else if (payload.type === "error") {
              errorMessage = payload.detail || "Ingestion failed";
              break;
            }
          }
        }

        if (errorMessage) throw new Error(errorMessage);
        if (!finished) throw new Error("Upload was interrupted.");
      } catch (err) {
        setUploadError(err.message);
      } finally {
        setTimeout(() => setUploadProgress(null), 500);
      }
    }
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  // ------------------------------------------------------------------
  // Chat / streaming
  // ------------------------------------------------------------------
  const sendMessage = useCallback(async () => {
    const question = input.trim();
    if (!question || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: question }]);

    // Add placeholder assistant message
    const assistantId = Date.now();
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", sources: [], id: assistantId },
    ]);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Request failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events (separated by blank lines)
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const event of events) {
          const line = event.trim();
          if (!line.startsWith("data:")) continue;
          const payload = JSON.parse(line.slice(5).trim());

          if (payload.type === "sources") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, sources: payload.sources } : m
              )
            );
          } else if (payload.type === "token") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + payload.content }
                  : m
              )
            );
          } else if (payload.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: `⚠️ ${payload.detail}`,
                      isError: true,
                    }
                  : m
              )
            );
            break;
          } else if (payload.type === "done") {
            break;
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Error: ${err.message}` }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  }, [input, isStreaming]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Upload progress bar helpers
  const uploadPercent =
    uploadProgress && uploadProgress.total > 0
      ? Math.round((uploadProgress.current / uploadProgress.total) * 100)
      : null;
  const uploadLabel = !uploadProgress
    ? ""
    : uploadProgress.stage === "uploading"
      ? `Uploading ${uploadProgress.filename}…`
      : uploadProgress.stage === "parsing"
        ? "Parsing PDF…"
        : `Embedding page ${uploadProgress.current} / ${uploadProgress.total}…`;

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div className="app">
      <header>
        <h1>📚 Research Paper RAG Assistant</h1>
        <p>Upload PDFs, ask questions, get cited answers.</p>
      </header>

      {/* Upload area */}
      <div
        className={`upload-section${isDragging ? " dragging" : ""}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        <div className="icon">📄</div>
        <p>
          Drag & drop PDFs here, or click to browse
          <br />
          <small>(multiple files supported)</small>
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          onChange={(e) => handleFiles(e.target.files)}
        />
        {uploadProgress && (
          <div className="upload-progress">
            <div className="upload-progress-label">{uploadLabel}</div>
            <div
              className={`progress-bar${uploadPercent === null ? " indeterminate" : ""}`}
            >
              {uploadPercent !== null && (
                <div
                  className="progress-fill"
                  style={{ width: `${uploadPercent}%` }}
                />
              )}
            </div>
            {uploadPercent !== null && (
              <div className="upload-progress-pct">{uploadPercent}%</div>
            )}
          </div>
        )}
        {uploadStatus && !uploadProgress && (
          <div className="upload-status">{uploadStatus}</div>
        )}
        {uploadError && <div className="upload-status error">{uploadError}</div>}
      </div>

      {/* Chat */}
      <div className="chat-section">
        <div className="messages">
          {messages.length === 0 && (
            <div className="typing">
              Upload research papers above, then ask a question about their
              content.
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message ${msg.role}${msg.isError ? " error" : ""}`}
            >
              {msg.content || (msg.role === "assistant" && isStreaming ? "…" : "")}
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <strong>Sources:</strong>
                  <ul>
                    {msg.sources.map((s, j) => (
                      <li key={j}>
                        {s.filename} —{" "}
                        <span className="page">p. {s.page}</span>{" "}
                        <span className="distance">
                          (score: {s.distance.toFixed(3)})
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about the papers…"
            disabled={isStreaming}
          />
          <button onClick={sendMessage} disabled={isStreaming || !input.trim()}>
            {isStreaming ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}