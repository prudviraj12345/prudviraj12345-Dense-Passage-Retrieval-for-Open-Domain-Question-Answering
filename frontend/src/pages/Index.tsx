import { useState } from "react";

const Index = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setAnswer(data.answer || "No answer found.");
    } catch {
      setAnswer(
        "Backend not reachable. This is expected on GitHub Pages unless backend is deployed."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial, sans-serif" }}>
      <h1>Dense Passage Retrieval</h1>
      <p>Open-Domain Question Answering System</p>

      <textarea
        placeholder="Ask a question (e.g., What is AI?)"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "100%", height: "80px", marginTop: "10px" }}
      />

      <br /><br />

      <button onClick={handleAsk} disabled={loading}>
        {loading ? "Searching..." : "Get Answer"}
      </button>

      <h3 style={{ marginTop: "20px" }}>Answer</h3>
      <p>{answer || "Your answer will appear here."}</p>
    </div>
  );
};

export default Index;