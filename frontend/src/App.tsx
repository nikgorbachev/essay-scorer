import { useState } from 'react'
import axios from 'axios'
import './App.css'

// Define the shape of the data based on your Python backend models
interface ScoreResponse {
  numeric_score: number;
  actfl_label: string;
  confidence: number;
}

interface InsightsResponse {
  metrics: Record<string, number>; // A dictionary of string keys to numbers
  feedback: string[];
}

function App() {
  // We explicitly tell React what kind of data to expect
  const [text, setText] = useState<string>('')
  const [scoreData, setScoreData] = useState<ScoreResponse | null>(null)
  const [insightsData, setInsightsData] = useState<InsightsResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      // TypeScript now knows exactly what 'response.data' looks like
      const scoreRes = await axios.post<ScoreResponse>('http://127.0.0.1:8000/score', { text });
      setScoreData(scoreRes.data);

      const insightsRes = await axios.post<InsightsResponse>('http://127.0.0.1:8000/insights', { text });
      setInsightsData(insightsRes.data);
      
    } catch (error) {
      console.error("Error connecting to backend:", error);
      alert("Backend not reachable. Make sure FastAPI is running!");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
      <h1>L2 Essay Scorer</h1>
      
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Введите текст эссе здесь..."
        rows={10}
        style={{ width: '100%', padding: '10px', fontSize: '16px' }}
      />
      
      <div style={{ marginTop: '20px' }}>
        <button 
          onClick={handleAnalyze} 
          disabled={loading || !text}
          style={{ padding: '10px 20px', fontSize: '18px', cursor: 'pointer' }}
        >
          {loading ? 'Analyzing...' : 'Grade Essay'}
        </button>
      </div>

      {scoreData && (
        <div style={{ marginTop: '30px', border: '1px solid #ddd', padding: '20px', borderRadius: '8px' }}>
          <h2>Proficiency Assessment</h2>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2c3e50' }}>
            ACTFL Level: {scoreData.actfl_label} (Level {scoreData.numeric_score})
          </div>
          <p style={{ color: '#7f8c8d' }}>Model Confidence (QWK): {scoreData.confidence}</p>
        </div>
      )}

      {insightsData && (
        <div style={{ marginTop: '20px', border: '1px solid #ddd', padding: '20px', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
          <h3>Linguistic Insights (Feature Analysis)</h3>
          <ul>
            {insightsData.feedback.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '15px' }}>
            {/* We map over the metrics dictionary dynamically */}
            {Object.entries(insightsData.metrics).map(([key, value]) => (
               <div key={key}><strong>{key}:</strong> {value}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App