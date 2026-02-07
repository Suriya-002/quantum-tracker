import { useState, useEffect } from 'react'

function MarkdownText({ text }) {
  const parseMarkdown = (content) => {
    const lines = content.split('\n')
    const elements = []
    
    lines.forEach((line, i) => {
      // Headers (## text)
      if (line.startsWith('## ')) {
        elements.push(
          <h3 key={i} style={{ fontSize: '20px', fontWeight: 'bold', color: '#667eea', marginTop: '24px', marginBottom: '12px' }}>
            {line.replace('## ', '').replace(/[📊🏢🚀💰📈]/g, '')}
          </h3>
        )
      }
      // Bullet points (• text)
      else if (line.startsWith('• ')) {
        const boldRegex = /\*\*(.*?)\*\*/g
        const text = line.substring(2)
        const parts = []
        let lastIndex = 0
        let match
        
        while ((match = boldRegex.exec(text)) !== null) {
          if (match.index > lastIndex) {
            parts.push(text.substring(lastIndex, match.index))
          }
          parts.push(<strong key={match.index}>{match[1]}</strong>)
          lastIndex = match.index + match[0].length
        }
        if (lastIndex < text.length) {
          parts.push(text.substring(lastIndex))
        }
        
        elements.push(
          <div key={i} style={{ marginLeft: '20px', marginBottom: '10px', fontSize: '15px' }}>
            • {parts}
          </div>
        )
      }
      // Italic lines (*text*)
      else if (line.startsWith('*') && !line.startsWith('**')) {
        elements.push(
          <div key={i} style={{ fontStyle: 'italic', color: '#666', fontSize: '14px', marginBottom: '16px' }}>
            {line.replace(/\*/g, '')}
          </div>
        )
      }
      // Section headers (# text)
      else if (line.startsWith('# ')) {
        elements.push(
          <h2 key={i} style={{ fontSize: '24px', fontWeight: 'bold', color: '#333', marginBottom: '8px' }}>
            {line.replace('# ', '')}
          </h2>
        )
      }
      // Regular text
      else if (line.trim() && !line.startsWith('#')) {
        elements.push(
          <p key={i} style={{ marginBottom: '12px', fontSize: '15px', lineHeight: '1.7', color: '#333' }}>
            {line}
          </p>
        )
      }
    })
    
    return elements
  }
  
  return <div>{parseMarkdown(text)}</div>
}

function App() {
  const [news, setNews] = useState([])
  const [stocks, setStocks] = useState([])
  const [portfolio, setPortfolio] = useState(null)
  const [aiSummary, setAiSummary] = useState('')
  const [predictions, setPredictions] = useState('')
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 300000)
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      const newsRes = await fetch('https://quantum-tracker-api-4g12.onrender.com/api/v1/news?limit=20')
      const newsData = await newsRes.json()
      if (newsData.success) setNews(newsData.data)

      const stocksRes = await fetch('https://quantum-tracker-api-4g12.onrender.com/api/v1/stocks')
      const stocksData = await stocksRes.json()
      if (stocksData.success) setStocks(stocksData.data)

      const summaryRes = await fetch('https://quantum-tracker-api-4g12.onrender.com/api/v1/analysis/news-summary')
      const summaryData = await summaryRes.json()
      if (summaryData.success) {
        setAiSummary(summaryData.summary)
        if (summaryData.predictions) setPredictions(summaryData.predictions)
      }

      const portfolioRes = await fetch('https://quantum-tracker-api-4g12.onrender.com/api/v1/analysis/portfolio')
      const portfolioData = await portfolioRes.json()
      if (portfolioData.success) setPortfolio(portfolioData.data)

    } catch (err) {
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getAllocationColor = (allocation) => {
    const colors = {
      'OVERWEIGHT': { bg: '#d1fae5', text: '#065f46' },
      'MARKET_WEIGHT': { bg: '#dbeafe', text: '#1e40af' },
      'UNDERWEIGHT': { bg: '#fef3c7', text: '#92400e' },
      'AVOID': { bg: '#fee2e2', text: '#991b1b' }
    }
    return colors[allocation] || { bg: '#f3f4f6', text: '#666' }
  }

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚛️</div>
          <p style={{ fontSize: '18px' }}>Loading quantum data...</p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      <header style={{ background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', padding: '24px', position: 'sticky', top: 0, zIndex: 1000 }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ fontSize: '48px' }}>⚛️</span>
            <div>
              <h1 style={{ margin: 0, fontSize: '36px', background: 'linear-gradient(135deg, #667eea, #764ba2)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: 'bold' }}>Quantum Tracker</h1>
              <p style={{ margin: 0, color: '#666', fontSize: '14px' }}>Real-time Quantum Computing Industry News & Portfolio Analysis</p>
            </div>
            <div style={{ marginLeft: 'auto' }}>
              <span style={{ background: '#10b981', color: 'white', padding: '8px 16px', borderRadius: '20px', fontSize: '14px', fontWeight: '600' }}>🟢 Live</span>
            </div>
          </div>
        </div>
      </header>

      <div style={{ maxWidth: '1400px', margin: '24px auto', padding: '0 24px' }}>
        <div style={{ display: 'flex', gap: '12px', background: 'rgba(255, 255, 255, 0.95)', padding: '8px', borderRadius: '12px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          {['overview', 'portfolio', 'stocks', 'news'].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{ flex: 1, padding: '12px 24px', border: 'none', borderRadius: '8px', background: activeTab === tab ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'transparent', color: activeTab === tab ? 'white' : '#666', fontWeight: '600', fontSize: '16px', cursor: 'pointer' }}>
              {tab === 'overview' ? '📊 Overview' : tab === 'portfolio' ? '💼 Portfolio' : tab === 'stocks' ? '📈 Stocks' : '📰 News'}
            </button>
          ))}
        </div>
      </div>

      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 24px 48px' }}>
        {activeTab === 'overview' && (
          <>
            {aiSummary && (
              <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', marginBottom: '24px' }}>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '20px' }}>
                  <span style={{ fontSize: '32px' }}>🤖</span>
                  <h2 style={{ margin: 0, fontSize: '24px', color: '#333' }}>AI Market Analysis</h2>
                </div>
                <MarkdownText text={aiSummary} />
              </div>
            )}

            {predictions && (
              <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', marginBottom: '24px' }}>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '20px' }}>
                  <span style={{ fontSize: '32px' }}>📈</span>
                  <h2 style={{ margin: 0, fontSize: '24px', color: '#333' }}>Stock Predictions</h2>
                </div>
                <MarkdownText text={predictions} />
              </div>
            )}
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
                <h2 style={{ margin: '0 0 20px 0', fontSize: '20px', color: '#333' }}>📈 Top Stock Movers</h2>
                {stocks.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {stocks.slice().sort((a, b) => Math.abs(b.change_percent) - Math.abs(a.change_percent)).slice(0, 5).map(stock => (
                      <div key={stock.ticker} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', background: '#f9fafb', borderRadius: '8px' }}>
                        <div>
                          <strong style={{ fontSize: '16px' }}>{stock.ticker}</strong>
                          <div style={{ fontSize: '13px', color: '#666' }}>{stock.company_name}</div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontSize: '18px', fontWeight: 'bold' }}>${stock.price?.toFixed(2)}</div>
                          <div style={{ fontSize: '14px', fontWeight: '600', color: stock.change_percent >= 0 ? '#10b981' : '#ef4444' }}>
                            {stock.change_percent >= 0 ? '▲' : '▼'} {Math.abs(stock.change_percent)?.toFixed(2)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (<p style={{ color: '#666' }}>No stock data</p>)}
              </div>

              <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
                <h2 style={{ margin: '0 0 20px 0', fontSize: '20px', color: '#333' }}>📰 Latest Headlines</h2>
                {news.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {news.slice(0, 5).map((article, idx) => (
                      <div key={idx} style={{ borderBottom: idx < 4 ? '1px solid #e5e7eb' : 'none', paddingBottom: '16px' }}>
                        <a href={article.url} target="_blank" rel="noopener noreferrer" style={{ color: '#667eea', textDecoration: 'none', fontWeight: '600', fontSize: '15px', lineHeight: '1.4', display: 'block', marginBottom: '4px' }}>{article.title}</a>
                        <div style={{ fontSize: '12px', color: '#999' }}>{article.source}</div>
                      </div>
                    ))}
                  </div>
                ) : (<p style={{ color: '#666' }}>No news</p>)}
              </div>
            </div>
          </>
        )}

        {activeTab === 'portfolio' && portfolio && (
          <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
            <h2 style={{ margin: '0 0 12px 0', fontSize: '24px', color: '#333' }}>💼 ML-Optimized Portfolio Analysis</h2>
            <p style={{ color: '#666', fontSize: '15px', margin: '0 0 16px 0' }}>{portfolio.summary}</p>
            
            {portfolio.portfolio_metrics && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px', padding: '20px', background: 'linear-gradient(135deg, #667eea15, #764ba215)', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>Expected Return</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#10b981' }}>+{portfolio.portfolio_metrics.expected_return_pct}%</div>
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>Portfolio Volatility</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>{portfolio.portfolio_metrics.volatility_pct}%</div>
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>Sharpe Ratio</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#764ba2' }}>{portfolio.portfolio_metrics.sharpe_ratio}</div>
                </div>
              </div>
            )}
            
            {portfolio.methodology && (
              <div style={{ padding: '12px 16px', background: '#f9fafb', borderRadius: '8px', marginBottom: '24px', fontSize: '13px', color: '#666' }}>
                <strong>Methodology:</strong> {portfolio.methodology}
              </div>
            )}
            {portfolio.recommendations && portfolio.recommendations.length > 0 ? (
              <div style={{ display: 'grid', gap: '16px' }}>
                {portfolio.recommendations.map((rec, idx) => {
                  const colors = getAllocationColor(rec.allocation)
                  return (
                    <div key={idx} style={{ background: '#fff', border: '2px solid #e5e7eb', borderRadius: '12px', padding: '20px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                        <div>
                          <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{rec.ticker}</div>
                          <div style={{ fontSize: '14px', color: '#666' }}>${rec.current_price?.toFixed(2)}</div>
                        </div>
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                          <div style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '14px', fontWeight: '600', background: colors.bg, color: colors.text }}>{rec.allocation.replace('_', ' ')}</div>
                          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>{rec.weight}%</div>
                        </div>
                      </div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', padding: '16px', background: '#f9fafb', borderRadius: '8px', marginBottom: '12px' }}>
                        <div>
                          <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Expected Return</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: rec.expected_return > 0 ? '#10b981' : '#ef4444' }}>
                            {rec.expected_return > 0 ? '+' : ''}{rec.expected_return}%
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Volatility</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold' }}>{rec.volatility}%</div>
                        </div>
                        <div>
                          <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Sharpe Ratio</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: rec.sharpe > 1 ? '#10b981' : rec.sharpe > 0.5 ? '#667eea' : '#666' }}>{rec.sharpe}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Direction Accuracy</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: rec.direction_accuracy > 60 ? '#10b981' : '#ef4444' }}>{rec.direction_accuracy}%</div>
                        </div>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', background: 'linear-gradient(135deg, #667eea10, #764ba210)', borderRadius: '8px', fontSize: '13px' }}>
                        <div>
                          <span style={{ color: '#666' }}>🤖 ML Model:</span>
                          <strong style={{ marginLeft: '8px', color: '#667eea' }}>Ridge + Random Forest Ensemble</strong>
                        </div>
                        <div>
                          <span style={{ color: '#666' }}>📊 Training Data:</span>
                          <strong style={{ marginLeft: '8px' }}>{rec.model_metrics?.train_samples || 0} samples</strong>
                        </div>
                        <div>
                          <span style={{ color: '#666' }}>✅ Test Data:</span>
                          <strong style={{ marginLeft: '8px' }}>{rec.model_metrics?.test_samples || 0} samples</strong>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (<p style={{ textAlign: 'center', color: '#666', padding: '40px' }}>Insufficient data</p>)}
          </div>
        )}

        {activeTab === 'stocks' && (
          <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
            <h2 style={{ margin: '0 0 24px 0', fontSize: '24px', color: '#333' }}>Quantum Computing Stocks ({stocks.length})</h2>
            {stocks.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
                {stocks.map(stock => (
                  <div key={stock.ticker} style={{ background: '#fff', border: '2px solid #e5e7eb', borderRadius: '12px', padding: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <div><div style={{ fontSize: '24px', fontWeight: 'bold' }}>{stock.ticker}</div><div style={{ fontSize: '13px', color: '#666' }}>{stock.company_name}</div></div>
                      <div style={{ padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', background: stock.change_percent >= 0 ? '#d1fae5' : '#fee2e2', color: stock.change_percent >= 0 ? '#065f46' : '#991b1b' }}>
                        {stock.change_percent >= 0 ? '▲' : '▼'} {Math.abs(stock.change_percent)?.toFixed(2)}%
                      </div>
                    </div>
                    <div style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '16px' }}>${stock.price?.toFixed(2)}</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '13px' }}>
                      <div><div style={{ color: '#999' }}>Change</div><div style={{ fontWeight: '600', color: stock.change >= 0 ? '#10b981' : '#ef4444' }}>${Math.abs(stock.change)?.toFixed(2)}</div></div>
                      <div><div style={{ color: '#999' }}>Volume</div><div style={{ fontWeight: '600' }}>{stock.volume ? (stock.volume >= 1e6 ? `${(stock.volume / 1e6).toFixed(2)}M` : `${(stock.volume / 1e3).toFixed(2)}K`) : 'N/A'}</div></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (<p style={{ color: '#666', textAlign: 'center', padding: '40px' }}>No stock data</p>)}
          </div>
        )}

        {activeTab === 'news' && (
          <div style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '16px', padding: '24px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
            <h2 style={{ margin: '0 0 24px 0', fontSize: '24px', color: '#333' }}>Quantum Industry News ({news.length})</h2>
            {news.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {news.map((article, idx) => (
                  <div key={idx} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '12px', padding: '20px' }}>
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
                      <span style={{ padding: '4px 12px', borderRadius: '12px', fontSize: '12px', fontWeight: '600', background: '#f3f4f6', color: '#666' }}>{article.source}</span>
                    </div>
                    <a href={article.url} target="_blank" rel="noopener noreferrer" style={{ color: '#667eea', textDecoration: 'none', fontSize: '18px', fontWeight: '600', lineHeight: '1.4', display: 'block', marginBottom: '12px' }}>{article.title}</a>
                    {article.description && (<p style={{ color: '#666', fontSize: '14px', lineHeight: '1.6', margin: 0 }}>{article.description.substring(0, 200)}...</p>)}
                  </div>
                ))}
              </div>
            ) : (<p style={{ color: '#666', textAlign: 'center', padding: '40px' }}>No news</p>)}
          </div>
        )}
      </main>
    </div>
  )
}

export default App







// Force redeploy
