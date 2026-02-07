import anthropic
import os
import logging

logger = logging.getLogger('quantum_tracker')

def get_anthropic_client():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.warning('ANTHROPIC_API_KEY not set')
        return None
    return anthropic.Anthropic(api_key=api_key)

def summarize_news_with_llm(articles):
    if not articles:
        return "No quantum computing news available."
    
    try:
        client = get_anthropic_client()
        if client is None:
            return generate_basic_summary(articles)
        
        # Prepare detailed news context
        news_text = "\n\n".join([
            f"**Article {i+1}:** {article.get('title', '')}\n**Source:** {article.get('source', '')}\n**Content:** {article.get('description', '')[:400]}"
            for i, article in enumerate(articles[:15])
        ])
        
        prompt = f"""You are a senior quantum computing industry analyst providing insights for institutional investors and hedge funds.

Analyze these recent quantum computing industry news articles and provide a comprehensive, structured summary:

**STRUCTURE YOUR ANALYSIS AS FOLLOWS:**

**📊 MARKET OVERVIEW (2-3 sentences)**
- Overall industry momentum and sentiment
- Key trends driving the sector

**🏢 COMPANY HIGHLIGHTS (3-4 bullet points)**
- Major company announcements (funding, partnerships, product launches)
- Specific companies: IonQ, Rigetti, D-Wave, IBM Quantum, Google Quantum, PsiQuantum, Atom Computing, etc.
- Include dollar amounts and specifics

**🚀 TECHNOLOGICAL BREAKTHROUGHS (2-3 bullet points)**
- Key technical advances (qubit counts, error rates, new architectures)
- Commercial applications and use cases

**💰 FINANCIAL & STRATEGIC MOVES (2-3 bullet points)**
- Funding rounds, M&A activity, partnerships
- Geographic expansion, market entry strategies

**📈 INVESTMENT IMPLICATIONS (2-3 sentences)**
- What this means for quantum computing stocks
- Sector outlook and timing considerations
- Risk factors to monitor

NEWS ARTICLES:
{news_text}

Provide a professional, data-driven analysis suitable for investment decision-making. Use specific numbers, company names, and timelines."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        summary = message.content[0].text.strip()
        logger.info(f'Generated structured summary: {len(summary)} chars')
        return summary
        
    except Exception as e:
        logger.error(f'LLM error: {e}')
        return generate_basic_summary(articles)

def predict_stock_impact(articles, stocks):
    """Predict stock price movements based on news sentiment"""
    
    if not articles or not stocks:
        return None
    
    try:
        client = get_anthropic_client()
        if client is None:
            return None
        
        # Get stock tickers
        tickers = [s['ticker'] for s in stocks]
        
        # Prepare news with company mentions
        news_by_company = {}
        for article in articles[:20]:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            company_map = {
                'ionq': 'IONQ',
                'rigetti': 'RGTI',
                'd-wave': 'QBTS',
                'ibm quantum': 'IBM',
                'google quantum': 'GOOGL',
                'microsoft quantum': 'MSFT'
            }
            
            for keyword, ticker in company_map.items():
                if keyword in text and ticker in tickers:
                    if ticker not in news_by_company:
                        news_by_company[ticker] = []
                    news_by_company[ticker].append(article.get('title', ''))
        
        if not news_by_company:
            return None
        
        # Create prediction prompt
        news_summary = "\n\n".join([
            f"**{ticker}:** {len(articles)} mentions\n" + "\n".join(f"- {title}" for title in articles[:3])
            for ticker, articles in news_by_company.items()
        ])
        
        prompt = f"""As a quantitative analyst, predict SHORT-TERM (1-5 day) stock price movement based on this quantum computing news:

{news_summary}

For each mentioned stock, provide:
1. **Direction:** BULLISH / NEUTRAL / BEARISH
2. **Confidence:** HIGH / MEDIUM / LOW
3. **Catalysts:** Key factors driving prediction (1 sentence)
4. **Price Target:** Expected % move (+5%, -3%, etc.)

Format as:
**TICKER:** Direction (Confidence) | Expected: +X%
Catalysts: [brief explanation]

Focus on NEWS SENTIMENT, company-specific developments, and competitive positioning."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        prediction = message.content[0].text.strip()
        logger.info(f'Generated stock predictions: {len(prediction)} chars')
        return prediction
        
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        return None

def generate_basic_summary(articles):
    return f"Tracking {len(articles)} quantum computing news articles."

def analyze_sentiment(text, use_cache=False):
    return {'label': 'NEUTRAL', 'score': 0.5}
