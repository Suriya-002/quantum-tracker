# API Documentation

## Base URL
\\\
http://localhost:5000/api/v1
\\\

## Endpoints

### News Articles

#### Get All News
\\\
GET /api/v1/news
\\\

**Query Parameters:**
- \category\ (optional): Filter by category (news, company, research)
- \source\ (optional): Filter by source name
- \sentiment\ (optional): Filter by sentiment (positive, negative, neutral)
- \limit\ (optional): Number of articles (default: 50)
- \offset\ (optional): Pagination offset (default: 0)

**Response:**
\\\json
{
  "success": true,
  "count": 50,
  "data": [
    {
      "id": 1,
      "title": "Article Title",
      "description": "Article description",
      "url": "https://example.com/article",
      "source": "Quantum Computing Report",
      "category": "news",
      "published_date": "2024-02-01T10:00:00",
      "sentiment_score": 0.85,
      "sentiment_label": "positive"
    }
  ]
}
\\\

#### Get Single Article
\\\
GET /api/v1/news/:id
\\\

#### Get News Sources
\\\
GET /api/v1/news/sources
\\\

### Stock Data

#### Get All Stocks
\\\
GET /api/v1/stocks
\\\

**Response:**
\\\json
{
  "success": true,
  "count": 9,
  "data": [
    {
      "ticker": "IONQ",
      "company_name": "IonQ Inc",
      "price": 12.34,
      "change": 0.45,
      "change_percent": 3.78,
      "volume": 1234567,
      "market_cap": 2500000000,
      "pe_ratio": 45.6,
      "timestamp": "2024-02-01T16:00:00"
    }
  ]
}
\\\

#### Get Single Stock
\\\
GET /api/v1/stocks/:ticker
\\\

#### Get Stock History
\\\
GET /api/v1/stocks/:ticker/history?limit=100
\\\

### Companies

#### Get All Companies
\\\
GET /api/v1/companies
\\\

#### Get Single Company
\\\
GET /api/v1/companies/:id
\\\

### Sentiment Analysis

#### Analyze Text
\\\
POST /api/v1/sentiment/analyze
Content-Type: application/json

{
  "text": "Quantum computing breakthrough announced!"
}
\\\

**Response:**
\\\json
{
  "success": true,
  "sentiment": {
    "label": "positive",
    "score": 0.95
  }
}
\\\

## Health Check
\\\
GET /health
\\\

Returns API health status.
