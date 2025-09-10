#!/bin/bash

# Create necessary directories
mkdir -p data/documents data/chroma

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
ANTHROPIC_API_KEY=your_anthropic_key
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_key

# Model Settings
DEFAULT_MODEL=claude-3-haiku-20240307
EMBEDDING_MODEL=all-MiniLM-L6-v2
SENTIMENT_MODEL=ProsusAI/finbert

# Database
CHROMA_PERSIST_DIR=./data/chroma
DOCUMENT_STORE_DIR=./data/documents

# Server
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
EOL
    echo "Please update the API keys in .env file"
fi

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install

echo "Setup complete! Next steps:"
echo "1. Update API keys in .env file"
echo "2. Start backend: uvicorn app.main:app --reload"
echo "3. Start frontend: cd frontend && npm run dev" 