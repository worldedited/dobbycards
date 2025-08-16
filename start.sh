#!/bin/bash
# start.sh - Startup script for Telegram Flashcards Bot

set -e

echo "🚀 Starting Telegram Flashcards Bot..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your tokens before running again!"
    echo "   - BOT_TOKEN: Get from @BotFather"
    echo "   - FIREWORKS_API_KEY: Get from fireworks.ai"
    echo "   - WEB_APP_URL: Your deployed web app URL"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if database exists, if not create it
if [ ! -f "flashcards.db" ]; then
    echo "🗄️  Initializing database..."
    python -c "
import asyncio
from bot import DatabaseManager
import os
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///flashcards.db')

async def init_db():
    db = DatabaseManager(db_url)
    await db.init_db()
    print('Database initialized successfully!')

asyncio.run(init_db())
"
fi

echo "✅ Setup complete!"
echo ""
echo "🤖 To start the bot: python bot.py"
echo "🌐 To start the API server: python main.py"
echo "🔧 To run both: python bot.py & python main.py"
echo ""
echo "📱 Don't forget to:"
echo "   1. Set up your Web App URL in @BotFather"
echo "   2. Deploy web.html to your hosting service"
echo "   3. Update WEB_APP_URL in .env"
