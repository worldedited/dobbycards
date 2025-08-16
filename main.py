# main.py
"""
FastAPI server for Telegram Flashcards Bot API
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func

from models import Base, User, Word, TranslationCache
from bot import TranslationService, DatabaseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Telegram Flashcards Bot API",
    description="API for Telegram Mini App flashcards bot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
database_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///flashcards.db')
fireworks_api_key = os.getenv('FIREWORKS_API_KEY')

db_manager = DatabaseManager(database_url)
translator = TranslationService(fireworks_api_key) if fireworks_api_key else None

# Pydantic models
class WordCreate(BaseModel):
    word: str

class WordResponse(BaseModel):
    id: int
    word: str
    translation: Optional[str]
    pronunciation: Optional[str]
    example: Optional[str]
    example_translation: Optional[str] = None
    learned: bool = False
    difficulty: int = 0
    created_at: datetime

    class Config:
        from_attributes = True

class WordUpdate(BaseModel):
    learned: Optional[bool] = None
    difficulty: Optional[int] = None

class StatsResponse(BaseModel):
    total_words: int
    learned_words: int
    words_in_progress: int
    accuracy_rate: float

# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await db_manager.init_db()

@app.get("/")
async def root():
    """Serve the web app"""
    return FileResponse('web.html')

@app.get("/api/words/{telegram_id}", response_model=List[WordResponse])
async def get_words(telegram_id: int, learned: Optional[bool] = None):
    """Get user's words"""
    try:
        words = await db_manager.get_user_words(telegram_id, learned)
        return [
            WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                pronunciation=word.pronunciation,
                example=word.example,
                example_translation=word.example_translation,
                learned=word.learned,
                difficulty=word.difficulty,
                created_at=word.created_at
            )
            for word in words
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/words/{telegram_id}", response_model=WordResponse)
async def add_word(telegram_id: int, word_data: WordCreate, background_tasks: BackgroundTasks):
    """Add a new word"""
    try:
        # Create user if doesn't exist
        await db_manager.get_or_create_user(telegram_id)
        
        # Check if word already exists
        existing_words = await db_manager.get_user_words(telegram_id)
        if any(w.word == word_data.word.lower() for w in existing_words):
            raise HTTPException(status_code=400, detail="Word already exists")
        
        # Translate word if translator is available
        translation_data = {}
        if translator:
            translation_data = await translator.translate_word(word_data.word)
        else:
            # Fallback translation
            translation_data = {
                'translation': f'перевод слова "{word_data.word}"',
                'pronunciation': f'[{word_data.word}]',
                'example': f'Example with {word_data.word}',
                'example_translation': f'Пример с {word_data.word}'
            }
        
        # Add word to database
        word = await db_manager.add_word(telegram_id, {
            'word': word_data.word.lower(),
            **translation_data
        })
        
        return WordResponse(
            id=word.id,
            word=word.word,
            translation=word.translation,
            pronunciation=word.pronunciation,
            example=word.example,
            example_translation=word.example_translation,
            learned=word.learned,
            difficulty=word.difficulty,
            created_at=word.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/words/update")
async def update_word(word_update: WordUpdate, word_id: int):
    """Update word status"""
    try:
        await db_manager.update_word_status(
            word_id=word_id,
            learned=word_update.learned,
            difficulty=word_update.difficulty
        )
        return {"message": "Word updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/{telegram_id}", response_model=StatsResponse)
async def get_stats(telegram_id: int):
    """Get user statistics"""
    try:
        stats = await db_manager.get_user_stats(telegram_id)
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "database": "connected" if db_manager else "disconnected",
        "translator": "available" if translator else "unavailable"
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    
    uvicorn.run(app, host=host, port=port, reload=True)
