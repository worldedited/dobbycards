# api_docs.py
"""
FastAPI application with complete API documentation
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import os
from enum import Enum

# Create FastAPI app with metadata
app = FastAPI(
    title="Telegram Flashcards Bot API",
    description="""
    ## 📚 Telegram Flashcards Bot API
    
    This API provides backend services for the Telegram Flashcards Bot, 
    enabling word management, translation services, and learning statistics.
    
    ### Features:
    - 🔐 JWT Authentication
    - 📝 Word Management (CRUD operations)
    - 🌐 AI-powered translations
    - 📊 Learning statistics and progress tracking
    - 🏆 Achievements system
    - 🔄 Spaced repetition algorithm
    - 📱 WebSocket support for real-time updates
    
    ### Authentication:
    Most endpoints require a valid JWT token. Obtain a token using the `/auth/telegram` endpoint.
    """,
    version="1.0.0",
    terms_of_service="https://flashcards.bot/terms",
    contact={
        "name": "Support Team",
        "url": "https://flashcards.bot/support",
        "email": "support@flashcards.bot",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {"url": "https://api.flashcards.bot", "description": "Production server"},
        {"url": "https://staging-api.flashcards.bot", "description": "Staging server"},
        {"url": "http://localhost:8000", "description": "Development server"},
    ],
    tags_metadata=[
        {
            "name": "Authentication",
            "description": "Authentication endpoints for obtaining and refreshing tokens",
        },
        {
            "name": "Words",
            "description": "Operations with words - add, update, delete, and retrieve",
        },
        {
            "name": "Translation",
            "description": "AI-powered translation services",
        },
        {
            "name": "Statistics",
            "description": "User statistics and learning progress",
        },
        {
            "name": "Achievements",
            "description": "User achievements and gamification",
        },
        {
            "name": "Categories",
            "description": "Word categories management",
        },
        {
            "name": "Export",
            "description": "Data export and import operations",
        },
        {
            "name": "WebSocket",
            "description": "Real-time updates via WebSocket",
        },
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Pydantic Models ---

class UserBase(BaseModel):
    """Base user model"""
    telegram_id: int = Field(..., description="Telegram user ID")
    username: Optional[str] = Field(None, description="Telegram username")
    first_name: Optional[str] = Field(None, description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    language_code: Optional[str] = Field("en", description="User's language code")

class UserCreate(UserBase):
    """User creation model"""
    pass

class UserResponse(UserBase):
    """User response model"""
    id: int
    created_at: datetime
    is_premium: bool = False
    words_count: int = 0
    achievements_count: int = 0
    
    class Config:
        orm_mode = True

class WordBase(BaseModel):
    """Base word model"""
    word: str = Field(..., min_length=1, max_length=100, description="English word")
    translation: Optional[str] = Field(None, description="Translation")
    pronunciation: Optional[str] = Field(None, description="IPA pronunciation")
    example: Optional[str] = Field(None, description="Example sentence")
    example_translation: Optional[str] = Field(None, description="Example translation")

class WordCreate(BaseModel):
    """Word creation model"""
    word: str = Field(..., min_length=1, max_length=100, description="English word to add")
    category_id: Optional[int] = Field(None, description="Category ID")
    
    @validator('word')
    def validate_word(cls, v):
        if not v.replace(' ', '').replace('-', '').isalpha():
            raise ValueError('Word must contain only letters, spaces, and hyphens')
        return v.lower().strip()

class WordUpdate(BaseModel):
    """Word update model"""
    translation: Optional[str] = None
    learned: Optional[bool] = None
    difficulty: Optional[int] = Field(None, ge=0, le=5)
    
class WordResponse(WordBase):
    """Word response model"""
    id: int
    user_id: int
    learned: bool = False
    difficulty: int = 0
    review_count: int = 0
    correct_count: int = 0
    last_reviewed: Optional[datetime] = None
    next_review: Optional[datetime] = None
    created_at: datetime
    audio_url: Optional[str] = None
    
    class Config:
        orm_mode = True

class TranslationRequest(BaseModel):
    """Translation request model"""
    word: str = Field(..., description="Word to translate")
    target_language: str = Field("ru", description="Target language code")
    include_audio: bool = Field(False, description="Include audio pronunciation")

class TranslationResponse(BaseModel):
    """Translation response model"""
    word: str
    translation: str
    pronunciation: str
    example: str
    example_translation: str
    audio_url: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1, description="Translation confidence score")

class StatisticsResponse(BaseModel):
    """User statistics response"""
    total_words: int = 0
    learned_words: int = 0
    words_in_progress: int = 0
    total_reviews: int = 0
    correct_answers: int = 0
    accuracy_rate: float = 0.0
    current_streak: int = 0
    longest_streak: int = 0
    study_time_minutes: int = 0
    achievements_earned: int = 0
    level: int = 1
    experience_points: int = 0
    daily_goal: int = 10
    daily_progress: int = 0

class AchievementResponse(BaseModel):
    """Achievement response model"""
    id: int
    name: str
    description: str
    icon: str
    points: int
    earned: bool = False
    earned_at: Optional[datetime] = None
    progress: float = Field(0.0, ge=0, le=1)

class CategoryResponse(BaseModel):
    """Category response model"""
    id: int
    name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    words_count: int = 0

class ReviewQuality(int, Enum):
    """Review quality for spaced repetition"""
    FORGOT = 0
    HARD = 1
    GOOD = 2
    EASY = 3

class ReviewRequest(BaseModel):
    """Review request for spaced repetition"""
    word_id: int
    quality: ReviewQuality
    time_spent_seconds: int = Field(..., ge=0)

class ExportFormat(str, Enum):
    """Export format options"""
    JSON = "json"
    CSV = "csv"
    ANKI = "anki"
    QUIZLET = "quizlet"

class ImportRequest(BaseModel):
    """Import request model"""
    format: ExportFormat
    data: str = Field(..., description="Base64 encoded data or URL")
    merge_duplicates: bool = Field(True, description="Merge duplicate words")

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class TelegramAuthData(BaseModel):
    """Telegram authentication data"""
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    photo_url: Optional[str] = None
    auth_date: int
    hash: str

# --- Dependency Functions ---

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Here you would fetch the user from database
        return {"id": user_id, "telegram_id": payload.get("telegram_id")}
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- API Endpoints ---

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Telegram Flashcards Bot API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Root"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "database": "operational",
            "translation": "operational",
            "cache": "operational"
        }
    }

# --- Authentication Endpoints ---

@app.post("/auth/telegram", response_model=TokenResponse, tags=["Authentication"])
async def authenticate_telegram(auth_data: TelegramAuthData):
    """
    Authenticate user with Telegram data
    
    This endpoint verifies the Telegram authentication data and returns a JWT token.
    The hash should be calculated according to Telegram's authentication protocol.
    """
    # Verify Telegram auth hash (implementation needed)
    # Create or get user from database
    # Generate JWT token
    
    access_token = jwt.encode(
        {
            "sub": auth_data.id,
            "telegram_id": auth_data.id,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "id": 1,
            "telegram_id": auth_data.id,
            "username": auth_data.username,
            "first_name": auth_data.first_name,
            "last_name": auth_data.last_name,
            "created_at": datetime.utcnow(),
            "is_premium": False,
            "words_count": 0,
            "achievements_count": 0
        }
    }

@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """Refresh access token"""
    access_token = jwt.encode(
        {
            "sub": current_user["id"],
            "telegram_id": current_user["telegram_id"],
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": current_user
    }

# --- Words Endpoints ---

@app.get("/words", response_model=List[WordResponse], tags=["Words"])
async def get_words(
    skip: int = 0,
    limit: int = 100,
    learned: Optional[bool] = None,
    category_id: Optional[int] = None,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's words with filtering and pagination
    
    - **skip**: Number of words to skip
    - **limit**: Maximum number of words to return
    - **learned**: Filter by learned status
    - **category_id**: Filter by category
    - **search**: Search words by text
    """
    # Implementation would fetch from database
    return []

@app.post("/words", response_model=WordResponse, tags=["Words"])
async def create_word(
    word: WordCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Add a new word to user's collection
    
    The word will be automatically translated using AI.
    """
    # Implementation would:
    # 1. Check if word already exists
    # 2. Get translation from AI service
    # 3. Save to database
    # 4. Return created word
    return {
        "id": 1,
        "user_id": current_user["id"],
        "word": word.word,
        "translation": "перевод",
        "pronunciation": "[wɜːrd]",
        "example": "Example sentence",
        "example_translation": "Пример предложения",
        "learned": False,
        "difficulty": 0,
        "review_count": 0,
        "correct_count": 0,
        "created_at": datetime.utcnow()
    }

@app.get("/words/{word_id}", response_model=WordResponse, tags=["Words"])
async def get_word(
    word_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific word by ID"""
    # Implementation would fetch from database
    return {
        "id": word_id,
        "user_id": current_user["id"],
        "word": "example",
        "translation": "пример",
        "pronunciation": "[ɪɡˈzɑːmpl]",
        "example": "This is an example",
        "example_translation": "Это пример",
        "learned": False,
        "difficulty": 0,
        "review_count": 0,
        "correct_count": 0,
        "created_at": datetime.utcnow()
    }

@app.put("/words/{word_id}", response_model=WordResponse, tags=["Words"])
async def update_word(
    word_id: int,
    word_update: WordUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a word's properties"""
    # Implementation would update in database
    pass

@app.delete("/words/{word_id}", tags=["Words"])
async def delete_word(
    word_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a word from collection"""
    return {"message": "Word deleted successfully"}

@app.post("/words/{word_id}/review", tags=["Words"])
async def review_word(
    word_id: int,
    review: ReviewRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit a review for spaced repetition
    
    This endpoint updates the word's repetition schedule based on the review quality.
    """
    # Implementation would use SM-2 algorithm
    return {"next_review": datetime.utcnow() + timedelta(days=1)}

# --- Translation Endpoints ---

@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_word(
    request: TranslationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Translate a word using AI service
    
    This endpoint uses Fireworks AI with Dobby model for translation.
    """
    return {
        "word": request.word,
        "translation": "перевод",
        "pronunciation": "[wɜːrd]",
        "example": "Example sentence",
        "example_translation": "Пример предложения",
        "confidence": 0.95
    }

@app.get("/translate/cache/{word}", response_model=TranslationResponse, tags=["Translation"])
async def get_cached_translation(word: str):
    """Get translation from cache if available"""
    # Implementation would check cache first
    pass

# --- Statistics Endpoints ---

@app.get("/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get user's learning statistics"""
    return StatisticsResponse(
        total_words=100,
        learned_words=45,
        words_in_progress=55,
        total_reviews=500,
        correct_answers=400,
        accuracy_rate=0.8,
        current_streak=7,
        longest_streak=30,
        study_time_minutes=1200,
        achievements_earned=15,
        level=5,
        experience_points=2500,
        daily_goal=10,
        daily_progress=7
    )

@app.get("/statistics/daily", tags=["Statistics"])
async def get_daily_statistics(
    date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get daily statistics for a specific date"""
    target_date = date or datetime.utcnow().date()
    return {
        "date": target_date,
        "words_learned": 5,
        "words_reviewed": 20,
        "correct_answers": 18,
        "study_time_minutes": 45,
        "streak_maintained": True
    }

@app.get("/statistics/weekly", tags=["Statistics"])
async def get_weekly_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get weekly statistics summary"""
    return {
        "week_start": datetime.utcnow().date(),
        "total_words_learned": 35,
        "total_reviews": 150,
        "average_accuracy": 0.85,
        "study_days": 6,
        "total_study_time_minutes": 300
    }

# --- Achievements Endpoints ---

@app.get("/achievements", response_model=List[AchievementResponse], tags=["Achievements"])
async def get_achievements(
    current_user: dict = Depends(get_current_user)
):
    """Get all achievements with user's progress"""
    return [
        {
            "id": 1,
            "name": "First Word",
            "description": "Learn your first word",
            "icon": "🎯",
            "points": 10,
            "earned": True,
            "earned_at": datetime.utcnow(),
            "progress": 1.0
        },
        {
            "id": 2,
            "name": "Week Streak",
            "description": "Maintain a 7-day streak",
            "icon": "🔥",
            "points": 50,
            "earned": False,
            "progress": 0.5
        }
    ]

@app.get("/achievements/{achievement_id}", response_model=AchievementResponse, tags=["Achievements"])
async def get_achievement(
    achievement_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get specific achievement details"""
    pass

# --- Categories Endpoints ---

@app.get("/categories", response_model=List[CategoryResponse], tags=["Categories"])
async def get_categories():
    """Get all available word categories"""
    return [
        {
            "id": 1,
            "name": "Basic",
            "description": "Common everyday words",
            "icon": "📝",
            "color": "#667eea",
            "words_count": 150
        },
        {
            "id": 2,
            "name": "Business",
            "description": "Business vocabulary",
            "icon": "💼",
            "color": "#4a5568",
            "words_count": 200
        }
    ]

@app.post("/categories", response_model=CategoryResponse, tags=["Categories"])
async def create_category(
    name: str,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create a custom category (premium feature)"""
    pass

# --- Export/Import Endpoints ---

@app.get("/export", tags=["Export"])
async def export_data(
    format: ExportFormat = ExportFormat.JSON,
    include_learned: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """
    Export user's words in various formats
    
    Supported formats:
    - **JSON**: Standard JSON format
    - **CSV**: Comma-separated values
    - **Anki**: Anki flashcard format
    - **Quizlet**: Quizlet compatible format
    """
    # Implementation would generate file in requested format
    return FileResponse(
        path="export.json",
        media_type="application/json",
        filename=f"flashcards_export_{datetime.utcnow().date()}.json"
    )

@app.post("/import", tags=["Export"])
async def import_data(
    file: UploadFile = File(...),
    format: ExportFormat = ExportFormat.JSON,
    merge_duplicates: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Import words from file"""
    # Implementation would parse file and import words
    return {
        "imported": 50,
        "duplicates": 5,
        "errors": 0,
        "message": "Import completed successfully"
    }

# --- WebSocket Endpoint ---

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict

class ConnectionManager:
    """WebSocket connection manager"""
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """
    WebSocket endpoint for real-time updates
    
    Use this for:
    - Real-time word updates
    - Achievement notifications
    - Study reminders
    - Collaborative features
    """
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Process received data
            await manager.send_personal_message(f"Echo: {data}", user_id)
    except WebSocketDisconnect:
        manager.disconnect(user_id)

# --- Admin Endpoints (Optional) ---

@app.get("/admin/users", tags=["Admin"])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get all users (admin only)"""
    # Check if user is admin
    # Return users list
    pass

@app.get("/admin/statistics", tags=["Admin"])
async def get_platform_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get platform-wide statistics (admin only)"""
    return {
        "total_users": 10000,
        "active_users_today": 2500,
        "total_words": 1000000,
        "total_translations": 5000000,
        "average_words_per_user": 100,
        "platform_accuracy_rate": 0.82
    }

# Mount static files for documentation assets
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)