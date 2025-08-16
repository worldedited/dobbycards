# models.py
"""
Database models for Telegram Flashcards Bot
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, BigInteger, ForeignKey, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    language_code = Column(String(10), default='ru')
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_active = Column(DateTime, default=func.now())
    
    # Relationships
    words = relationship("Word", back_populates="user", cascade="all, delete-orphan")
    statistics = relationship("UserStatistics", back_populates="user", cascade="all, delete-orphan")
    achievements = relationship("UserAchievement", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, username={self.username})>"


class Word(Base):
    """Word model"""
    __tablename__ = 'words'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    word = Column(String(100), nullable=False, index=True)
    translation = Column(String(200), nullable=True)
    pronunciation = Column(String(100), nullable=True)
    example = Column(Text, nullable=True)
    example_translation = Column(Text, nullable=True)
    learned = Column(Boolean, default=False, index=True)
    difficulty = Column(Integer, default=0)
    review_count = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    last_reviewed = Column(DateTime, nullable=True)
    next_review = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Spaced repetition fields
    easiness_factor = Column(Float, default=2.5)
    repetition_interval = Column(Integer, default=1)
    repetition_count = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    
    # Audio support
    audio_url = Column(String(500), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="words")
    categories = relationship("WordCategoryMapping", back_populates="word", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Word(word={self.word}, translation={self.translation}, learned={self.learned})>"


class TranslationCache(Base):
    """Translation cache model"""
    __tablename__ = 'translation_cache'
    
    word = Column(String(100), primary_key=True)
    translation = Column(String(200), nullable=True)
    pronunciation = Column(String(100), nullable=True)
    example = Column(Text, nullable=True)
    example_translation = Column(Text, nullable=True)
    source = Column(String(50), default='fireworks')
    created_at = Column(DateTime, default=func.now(), index=True)
    hit_count = Column(Integer, default=0)
    audio_url = Column(String(500), nullable=True)
    
    def __repr__(self):
        return f"<TranslationCache(word={self.word}, translation={self.translation})>"


class UserStatistics(Base):
    """User daily statistics"""
    __tablename__ = 'user_statistics'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False, index=True)
    words_learned = Column(Integer, default=0)
    words_reviewed = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    wrong_answers = Column(Integer, default=0)
    streak_count = Column(Integer, default=0)
    study_time_minutes = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="statistics")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f"<UserStatistics(user_id={self.user_id}, date={self.date})>"


class Achievement(Base):
    """Achievement definitions"""
    __tablename__ = 'achievements'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(String(200), nullable=True)
    icon = Column(String(10), nullable=True)
    requirement_type = Column(String(50), nullable=False)  # words_learned, streak, etc.
    requirement_value = Column(Integer, nullable=False)
    points = Column(Integer, default=10)
    
    # Relationships
    user_achievements = relationship("UserAchievement", back_populates="achievement", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Achievement(name={self.name}, points={self.points})>"


class UserAchievement(Base):
    """User earned achievements"""
    __tablename__ = 'user_achievements'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    achievement_id = Column(Integer, ForeignKey('achievements.id', ondelete='CASCADE'), nullable=False)
    earned_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="achievements")
    achievement = relationship("Achievement", back_populates="user_achievements")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f"<UserAchievement(user_id={self.user_id}, achievement_id={self.achievement_id})>"


class UserPreferences(Base):
    """User preferences and settings"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True)
    daily_goal = Column(Integer, default=10)
    notification_enabled = Column(Boolean, default=True)
    notification_time = Column(Time, nullable=True)
    theme = Column(String(20), default='light')
    sound_enabled = Column(Boolean, default=True)
    autoflip_enabled = Column(Boolean, default=False)
    review_algorithm = Column(String(20), default='sm2')
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    def __repr__(self):
        return f"<UserPreferences(user_id={self.user_id}, daily_goal={self.daily_goal})>"


class WordCategory(Base):
    """Word categories"""
    __tablename__ = 'word_categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(String(200), nullable=True)
    icon = Column(String(10), nullable=True)
    color = Column(String(7), nullable=True)  # Hex color
    
    # Relationships
    words = relationship("WordCategoryMapping", back_populates="category", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WordCategory(name={self.name})>"


class WordCategoryMapping(Base):
    """Many-to-many mapping between words and categories"""
    __tablename__ = 'word_category_mapping'
    
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), primary_key=True)
    category_id = Column(Integer, ForeignKey('word_categories.id', ondelete='CASCADE'), primary_key=True)
    
    # Relationships
    word = relationship("Word", back_populates="categories")
    category = relationship("WordCategory", back_populates="words")
    
    def __repr__(self):
        return f"<WordCategoryMapping(word_id={self.word_id}, category_id={self.category_id})>"


class AudioFile(Base):
    """Cached audio files"""
    __tablename__ = 'audio_files'
    
    id = Column(Integer, primary_key=True)
    word = Column(String(100), nullable=False)
    language = Column(String(10), default='en')
    voice = Column(String(50), nullable=True)
    file_path = Column(String(500), nullable=True)
    url = Column(String(500), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f"<AudioFile(word={self.word}, language={self.language})>"
