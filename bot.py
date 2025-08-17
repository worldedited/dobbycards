# bot.py
"""
Telegram Flashcards Bot - Main bot file
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func

from models import Base, User, Word, TranslationCache, UserStatistics, Achievement, UserAchievement
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def init_db(self):
        """Initialize database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_or_create_user(self, telegram_id: int, username: str = None, 
                                first_name: str = None, last_name: str = None) -> User:
        """Get or create user"""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                user = User(
                    telegram_id=telegram_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    created_at=datetime.utcnow()
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
                logger.info(f"Created new user: {telegram_id}")
            else:
                # Update user info
                user.username = username
                user.first_name = first_name
                user.last_name = last_name
                user.last_active = datetime.utcnow()
                await session.commit()
            
            return user
    
    async def add_word(self, telegram_id: int, word_data: Dict[str, Any]) -> Word:
        """Add a new word for user"""
        async with self.async_session() as session:
            # Get user
            user_result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = user_result.scalar_one()
            
            # Check if word already exists
            existing_result = await session.execute(
                select(Word).where(
                    Word.user_id == user.id,
                    Word.word == word_data['word'].lower()
                )
            )
            existing_word = existing_result.scalar_one_or_none()
            
            if existing_word:
                return existing_word
            
            # Create new word
            word = Word(
                user_id=user.id,
                word=word_data['word'].lower(),
                translation=word_data.get('translation'),
                pronunciation=word_data.get('pronunciation'),
                example=word_data.get('example'),
                example_translation=word_data.get('example_translation'),
                created_at=datetime.utcnow()
            )
            session.add(word)
            await session.commit()
            await session.refresh(word)
            
            logger.info(f"Added word '{word.word}' for user {telegram_id}")
            return word
    
    async def get_user_words(self, telegram_id: int, learned: Optional[bool] = None) -> List[Word]:
        """Get user's words"""
        async with self.async_session() as session:
            query = select(Word).join(User).where(User.telegram_id == telegram_id)
            
            if learned is not None:
                query = query.where(Word.learned == learned)
            
            result = await session.execute(query.order_by(Word.created_at.desc()))
            return result.scalars().all()
    
    async def update_word_status(self, word_id: int, learned: bool = None, difficulty: int = None):
        """Update word learning status"""
        async with self.async_session() as session:
            result = await session.execute(select(Word).where(Word.id == word_id))
            word = result.scalar_one()
            
            if learned is not None:
                word.learned = learned
                word.last_reviewed = datetime.utcnow()
                
                if learned:
                    word.correct_count += 1
                
                word.review_count += 1
            
            if difficulty is not None:
                word.difficulty = difficulty
            
            await session.commit()
    
    async def get_user_stats(self, telegram_id: int) -> Dict[str, Any]:
        """Get user statistics"""
        async with self.async_session() as session:
            user_result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = user_result.scalar_one()
            
            # Get word counts
            total_words_result = await session.execute(
                select(func.count(Word.id)).where(Word.user_id == user.id)
            )
            total_words = total_words_result.scalar() or 0
            
            learned_words_result = await session.execute(
                select(func.count(Word.id)).where(
                    Word.user_id == user.id,
                    Word.learned == True
                )
            )
            learned_words = learned_words_result.scalar() or 0
            
            return {
                'total_words': total_words,
                'learned_words': learned_words,
                'words_in_progress': total_words - learned_words,
                'accuracy_rate': (learned_words / total_words * 100) if total_words > 0 else 0
            }


class TranslationService:
    """AI translation service using Fireworks AI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    
    async def translate_word(self, word: str, target_language: str = "ru") -> Dict[str, str]:
        """Translate word using Fireworks AI"""
        
        prompt = f"""
        Переведи английское слово "{word}" на русский язык и предоставь следующую информацию в JSON формате:
        
        {{
            "translation": "основной перевод слова",
            "pronunciation": "транскрипция в формате [фонетика]",
            "example": "пример предложения на английском",
            "example_translation": "перевод примера на русский"
        }}
        
        Отвечай только JSON, без дополнительного текста.
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.1
                }
                
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Parse JSON response
                        try:
                            translation_data = json.loads(content)
                            return translation_data
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {content}")
                            return await self.fallback_translate(word)
                    else:
                        logger.error(f"Fireworks API error: {response.status}")
                        return await self.fallback_translate(word)
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return await self.fallback_translate(word)
    
    async def fallback_translate(self, word: str) -> Dict[str, str]:
        """Fallback translation using simple dictionary"""
        # Simple fallback - in production you might use another API
        return {
            "translation": f"перевод слова '{word}'",
            "pronunciation": f"[{word}]",
            "example": f"Example with {word}",
            "example_translation": f"Пример с {word}"
        }
    
    def generate_simple_pronunciation(self, word: str) -> str:
        """Generate simple pronunciation"""
        # Basic pronunciation rules
        word = word.lower()
        pronunciation = word
        
        # Simple replacements
        replacements = {
            'ph': 'f',
            'th': 'θ',
            'ch': 'tʃ',
            'sh': 'ʃ',
            'oo': 'u:',
            'ee': 'i:',
            'ea': 'i:',
        }
        
        for old, new in replacements.items():
            pronunciation = pronunciation.replace(old, new)
        
        return f"[{pronunciation}]"


class FlashcardsBot:
    """Main bot class"""
    
    def __init__(self):
        self.token = os.getenv('BOT_TOKEN')
        self.fireworks_api_key = os.getenv('FIREWORKS_API_KEY')
        self.web_app_url = os.getenv('WEB_APP_URL', 'https://your-webapp-url.com')
        self.database_url = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///flashcards.db')
        
        # Initialize services
        self.db = DatabaseManager(self.database_url)
        self.translator = TranslationService(self.fireworks_api_key)
        
        # Create application
        self.application = Application.builder().token(self.token).build()
        
        # Add error handler
        self.application.add_error_handler(self.error_handler)
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("cards", self.learn_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        # Create or get user
        await self.db.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        # Create Web App button
        web_app = WebAppInfo(url=self.web_app_url)
        keyboard = [
            [InlineKeyboardButton("📚 Открыть карточки", web_app=web_app)],
            [InlineKeyboardButton("➕ Добавить слово", callback_data="add_word")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = f"""
🎯 **Добро пожаловать в Flashcards Bot, {user.first_name}!**

Я помогу вам изучать английские слова с помощью интерактивных карточек.

**Что я умею:**
• 📝 Добавлять новые слова с AI-переводом
• 🎮 Интерактивные карточки в Mini App
• 📊 Отслеживать ваш прогресс
• 🔄 Система повторений для лучшего запоминания

**Как начать:**
1. Нажмите "📚 Открыть карточки" для изучения
2. Или отправьте мне английское слово для добавления

Удачи в изучении! 🚀
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📖 **Справка по командам:**

/start - Начать работу с ботом
/add [слово] - Добавить новое слово
/learn - Открыть карточки для изучения
/stats - Посмотреть статистику
/help - Показать эту справку

**Как использовать:**
• Просто отправьте английское слово, и я добавлю его с переводом
• Используйте Mini App для интерактивного изучения
• Отслеживайте прогресс в статистике

**Примеры:**
`hello` - добавить слово "hello"
`/add beautiful` - добавить слово "beautiful"
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def add_word_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command"""
        if not context.args:
            await update.message.reply_text(
                "📝 Отправьте слово для добавления:\n"
                "Например: `/add beautiful`",
                parse_mode='Markdown'
            )
            return
        
        word = ' '.join(context.args).strip().lower()
        await self.process_new_word(update, word)
    
    async def learn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /learn command"""
        web_app = WebAppInfo(url=self.web_app_url)
        keyboard = [[InlineKeyboardButton("📚 Открыть карточки", web_app=web_app)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🎮 Нажмите кнопку ниже, чтобы открыть интерактивные карточки:",
            reply_markup=reply_markup
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user statistics"""
        user = update.effective_user
        
        # Get user stats from database
        stats = await self.db.get_user_stats(user.id)
        
        stats_text = f"📊 *Ваша статистика*\n\n"
        stats_text += f"📚 Всего слов: {stats.get('total_words', 0)}\n"
        stats_text += f"✅ Изучено: {stats.get('learned_words', 0)}\n"
        stats_text += f"📖 В процессе: {stats.get('learning_words', 0)}\n"
        stats_text += f"🎯 Точность: {stats.get('accuracy', 0):.1f}%\n\n"
        stats_text += f"🔥 Серия: {stats.get('streak', 0)} дней"
        
        # Create web app button
        web_app = WebAppInfo(url=self.web_app_url)
        keyboard = [[InlineKeyboardButton("📚 Открыть карточки", web_app=web_app)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Handle both callback query and message
        if update.callback_query:
            await update.callback_query.edit_message_text(
                stats_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        elif update.message:
            await update.message.reply_text(
                stats_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (new words)"""
        text = update.message.text.strip()
        
        # Check if it's a valid English word
        if not text.replace(' ', '').replace('-', '').isalpha():
            await update.message.reply_text(
                "❌ Пожалуйста, отправьте корректное английское слово.\n"
                "Используйте только буквы, пробелы и дефисы."
            )
            return
        
        await self.process_new_word(update, text.lower(), context)
    
    async def process_new_word(self, update: Update, word: str, context: ContextTypes.DEFAULT_TYPE = None):
        """Process new word addition"""
        user = update.effective_user
        
        # Send "typing" action
        if context:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Check if word already exists
            existing_words = await self.db.get_user_words(user.id)
            if any(w.word == word for w in existing_words):
                await update.message.reply_text(f"📝 Слово '{word}' уже есть в вашей коллекции!")
                return
            
            # Translate word
            translation_data = await self.translator.translate_word(word)
            
            # Add to database
            word_obj = await self.db.add_word(user.id, {
                'word': word,
                **translation_data
            })
            
            # Send confirmation
            confirmation_text = f"""
✅ **Слово добавлено!**

🔤 **{word.title()}** {translation_data.get('pronunciation', '')}
🔄 **{translation_data.get('translation', 'перевод')}**

📝 *Пример:* {translation_data.get('example', '')}
🔄 *Перевод:* {translation_data.get('example_translation', '')}

Теперь вы можете изучать его в карточках! 🎮
            """
            
            web_app = WebAppInfo(url=self.web_app_url)
            keyboard = [
                [InlineKeyboardButton("📚 Открыть карточки", web_app=web_app)],
                [InlineKeyboardButton("➕ Добавить еще", callback_data="add_word")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                confirmation_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error processing word '{word}': {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при добавлении слова. Попробуйте еще раз."
            )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "add_word":
            await query.edit_message_text(
                "📝 Отправьте английское слово, которое хотите добавить:"
            )
        elif query.data == "stats":
            await self.stats_command(update, context)
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot"""
        logger.error(f"Exception while handling an update: {context.error}")
        
        # Try to send error message to user if possible
        if update and hasattr(update, 'effective_chat') and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ Произошла ошибка. Попробуйте еще раз или обратитесь к администратору."
                )
            except Exception as e:
                logger.error(f"Failed to send error message: {e}")
    
    async def run(self):
        """Run the bot"""
        # Initialize database
        await self.db.init_db()
        logger.info("Database initialized")
        
        # Start bot
        logger.info("Starting bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        try:
            # Keep running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        finally:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()


async def main():
    """Main function"""
    bot = FlashcardsBot()
    await bot.run()


if __name__ == '__main__':
    asyncio.run(main())
