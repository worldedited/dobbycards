# 📚 Telegram Flashcards Bot - TMA Ready

Умный Telegram бот для изучения английских слов с AI-переводом и интерактивными карточками через Telegram Mini App.

## 🚀 Быстрый запуск

### 1. Настройка окружения

```bash
# Клонируйте проект (если еще не сделали)
cd dobbyTma

# Запустите скрипт установки
./start.sh
```

### 2. Настройка токенов

Отредактируйте файл `.env`:

```env
# Получите токен от @BotFather
BOT_TOKEN=your_telegram_bot_token_here

# Получите ключ от fireworks.ai
FIREWORKS_API_KEY=your_fireworks_api_key_here

# URL вашего веб-приложения (после деплоя)
WEB_APP_URL=https://your-webapp-url.com
```

### 3. Деплой Web App

Загрузите `web.html` на любой хостинг:

**GitHub Pages (бесплатно):**
1. Создайте репозиторий на GitHub
2. Загрузите `web.html` как `index.html`
3. Включите Pages в Settings → Pages
4. URL: `https://username.github.io/repo-name/`

**Vercel (бесплатно):**
```bash
npm i -g vercel
vercel --prod web.html
```

**Netlify (бесплатно):**
- Перетащите `web.html` на netlify.com

### 4. Настройка бота в Telegram

1. Откройте @BotFather
2. Выберите вашего бота → Bot Settings → Menu Button
3. Установите Web App URL: `https://your-webapp-url.com`

### 5. Запуск

```bash
# Запуск бота
python bot.py

# В другом терминале - запуск API сервера
python main.py
```

## 📱 Как работает TMA

### Структура проекта:
- `bot.py` - Telegram бот с командами и WebApp кнопками
- `main.py` - FastAPI сервер для API эндпоинтов
- `web.html` - Telegram Mini App интерфейс
- `models.py` - Модели базы данных
- `.env` - Конфигурация (токены, ключи)

### Интеграция с Telegram:
- ✅ Telegram WebApp API
- ✅ MainButton для добавления слов
- ✅ Тема Telegram (светлая/темная)
- ✅ Haptic Feedback
- ✅ Popup уведомления
- ✅ Данные пользователя из initData

## 🎮 Функции

### Для пользователей:
- 📝 Добавление слов через бота или Mini App
- 🎯 Интерактивные карточки с анимациями
- 👆 Свайпы для управления (влево - не знаю, вправо - знаю)
- 🔊 Произношение слов (TTS)
- 📊 Статистика изучения
- 🏆 Система достижений

### Технические:
- 🤖 AI-перевод через Fireworks AI
- 💾 SQLite/PostgreSQL база данных
- 🚀 Async/await архитектура
- 📱 Адаптивный дизайн
- 🎨 Анимации и эффекты

## 🔧 API Endpoints

```http
GET /api/words/{telegram_id}     # Получить слова пользователя
POST /api/words/{telegram_id}    # Добавить новое слово
PUT /api/words/update            # Обновить статус слова
GET /api/stats/{telegram_id}     # Статистика пользователя
GET /health                      # Проверка здоровья API
```

## 🐛 Отладка

### Проблемы с запуском:
```bash
# Проверьте зависимости
pip install -r requirements.txt

# Проверьте .env файл
cat .env

# Проверьте базу данных
python -c "from bot import DatabaseManager; import asyncio; asyncio.run(DatabaseManager('sqlite+aiosqlite:///flashcards.db').init_db())"
```

### Проблемы с Web App:
1. Проверьте URL в @BotFather
2. Убедитесь что `web.html` доступен по HTTPS
3. Проверьте консоль браузера (F12)

### Проблемы с API:
```bash
# Проверьте API сервер
curl http://localhost:8000/health

# Проверьте логи
python main.py
```

## 🌐 Деплой в продакшн

### Railway.app (рекомендуется):
1. Подключите GitHub репозиторий
2. Добавьте переменные окружения
3. Деплой автоматический

### VPS/Сервер:
```bash
# Установка
git clone your-repo
cd dobbyTma
./start.sh

# Запуск с PM2
npm install -g pm2
pm2 start bot.py --interpreter python3
pm2 start main.py --interpreter python3
```

## 📝 Что дальше?

1. **Настройте токены** в `.env`
2. **Задеплойте web.html** на хостинг
3. **Обновите WEB_APP_URL** в .env и @BotFather
4. **Запустите бота** и API сервер
5. **Протестируйте** Mini App в Telegram

## 🆘 Поддержка

Если что-то не работает:
1. Проверьте все токены и URL
2. Убедитесь что API сервер запущен
3. Проверьте логи в консоли
4. Web App должен быть доступен по HTTPS

Удачи с вашим Telegram Mini App! 🚀
