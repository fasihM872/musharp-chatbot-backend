## muSharp Chatbot Backend

### Run locally
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="YOUR_KEY"
$env:SITE_URL="https://musharp.com"
$env:FRONTEND_ORIGINS="https://musharp.com,https://www.musharp.com"
python -m uvicorn ai_respodner.AI_chatbot.chatbot_backend:app --host 0.0.0.0 --port 8000
```

### Endpoints
- GET /health
- POST /chat
- GET /debug/content
- GET /debug/search?query=...

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")