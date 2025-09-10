import re
import requests
import textwrap
import threading
import time
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class GeminiResponder:
    def __init__(self, api_key, site_url):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.knowledge_chunks = []
        self.site_url = site_url
        self.load_wordpress_site()  # initial load

        # Start background thread to refresh every 12 hours
        thread = threading.Thread(target=self._auto_refresh, daemon=True)
        thread.start()

    def _chunk_text(self, text, chunk_size=1500, overlap=150):
        """Split text into overlapping chunks for context search"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def load_wordpress_site(self, chunk_size=1500, overlap=150):
        """Fetch ALL content from WordPress REST API (with pagination) and clean them"""
        try:
            all_texts = []
            
            # 1. Fetch Pages (paginate)
            print("📄 Fetching pages...")
            page_num = 1
            while True:
                pages_url = f"{self.site_url}/wp-json/wp/v2/pages?per_page=100&page={page_num}"
            pages_response = requests.get(
                pages_url,
                    timeout=15,
                headers={"User-Agent": "MusharpChatbotBot/1.0"},
            )
                if pages_response.status_code == 400 and 'rest_post_invalid_page_number' in pages_response.text:
                    break
            pages_response.raise_for_status()
            pages = pages_response.json()
                if not pages:
                    break
            for page in pages:
                title = page.get("title", {}).get("rendered", "")
                content = page.get("content", {}).get("rendered", "")
                excerpt = page.get("excerpt", {}).get("rendered", "")
                slug = page.get("slug", "")
                combined = f"PAGE: {title}\nSlug: {slug}\nExcerpt: {excerpt}\nContent: {content}"
                clean_text = re.sub(r"<[^>]+>", "", combined)
                all_texts.append(clean_text.strip())
                page_num += 1

            # 2. Fetch Posts (Blogs) (paginate)
            print("📝 Fetching blog posts...")
            post_page = 1
            while True:
                posts_url = f"{self.site_url}/wp-json/wp/v2/posts?per_page=100&page={post_page}"
            posts_response = requests.get(
                posts_url,
                    timeout=15,
                headers={"User-Agent": "MusharpChatbotBot/1.0"},
            )
                if posts_response.status_code == 400 and 'rest_post_invalid_page_number' in posts_response.text:
                    break
            posts_response.raise_for_status()
            posts = posts_response.json()
                if not posts:
                    break
            for post in posts:
                title = post.get("title", {}).get("rendered", "")
                content = post.get("content", {}).get("rendered", "")
                excerpt = post.get("excerpt", {}).get("rendered", "")
                slug = post.get("slug", "")
                date = post.get("date", "")
                categories = post.get("categories", [])
                
                # Get category names
                cat_names = []
                for cat_id in categories:
                    try:
                        cat_url = f"{self.site_url}/wp-json/wp/v2/categories/{cat_id}"
                        cat_resp = requests.get(cat_url, timeout=5, headers={"User-Agent": "MusharpChatbotBot/1.0"})
                        if cat_resp.status_code == 200:
                            cat_data = cat_resp.json()
                            cat_names.append(cat_data.get("name", ""))
                    except:
                        pass
                
                combined = f"BLOG POST: {title}\nSlug: {slug}\nDate: {date}\nCategories: {', '.join(cat_names)}\nExcerpt: {excerpt}\nContent: {content}"
                clean_text = re.sub(r"<[^>]+>", "", combined)
                all_texts.append(clean_text.strip())
                post_page += 1

            # 3. Fetch Custom Post Types (if any)
            print("🔧 Fetching custom content...")
            try:
                # Try to fetch common custom post types
                custom_types = ["services", "products", "testimonials", "team", "portfolio"]
                for custom_type in custom_types:
                    custom_url = f"{self.site_url}/wp-json/wp/v2/{custom_type}?per_page=100"
                    custom_response = requests.get(
                        custom_url,
                        timeout=10,
                        headers={"User-Agent": "MusharpChatbotBot/1.0"},
                    )
                    if custom_response.status_code == 200:
                        custom_posts = custom_response.json()
                        for post in custom_posts:
                            title = post.get("title", {}).get("rendered", "")
                            content = post.get("content", {}).get("rendered", "")
                            excerpt = post.get("excerpt", {}).get("rendered", "")
                            combined = f"{custom_type.upper()}: {title}\nExcerpt: {excerpt}\nContent: {content}"
                            clean_text = re.sub(r"<[^>]+>", "", combined)
                            all_texts.append(clean_text.strip())
            except Exception as e:
                print(f"⚠️ Custom post types not available: {e}")

            # 4. Fetch Site Info (title, description, etc.)
            print("ℹ️ Fetching site information...")
            try:
                site_url = f"{self.site_url}/wp-json/wp/v2"
                site_response = requests.get(
                    site_url,
                    timeout=10,
                    headers={"User-Agent": "MusharpChatbotBot/1.0"},
                )
                if site_response.status_code == 200:
                    site_data = site_response.json()
                    site_info = f"SITE INFO: {site_data.get('name', '')}\nDescription: {site_data.get('description', '')}\nURL: {site_data.get('url', '')}"
                    all_texts.append(site_info)
            except Exception as e:
                print(f"⚠️ Site info not available: {e}")

            # 5. Fetch Menus (for navigation and contact info)
            print("📋 Fetching menus...")
            try:
                menus_url = f"{self.site_url}/wp-json/wp/v2/menus"
                menus_response = requests.get(
                    menus_url,
                    timeout=10,
                    headers={"User-Agent": "MusharpChatbotBot/1.0"},
                )
                if menus_response.status_code == 200:
                    menus = menus_response.json()
                    for menu in menus:
                        menu_name = menu.get("name", "")
                        menu_items = menu.get("items", [])
                        menu_text = f"MENU: {menu_name}\n"
                        for item in menu_items:
                            menu_text += f"- {item.get('title', '')}: {item.get('url', '')}\n"
                        all_texts.append(menu_text.strip())
            except Exception as e:
                print(f"⚠️ Menus not available: {e}")

            # 6. Try to fetch contact info from common WordPress plugins
            print("📞 Fetching contact information...")
            try:
                # Try Contact Form 7
                cf7_url = f"{self.site_url}/wp-json/contact-form-7/v1/contact-forms"
                cf7_response = requests.get(cf7_url, timeout=5, headers={"User-Agent": "MusharpChatbotBot/1.0"})
                if cf7_response.status_code == 200:
                    cf7_data = cf7_response.json()
                    contact_info = f"CONTACT FORMS: {cf7_data}"
                    all_texts.append(contact_info)
            except:
                pass

            # 7. Try to fetch theme options or customizer data
            try:
                customizer_url = f"{self.site_url}/wp-json/wp/v2/theme"
                customizer_response = requests.get(customizer_url, timeout=5, headers={"User-Agent": "MusharpChatbotBot/1.0"})
                if customizer_response.status_code == 200:
                    theme_data = customizer_response.json()
                    theme_info = f"THEME INFO: {theme_data}"
                    all_texts.append(theme_info)
            except:
                pass

            full_text = "\n\n".join(all_texts)
            self.knowledge_chunks = self._chunk_text(full_text, chunk_size, overlap)
            print(f"✅ Website refreshed: {len(self.knowledge_chunks)} chunks loaded from all content types.")

        except Exception as e:
            print(f"❌ Failed to load WordPress site: {e}")

    def _auto_refresh(self):
        """Auto-refresh site content every 12 hours"""
        while True:
            time.sleep(60 * 60 * 12)  # 12 hours
            print("🔄 Auto-refreshing website content...")
            self.load_wordpress_site()

    def _find_relevant_chunks(self, query, max_chunks=5):
        """Enhanced keyword-based search in chunks"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []
        
        # Contact-related keywords for better matching
        contact_keywords = ['contact', 'email', 'phone', 'address', 'location', 'office', 'reach', 'get in touch']
        is_contact_query = any(keyword in query_lower for keyword in contact_keywords)
        
        for chunk in self.knowledge_chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            
            # Calculate multiple scoring factors
            word_matches = len(query_words.intersection(chunk_words))
            exact_phrase_score = 0
            if query_lower in chunk_lower:
                exact_phrase_score = 10
            
            # Boost score for important content types
            content_type_boost = 0
            if chunk.startswith("BLOG POST:"):
                content_type_boost = 3
            elif chunk.startswith("PAGE:"):
                content_type_boost = 2
            elif chunk.startswith("MENU:"):
                content_type_boost = 1
            
            # Special boost for contact-related queries
            contact_boost = 0
            if is_contact_query:
                if any(keyword in chunk_lower for keyword in contact_keywords):
                    contact_boost = 5
                if chunk.startswith("MENU:") and any(keyword in chunk_lower for keyword in ['contact', 'about', 'reach']):
                    contact_boost = 8
            
            # Calculate final score
            total_score = word_matches + exact_phrase_score + content_type_boost + contact_boost
            scored.append((total_score, chunk))
        
        # Sort by score and return top chunks
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in scored[:max_chunks] if score > 0]

    def _max_relevance_score(self, query):
        """Compute a rough max relevance score across chunks for gating."""
        if not self.knowledge_chunks:
            return 0
        query_lower = query.lower()
        query_words = set(query_lower.split())
        contact_keywords = ['contact', 'email', 'phone', 'address', 'location', 'office', 'reach', 'get in touch']
        is_contact_query = any(keyword in query_lower for keyword in contact_keywords)
        best = 0
        for chunk in self.knowledge_chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            word_matches = len(query_words.intersection(chunk_words))
            exact_phrase_score = 10 if query_lower in chunk_lower else 0
            content_type_boost = 3 if chunk.startswith("BLOG POST:") else (2 if chunk.startswith("PAGE:") else (1 if chunk.startswith("MENU:") else 0))
            contact_boost = 0
            if is_contact_query:
                if any(keyword in chunk_lower for keyword in contact_keywords):
                    contact_boost = 5
                if chunk.startswith("MENU:") and any(keyword in chunk_lower for keyword in ['contact', 'about', 'reach']):
                    contact_boost = 8
            score = word_matches + exact_phrase_score + content_type_boost + contact_boost
            if score > best:
                best = score
        return best

    def generate_response(self, user_input):
        """Generate chatbot reply with small-talk, site context, and general fallback."""
        user_lower = user_input.lower().strip()

        # Small-talk
        if re.search(r"\b(hi|hello|hey)\b", user_lower):
            return "Hi! How can I help you?"
        if "how are you" in user_lower:
            return "I'm doing great and ready to help. What can I do for you today?"

        # Gather context and relevance
        relevant_chunks = self._find_relevant_chunks(user_input) if self.knowledge_chunks else []
        context = "\n".join(relevant_chunks) if relevant_chunks else ""
        relevance = self._max_relevance_score(user_input)

        # Contact fallback
        if any(k in user_lower for k in ['contact', 'email', 'phone', 'address', 'location', 'office', 'reach']):
            if not context or not any(k in context.lower() for k in ['contact', 'email', 'phone', 'address', 'location', 'office']):
                context += "\n\nFALLBACK CONTACT INFO: Please check the website's Contact page for the latest contact details."

        try:
            if not context.strip() or relevance <= 0:
                # Out-of-site or no match → general helpful answer
                general_prompt = f"""
                You are a helpful assistant. Answer the user's question concisely and accurately.

                User Question: {user_input}
                """
                response = self.model.generate_content(general_prompt)
            else:
                site_prompt = f"""
        You are a helpful assistant for muSharp's website. Answer the user's question using ONLY the information provided in the context below.

        Context from website:
        {context}

        User Question: {user_input}

        Instructions:
        - Answer ONLY what is asked, nothing more
        - Use ONLY information from the context above
        - If the information is not in the context, say "I don't have that information in my knowledge base"
        - Be direct and concise
        - Don't add extra information or suggestions unless specifically asked
        """
                response = self.model.generate_content(site_prompt)

            response_text = getattr(response, "text", None)
            if not response_text:
                return "I couldn't generate a response right now. Please try again."
            return textwrap.fill(response_text.strip(), width=80)
        except Exception as e:
            return f"⚠️ Error: {e}"


# ------------------- FastAPI Setup -------------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Configure allowed frontend origins from env or use sensible defaults
_origins_env = os.getenv("FRONTEND_ORIGINS", "").strip()
if _origins_env:
    ALLOW_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    ALLOW_ORIGINS = [
        "https://musharp.com",
        "https://www.musharp.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load sensitive values from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SITE_URL = os.getenv("SITE_URL", "https://musharp.com")

responder = GeminiResponder(
    api_key=GEMINI_API_KEY,
    site_url=SITE_URL
)

class Query(BaseModel):
    message: str | None = None
    query: str | None = None

@app.post("/chat")
def chat(query: Query):
    user_message = query.message or query.query or ""
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")
    answer = responder.generate_response(user_message)
    return {"answer": answer}


@app.get("/chat")
def chat_get(message: str = "", query: str = ""):
    user_message = message or query
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")
    answer = responder.generate_response(user_message)
    return {"answer": answer}

@app.get("/")
def root(message: str = "", query: str = ""):
    # If message is provided as a query param, answer directly (GET fallback)
    user_message = message or query
    if user_message:
        answer = responder.generate_response(user_message)
        return {"answer": answer}
    return {"message": "Chatbot API is running 🚀"}

@app.get("/favicon.ico")
def favicon():
    # Optional: prevents noisy 404s in logs
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/content")
def debug_content():
    """Debug endpoint to see what content is loaded"""
    if not responder.knowledge_chunks:
        return {"message": "No content loaded", "chunks": 0}
    
    # Show first few chunks as sample
    sample_chunks = responder.knowledge_chunks[:3]
    return {
        "total_chunks": len(responder.knowledge_chunks),
        "sample_chunks": sample_chunks,
        "chunk_previews": [chunk[:200] + "..." if len(chunk) > 200 else chunk for chunk in sample_chunks]
    }


@app.get("/debug/search")
def debug_search(query: str):
    """Debug endpoint to see what chunks match a search query"""
    if not responder.knowledge_chunks:
        return {"message": "No content loaded", "chunks": 0}
    
    relevant_chunks = responder._find_relevant_chunks(query, max_chunks=10)
    return {
        "query": query,
        "total_chunks": len(responder.knowledge_chunks),
        "relevant_chunks": relevant_chunks,
        "chunk_previews": [chunk[:300] + "..." if len(chunk) > 300 else chunk for chunk in relevant_chunks]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
