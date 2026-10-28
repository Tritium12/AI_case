import os
import json
import asyncio
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Импорты для краулера (crawl4ai)
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = "./documents"  # Папка с файлами
URLS_FILE = os.path.join(CURRENT_DIR, "urls.txt") # Файл со ссылками
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"
EMBEDDING_MODEL = "bge-m3"

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def load_json_file(file_path):
    """
    Зачем нужна: Читает JSON файлы (базы данных, выгрузки API) и превращает их в текстовые документы.
    Логика: Если это список объектов — создает отдельный документ для каждого. Если словарь — один общий.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, ensure_ascii=False, indent=2)
                docs.append(Document(page_content=content, metadata={"source": file_path, "item_index": i}))
        else:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            docs.append(Document(page_content=content, metadata={"source": file_path, "item_index": 0}))
        return docs
    except Exception as e:
        print(f"  [Ошибка JSON] Не удалось прочитать {file_path}: {e}")
        return []

async def crawl_urls_with_depth(urls, max_depth=1):
    """
    Зачем нужна: Скачивает сайты, вытаскивает чистый текст и убирает меню/рекламу.
    Логика: Запускает BFS-обход для КАЖДОГО URL отдельно, чтобы избежать багов
    с несколькими стартовыми точками в crawl4ai.
    """
    all_docs = []
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    content_filter = PruningContentFilter(threshold=0.48, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
    
    # Конфигурация запуска (без deep_crawl_strategy внутри!)
    run_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        page_timeout=15000
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            print(f"  -> 🕸️ Сканируем сайт: {url} (глубина {max_depth})...")
            try:
                # Создаём стратегию для КАЖДОГО url отдельно
                strategy = BFSDeepCrawlStrategy(
                    max_depth=max_depth,
                    include_external=False
                )
                
                # Передаём стратегию напрямую в arun, а не через конфиг
                result = await asyncio.wait_for(
                    crawler.arun(
                        url=url, 
                        config=run_config, 
                        deep_crawl_strategy=strategy  # <--- Ключевое исправление
                    ),
                    timeout=120.0 
                )
                
                # Обработка результатов
                if isinstance(result, list):
                    for page in result:
                        if page.success and page.markdown.fit_markdown:
                            all_docs.append(Document(
                                page_content=page.markdown.fit_markdown,
                                metadata={"source": page.url, "page": 1}
                            ))
                            print(f"     ✅ Прочитано: {page.url}")
                elif result.success and result.markdown.fit_markdown:
                    all_docs.append(Document(
                        page_content=result.markdown.fit_markdown,
                        metadata={"source": url, "page": 1}
                    ))
                    print(f"     ✅ Прочитано: {url}")
                else:
                    print(f"     ❌ Ошибка: {getattr(result, 'error_message', 'Нет данных')}")
                    
            except asyncio.TimeoutError:
                print(f"     ⏰ Таймаут: Сайт {url} не ответил за 120 секунд. Пропускаем.")
            except Exception as e:
                print(f"     ❌ Ошибка при обходе {url}: {e}")
                
    return all_docs

# ==========================================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================================

async def main():
    all_docs = []

    # ==========================================
    # 2. ЧТЕНИЕ ИЗ ПАПКИ (PDF/TXT/JSON)
    # ==========================================
    if os.path.exists(DOCS_DIR):
        print(f"⏳ Сканируем папку: {DOCS_DIR}...")
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if not os.path.isfile(file_path): continue
            
            print(f"  -> Читаем файл: {filename}")
            try:
                if filename.lower().endswith(".pdf"):
                    all_docs.extend(PyMuPDFLoader(file_path).load())
                elif filename.lower().endswith(".txt"):
                    all_docs.extend(TextLoader(file_path, encoding="utf-8").load())
                elif filename.lower().endswith(".json"):
                    all_docs.extend(load_json_file(file_path))
                else:
                    print(f"  [Пропуск] Неподдерживаемый формат: {filename}")
            except Exception as e:
                print(f"  [Ошибка] {filename}: {e}")
    else:
         print(f"⚠️ Папка {DOCS_DIR} не найдена (пропускаем файлы).")

    # ==========================================
    # 3. ЧТЕНИЕ ИЗ URL (ссылки)
    # ==========================================
    if os.path.exists(URLS_FILE):
        print(f"⏳ Читаем ссылки из файла: {URLS_FILE}...")
        with open(URLS_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
            
        if urls:
            url_docs = await crawl_urls_with_depth(urls, max_depth=1)
            all_docs.extend(url_docs)
    else:
        print(f"ℹ️ Файл {URLS_FILE} не найден (пропускаем ссылки).")

    if not all_docs:
        print("❌ Не найдено ни одного источника (ни файлов, ни ссылок).")
        return

    print(f"✅ Всего загружено документов/страниц: {len(all_docs)}")

    # ==========================================
    # 4. ЧАНКИНГ
    # ==========================================
    print("⏳ Режем на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)
    print(f"✅ Получено чанков: {len(chunks)}")

    # ==========================================
    # 5. СОХРАНЕНИЕ В CHROMADB (БАТЧАМИ)
    # ==========================================
    print(f"⏳ Векторизуем и сохраняем в ChromaDB (по 100 чанков)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Оптимизация: сохраняем по 100 чанков + пауза, чтобы Ollama не падала
    import time 
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  -> Сохраняем чанки {i+1} - {min(i+batch_size, len(chunks))} из {len(chunks)}...")
        vectorstore.add_documents(batch)
        time.sleep(1) # Пауза 1 секунда, чтобы Ollama "перевела дыхание"

    print(f"🎉 Готово! База обновлена.")

if __name__ == "__main__":
    asyncio.run(main())
