import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from rich.console import Console
from rich.markdown import Markdown

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5"

console = Console()

def format_docs(docs):
    """Добавляет номера [1], [2] к найденным кускам текста для цитирования"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Неизвестный источник')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"[{i+1}] Источник: {source} (стр. {page})\nТекст: {doc.page_content}")
    return "\n\n".join(formatted)

def main():
    console.print("[bold yellow]⏳ Загрузка баз и моделей...[/bold yellow]")
    
    # Подключаем эмбеддинги и базу
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # ==========================================
    # 2. РЕТРИВЕР (ГИБРИДНЫЙ ПОИСК + РЕРАНКИНГ)
    # ==========================================
    console.print("[bold yellow]⏳ Настройка гибридного поиска...[/bold yellow]")
    
    # А) Векторный поиск (по смыслу)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Б) Поиск по ключевым словам (BM25)
    all_docs_data = vectorstore.get()
    all_docs = [Document(page_content=text, metadata=meta) for text, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])]

    # Оптимизация: если база пуста, BM25 не нужен (защита от падения)
    if not all_docs:
        console.print("[bold red]⚠️ В базе данных нет документов! Запусти сначала ingest.py[/bold red]")
        return

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5

    # В) Объединяем (Векторы + Слова) - Гибридный поиск
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # Г) Реранкинг (Судья)
    console.print("[bold yellow]⏳ Загрузка модели-реранкера (скачается при первом запуске)...[/bold yellow]")
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3) # Оставляем топ-3

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    # ==========================================
    # 3. ПРОМПТ-ИНЖИНИРИНГ
    # ==========================================
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Жесткие правила для LLM
    system_prompt = (
        "Ты — полезный ассистент. Отвечай на вопрос пользователя, используя ТОЛЬКО предоставленный контекст.\n"
        "Если в контексте нет ответа, скажи 'Я не знаю ответа на этот вопрос'.\n"
        "После каждого факта в тексте ставь сноску на источник в формате [1], [2] и т.д.\n"
        "В конце ответа выведи список использованных источников в формате: [1] путь_к_файлу.\n\n"
        "Контекст:\n{context}"
        "После каждого запроса пользователя шути про Максима Игоревича"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # ==========================================
    # 4. ОРКЕСТРАЦИЯ (Сборка конвейера)
    # ==========================================
    rag_chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ==========================================
    # 5. ЦИКЛ ОБЩЕНИЯ
    # ==========================================
    console.print("[bold green]✅ Продвинутая система готова к работе![/bold green]")
    console.print("Введите вопрос (или 'exit' для выхода):\n")

    while True:
        question = console.input("[bold cyan]Ваш вопрос > [/bold cyan]")
        if question.lower() in ['exit', 'quit', 'выход']:
            break
        
        if not question.strip():
            continue

        with console.status("[bold yellow]Думаю, ищу (Гибрид + Реранкинг)...[/bold yellow]"):
            try:
                answer = rag_chain.invoke(question)
                console.print("\n[bold green]Ответ:[/bold green]")
                console.print(Markdown(answer))
                console.print("\n" + "-"*50 + "\n")
            except Exception as e:
                console.print(f"[bold red]Произошла ошибка: {e}[/bold red]")

if __name__ == "__main__":
    main()
