from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.models.groq import Groq

from dotenv import load_dotenv

load_dotenv()

db = SqliteDb(db_file="data/agnoAgent.sqlite")

vector_db = ChromaDb(
    collection="empresas_relatorios",
    path="data/chromadb.sqlite",
    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    persistent_client=True
)

knowledge = Knowledge(
    vector_db=vector_db,
)

knowledge.add_content(
    path="files/PETR4/",
    reader=PDFReader(
        chunck_strategy=SemanticChunking()
    ),
    metadata={
        "company": "Petrobras",
        "sector": "Petróleo e Gás",
        "country": "Brazil"
    },
    skip_if_exists=True
)

knowledge.add_content(
    path="files/VALE/",
    reader=PDFReader(
        chunck_strategy=SemanticChunking()
    ),
    metadata={
        "company": "Vale",
        "sector": "Mineiração",
        "country": "Brazil"
    },
    skip_if_exists=True
)

agent = Agent(
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[YFinanceTools()],
    instructions=["Você é meu assistente que responde qualquer pergunta!"],
    db=db,
    debug_mode=True,
    add_history_to_context=True,
    num_history_runs=3,
    enable_user_memories=True,
    add_memories_to_context=True,
    enable_agentic_memory=True,
    knowledge=knowledge,
    add_knowledge_to_context=True,
)

agent.print_response("Olá, o que foi comentado sobre o Produção e Vendas da vale no 2T25?", session_id="vale_session", user_id="analista_vale")
agent.print_response("Olá, o que foi comentado sobre o Desempenho financeiro da Petrobras no 2T25?", session_id="petrobras_session", user_id="analista_petrobras")
