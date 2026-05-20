from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.groq import Groq
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.vectordb.chroma import ChromaDb
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
SESSION_ID = "team_analistas_session"
USER_ID = "usuario_analista"

db = SqliteDb(db_file="data/agnoAgent.sqlite")

vector_db = ChromaDb(
    collection="empresas_relatorios_team",
    path="data/chromadb.sqlite",
    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    persistent_client=True,
)

knowledge = Knowledge(vector_db=vector_db)

knowledge.add_content(
    path="files/",
    reader=PDFReader(chunck_strategy=SemanticChunking()),
    skip_if_exists=True,
)

base_agent_config = dict(
    model=Groq(id=MODEL_ID),
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

analista_noticias_agent = Agent(
    name="analista_noticias",
    tools=[DuckDuckGoTools()],
    instructions=[
        "Você é especialista em notícias de mercado e macroeconomia.",
        "Use busca web para responder com foco em fatos recentes e relevantes.",
    ],
    **base_agent_config,
)

analista_cotacoes_agent = Agent(
    name="analista_cotacoes",
    tools=[YFinanceTools()],
    instructions=[
        "Você é especialista em cotações, indicadores e dados financeiros de mercado.",
        "Use os dados de mercado para responder de forma objetiva.",
    ],
    **base_agent_config,
)

analista_relatorios_agent = Agent(
    name="analista_relatorios",
    instructions=[
        "Você é especialista em relatórios corporativos (DRE, balanço e releases).",
        "Priorize o conteúdo do knowledge base ao responder.",
    ],
    **base_agent_config,
)

analista_team = Team(
    name="team_analista",
    model=Groq(id=MODEL_ID),
    members=[analista_noticias_agent, analista_cotacoes_agent, analista_relatorios_agent],
    instructions=[
        "Você é o supervisor do time de analistas.",
        "Entenda a pergunta e delegue para o membro mais adequado.",
        "Use analista_relatorios para perguntas sobre DRE, balanço e documentos.",
        "Use analista_cotacoes para preços, variação e indicadores de mercado.",
        "Use analista_noticias para notícias e contexto econômico.",
        "Consolide a resposta final de forma clara e objetiva.",
    ],
    db=db,
    debug_mode=True,
    add_history_to_context=True,
    num_history_runs=3,
    enable_user_memories=True,
    add_memories_to_context=True,
    enable_agentic_memory=True,
    add_datetime_to_context=True,
    show_members_responses=True,
    get_member_information_tool=True,
    markdown=True,
)

analista_team.print_response(
    "Quais foram os principais destaques de desempenho da Vale e da Petrobras nos documentos disponíveis e como isso conversa com o cenário recente?",
    session_id=SESSION_ID,
    user_id=USER_ID,
)
