from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "agnoAgent.sqlite"
SESSION_ID = "cotacoes_session"
USER_ID = "usuario_cotacoes"

db = SqliteDb(db_file=str(DB_PATH))

agent = Agent(
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[YFinanceTools()],
    instructions=["Você é meu assistente que responde qualquer pergunta!"],
    db=db,
    debug_mode=True,
    add_history_to_context=True,
    num_history_runs=3,
)

# agent.print_response("Qual a cotação da Petrobras?", session_id=SESSION_ID, user_id=USER_ID)
# agent.print_response("Qual é a cotação da Vale?", session_id=SESSION_ID, user_id=USER_ID)
# agent.print_response("Quais empresas já consultamos a cotação?", session_id=SESSION_ID, user_id=USER_ID)

agent.print_response("Qual é a cotação da Vale?", session_id="vale_session_2", user_id="analista_vale")

