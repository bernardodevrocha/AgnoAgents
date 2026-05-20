from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent = Agent(model=Groq(id="llama-3.1-8b-instant"))

resp = agent.run("Olá, meu nome é Bernardo")
print(resp.content)
