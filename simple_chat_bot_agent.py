from agno.models.groq import Groq
from agno.models.message import Message

from dotenv import load_dotenv
load_dotenv()

model = Groq(id="meta-llama/llama-4-scout-17b-16e-instruct")

msg = Message(
    role="user",
    content=[{"type": "text", "text": "Olá, quero que voce me cite quais sao os principais metodos e boas praticas de programacao"}]
)

response = model.response([msg])

print(response.content)