from dotenv import load_dotenv
import openai

load_dotenv()

embedding = openai.embeddings.create(input="Ritwik works at Global Logic", model="text-embedding-ada-002")

print(embedding)
