from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

text = ""

with open("./script_output.txt", "r") as file:
    text += file.read()

with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="echo",
        input=text,
    ) as response:

    response.stream_to_file("./output_speak.mp3")
