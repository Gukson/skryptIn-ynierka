import os
import json
import re
from typing import List

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") #load api key from .env file

class Person(BaseModel):
    persona: str = Field(description="type of the character")
    description: str = Field(description="description of the character")

def load_api_key() -> str:
    load_dotenv()
    return os.getenv("DEEPSEEK_API_KEY")

def load_prompts(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def init_llm(api_key: str) -> ChatDeepSeek:
    return ChatDeepSeek(model="deepseek-chat", api_key=api_key)

def extract_json_list(text: str) -> str:
    """Extracts the first JSON array from the response text."""
    match = re.search(r'\[[\s\S]*?\]', text)
    return match.group(0) if match else "[]"

def parse_personas(response_text: str) -> List[dict]:
    """Parses response text into a list of persona dictionaries."""
    parser = JsonOutputParser(pydantic_object=Person)
    json_array = extract_json_list(response_text)
    lines = json_array.strip().split("\n")[1:-1]

    def enrich_with_id(i, line):
        item = parser.invoke(line)
        item["id"] = i
        return item

    return [enrich_with_id(i, line) for i, line in enumerate(lines)]


def generate_personas(prompt: str, llm: ChatDeepSeek) -> List[dict]:
    response = llm.invoke(prompt)
    return parse_personas(response.content)


def save_to_file(data: List[dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    api_key = load_api_key()
    prompts = load_prompts("prompts.json")
    llm = init_llm(api_key)
    prompt_text = prompts["initial_prompt"]

    """Generate personas using the LLM and save them to a file."""
    personas = generate_personas(prompt_text, llm)
    save_to_file(personas, "data/people.json")

if __name__ == '__main__':
    main()