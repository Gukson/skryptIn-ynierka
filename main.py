import os
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
import re
import json
from pydantic import BaseModel, Field


if __name__ == '__main__':
    print("Hello, world!")