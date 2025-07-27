import json
import re
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.client import _format_feedback_score
from app.config import settings

translation_response_schema = [
    ResponseSchema(
        name="translated_text",
        description="The translated text in Bengali if the input is not in Bengali, or the original text if in Bengali."
    )
]
translation_output_parser = StructuredOutputParser.from_response_schemas(translation_response_schema)
format_instructions = translation_output_parser.get_format_instructions()

prompt_template = PromptTemplate(
        template="""
        You are an AI assistant tasked with processing a question for translation. Follow these instructions strictly:
            1. If the text is not in Bengali, translate it to Bengali.
            2. If the text is bengali then translate it into english.
            3. Do not add extra words or information beyond the translated or original text.
            4. Provide the output in the structured format specified below.
            5. Do not answer the question, only provide the translated or original text.

            Examples:
            Input: This is an English question.
            translated_text: "এটি একটি ইংরেজি প্রশ্ন।"

            Input: এই প্রশ্নটি বাংলায়।
            translated_text: "This question is in Bengali."
            
            User Query: {text}
            {format_instructions}
            """,

            input_variables=["text"],
            partial_variables={"format_instructions": format_instructions}        
    )


def process_prompt(prompt: str) -> dict:
    try:
        llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL_NAME, google_api_key=settings.GEMINI_API_KEY)
        chain = prompt_template | llm | translation_output_parser
        
        result = chain.invoke({"text": prompt})
        
        return {
            'response': result.get('translated_text', prompt)
        }
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return {
            'response': prompt,
        }

# Example run
if __name__ == '__main__':
    x = process_prompt("এটি একটি ইংরেজি প্রশ্ন।")
    print(x['response'])