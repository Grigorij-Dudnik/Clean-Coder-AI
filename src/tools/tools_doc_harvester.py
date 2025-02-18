"""Tolls for doc harvester."""
from pydantic import BaseModel, Field


class PythonLibraries(BaseModel):
    """Identify python libraries which are relevant to the user's task. No fewer than three."""

    libraries: list[str] = Field("The list of libraries.")



if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(PythonLibraries)
    structured_llm.invoke("I want to create a website scraper for abcnews.com. I want to use playwright for the scraping")
