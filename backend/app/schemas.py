from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    source: str = Field(
        ...,
        description="The exact filename of the document containing the cited fact (e.g. 'paper.pdf')."
    )
    page: int = Field(
        ...,
        description="The 1-based page number where the cited text snippet is located."
    )
    snippet: str = Field(
        ...,
        description="The exact sentence or passage from the context that supports the answer's claim."
    )

class CitationResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The direct answer to the user's query. Use markdown for styling and format numbers in brackets (e.g. [1], [2]) to represent references corresponding to the list of citations."
    )
    citations: List[Citation] = Field(
        description="The collection of unique citations backing the statements in the answer."
    )

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
