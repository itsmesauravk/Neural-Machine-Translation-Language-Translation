from pydantic import BaseModel, Field
from typing import Dict


class TranslationResponse(BaseModel):
    """
    Schema for the Translation response.
    """
    message: str = Field(
        ...,
        description="A message indicating the success of the translation operation."
    ),
    translated_text: str = Field(
        ...,
        description="The sentence translated from English to the target language."
    )
    english_text: str = Field(
        ...,
        description="The original English sentence that was translated."
    )
    language: str = Field(
        ...,
        description="The target language of the translation (e.g., 'nepali' or 'german')."
    )