import pydantic as pydantic
from pydantic import BaseModel, Field, validator


class InputValidationSchema(BaseModel):
    """
    Schema for validating input data.
    """
    text: str = Field(
        ...,
        description="The text to be validated."
    )
    