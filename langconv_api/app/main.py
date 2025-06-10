from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from app.models.translation import translate
from fastapi.responses import JSONResponse
from app.schemas.input_validation import InputValidationSchema
from app.schemas.predection_response import TranslationResponse



# fast api Instance
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI Language Translator!"}


# converter route
@app.post("/convert/{convert_to}", response_model=TranslationResponse)
def convert(convert_to:str,input_text: InputValidationSchema):
    """
    Converts a sentence to the specified language.
    Args:
        convert_to (str): The target language to convert to ('nepali' or 'german').
    """
    if convert_to.lower() not in ['nepali', 'german']:
        return JSONResponse(
            status_code=400,
            content={
                "message": "Invalid language specified. Please use 'nepali' or 'german'."
            },
        )
    
    input_text = input_text.text  # Extract the text from the validated input schema
    
    print(f"Received input text: {input_text}")
    print(f"Converting to: {convert_to}")
    try:
    
        # calling the translate function from translation module
        result = translate(sentence=input_text, convert_to=convert_to)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Translation successful",
                "english_text": input_text,
                "translated_text": result,
                "language": convert_to
            },
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred during translation",
                "error": str(e)
            },
        )



