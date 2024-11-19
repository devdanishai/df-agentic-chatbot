import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from io import BytesIO  # Import BytesIO from io module

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variable to store the DataFrame
global_df = None
pandas_df_agent = None


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global global_df, pandas_df_agent
    filename = file.filename
    content = await file.read()

    if filename.endswith('.csv'):
        global_df = pd.read_csv(BytesIO(content))  # Use BytesIO here
    elif filename.endswith(('.xlsx', '.xls')):
        global_df = pd.read_excel(BytesIO(content))  # Use BytesIO here
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Create a Langchain agent using the OpenAI API key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        global_df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  # Opt-in to allow execution of arbitrary code
    )

    return {"message": "File connect with LLM successfully"}


@app.post("/ask")
async def ask_question(question: Question):
    global pandas_df_agent
    if pandas_df_agent is None:
        raise HTTPException(status_code=400, detail="Please upload a file first")

    response = pandas_df_agent.run(question.question)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)