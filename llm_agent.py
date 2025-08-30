import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Dict

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# GitHub AI configuration
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-5"
TOKEN = os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set. Please set it before running this script.")

# Initialize GitHub AI client
client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

# Create ACP server
server = Server()

# Store conversation history for each context
conversation_history: Dict[str, list] = {}


@server.agent()
async def llm_assistant(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    An LLM-powered assistant that uses GitHub's AI model (OpenAI GPT-5) to generate responses.
    
    This agent will:
    1. Process incoming messages
    2. Send them to the GitHub AI model
    3. Return the AI's response
    """
    # Generate a unique conversation ID for this context
    conversation_id = str(context.run_id)
    
    # Initialize conversation history for this context if it doesn't exist
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    # Process each incoming message
    for message in input:
        # Extract message content
        message_content = ""
        for part in message.parts:
            if part.content_type == "text/plain":
                message_content += part.content
        
        # Skip empty messages
        if not message_content:
            continue
        
        # Add user message to conversation history
        conversation_history[conversation_id].append(UserMessage(message_content))
        
        # Yield a thought to show processing
        yield {"thought": "Processing request using GitHub AI model..."}
        await asyncio.sleep(0.5)
        
        try:
            # Create messages list with system prompt and conversation history
            messages = [
                SystemMessage("You are a helpful assistant powered by GitHub AI."),
                *conversation_history[conversation_id]
            ]
            
            # Call the GitHub AI model
            response = client.complete(
                messages=messages,
                model=MODEL
            )
            
            # Extract the response content
            ai_response = response.choices[0].message.content
            
            # Add assistant response to conversation history
            conversation_history[conversation_id].append(SystemMessage(ai_response))
            
            # Create and yield the response message
            response_message = Message(
                role="assistant",
                parts=[MessagePart(content=ai_response, content_type="text/plain")]
            )
            
            yield response_message
            
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error calling GitHub AI model: {str(e)}"
            
            # Create and yield error message
            error_response = Message(
                role="assistant",
                parts=[MessagePart(content=error_message, content_type="text/plain")]
            )
            
            yield error_response


# TODO: Enhance this agent to be a specialized knowledge assistant
# 1. Update the system prompt to focus on a specific knowledge domain
# 2. Improve conversation history management
# 3. Add better error handling


# Run the server
server.run()
