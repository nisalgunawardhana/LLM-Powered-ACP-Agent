import asyncio
import os
import json
import time
from collections.abc import AsyncGenerator
from typing import Dict, Any, List, Tuple
from datetime import datetime

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
MODEL = "openai/gpt-4o"
TOKEN = os.environ.get("GITHUB_TOKEN")
# Enable mock responses when rate limited (for educational purposes)
USE_MOCK_WHEN_RATE_LIMITED = True

if not TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set. Please set it before running this script.")

# Initialize GitHub AI client
client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

# Create ACP server
server = Server()

# Maximum number of messages to keep in conversation history
MAX_HISTORY_LENGTH = 10

# Store conversation history for each context
# Structure: Dict[conversation_id, List[Tuple[timestamp, message, is_user]]]
conversation_history: Dict[str, List[Tuple[float, SystemMessage | UserMessage, bool]]] = {}

def add_to_conversation_history(conversation_id: str, message: SystemMessage | UserMessage, is_user: bool = False):
    """
    Add a message to the conversation history with timestamp and manage history length
    
    Args:
        conversation_id: Unique identifier for the conversation
        message: The message to add
        is_user: Whether this is a user message (True) or assistant message (False)
    """
    # Initialize history if it doesn't exist
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    
    # Add message with current timestamp
    timestamp = time.time()
    conversation_history[conversation_id].append((timestamp, message, is_user))
    
    # Limit history length to prevent token explosion
    if len(conversation_history[conversation_id]) > MAX_HISTORY_LENGTH:
        # Keep system messages and recent messages
        system_messages = [(ts, msg, is_u) for ts, msg, is_u in conversation_history[conversation_id] 
                          if isinstance(msg, SystemMessage) and not is_u]
        
        # Get recent messages, excluding system messages
        regular_messages = [(ts, msg, is_u) for ts, msg, is_u in conversation_history[conversation_id] 
                           if not (isinstance(msg, SystemMessage) and not is_u)]
        
        # Sort by timestamp (newest first)
        regular_messages.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only the most recent messages
        regular_messages = regular_messages[:MAX_HISTORY_LENGTH-len(system_messages)]
        
        # Combine and sort by timestamp (oldest first)
        conversation_history[conversation_id] = system_messages + regular_messages
        conversation_history[conversation_id].sort(key=lambda x: x[0])

def get_conversation_messages(conversation_id: str) -> List[SystemMessage | UserMessage]:
    """
    Get all messages from the conversation history, properly ordered
    
    Args:
        conversation_id: Unique identifier for the conversation
        
    Returns:
        List of messages for the model context
    """
    if conversation_id not in conversation_history:
        return []
    
    # Extract just the messages, not timestamps or user flags
    return [msg for _, msg, _ in conversation_history[conversation_id]]


@server.agent()
async def llm_assistant(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    An LLM-powered assistant that uses GitHub's AI model (OpenAI GPT-4o) to generate responses.
    
    This agent will:
    1. Process incoming messages
    2. Send them to the GitHub AI model
    3. Return the AI's response
    """
    # Inspect context object to find available attributes
    context_dict = {}
    for attr in dir(context):
        if not attr.startswith('_') and not callable(getattr(context, attr)):
            try:
                value = getattr(context, attr)
                if not callable(value):
                    context_dict[attr] = str(value)
            except Exception as e:
                context_dict[attr] = f"Error accessing: {str(e)}"
    
    # Debug information as a thought
    yield {"thought": f"Context object attributes: {json.dumps(context_dict)}"}
    
    # Generate a unique conversation ID using session_id from context variables
    # This is more reliable than trying to access attributes directly
    variables = getattr(context, 'variables', {})
    session_id = variables.get('session_id', None) if variables else None
    
    # Fallback to other identifiers if session_id is not available
    if not session_id:
        # Try different possible attributes for identification
        for id_attr in ['session_id', 'run_id', 'id']:
            if hasattr(context, id_attr):
                session_id = getattr(context, id_attr)
                break
        
        # Last resort: use the object's memory address
        if not session_id:
            session_id = id(context)
    
    conversation_id = str(session_id)
    
    # Debug information
    yield {"thought": f"Using conversation ID: {conversation_id}"}
    
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
        
        # Add user message to conversation history with timestamp
        user_message = UserMessage(message_content)
        add_to_conversation_history(conversation_id, user_message, is_user=True)
        
        # Get current timestamp for logging
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Yield a thought to show processing
        yield {"thought": f"[{current_time}] Processing request using GitHub AI model..."}
        await asyncio.sleep(0.5)
        
        try:
            # Create messages list with system prompt and conversation history
            messages = [
                SystemMessage("""You are an AI and Machine Learning specialist powered by GitHub AI.
You have expertise in machine learning algorithms, neural networks, deep learning frameworks, 
and AI development best practices. Focus on providing clear, accurate explanations of AI concepts,
implementation guidance, and practical advice.

When sharing code, include detailed comments explaining key components.
Offer insights into common pitfalls and optimization techniques.
Keep explanations accessible but technically precise."""),
            ]
            
            # Add conversation history messages
            messages.extend(get_conversation_messages(conversation_id))
            
            # Log the number of messages being sent to the API
            yield {"thought": f"Sending {len(messages)} messages to the GitHub AI API..."}
            
            # Check if we have a valid token before making API call
            if not TOKEN or TOKEN.strip() == "":
                raise ValueError("GitHub token is missing or empty")
                
            print(f"Making API call to GitHub AI model {MODEL}...")
            
            try:
                # Call the GitHub AI model
                response = client.complete(
                    messages=messages,
                    model=MODEL
                )
                print("API call successful, processing response...")
            except Exception as e:
                error_str = str(e).lower()
                # If rate limited and mock responses are enabled, provide a mock response
                if USE_MOCK_WHEN_RATE_LIMITED and ("429" in error_str or "rate limit" in error_str):
                    print("Rate limit detected, using mock response for educational purposes...")
                    
                    # Create a mock response object with structure similar to the real response
                    class MockMessage:
                        def __init__(self, content):
                            self.content = content
                    
                    class MockChoice:
                        def __init__(self, message):
                            self.message = message
                    
                    class MockResponse:
                        def __init__(self, content):
                            self.choices = [MockChoice(MockMessage(content))]
                    
                    # Create educational mock response
                    mock_content = """[MOCK RESPONSE FOR EDUCATIONAL PURPOSES]

This is a simulated response since the GitHub AI API rate limit has been exceeded.
In a real application, you would need to implement proper rate limit handling.

Some best practices for handling API rate limits:
1. Implement exponential backoff and retry mechanisms
2. Cache responses when possible
3. Use a token bucket algorithm for client-side rate limiting
4. Monitor your usage and adjust request patterns
5. Consider implementing a queue system for high-volume applications

For this educational example, you can continue exploring the application, but responses
will be simulated until the rate limit resets.
"""
                    response = MockResponse(mock_content)
                else:
                    # Re-raise the exception if not a rate limit or mock is disabled
                    raise
            
            # Extract the response content
            ai_response = response.choices[0].message.content
            
            # Add assistant response to conversation history
            assistant_message = SystemMessage(ai_response)
            add_to_conversation_history(conversation_id, assistant_message)
            
            # Create and yield the response message
            response_message = Message(
                role="agent",  # Changed from 'assistant' to 'agent' to match ACP SDK requirements
                parts=[MessagePart(content=ai_response, content_type="text/plain")]
            )
            
            print(f"Sending response (length: {len(ai_response)})")
            yield response_message
            
        except Exception as e:
            # Handle errors gracefully with more specific error messages
            error_message = "I'm sorry, but I encountered an issue."
            
            # Check for specific error types
            error_str = str(e).lower()
            print(f"ERROR: {str(e)}")
            
            if "429" in error_str or "rate limit" in error_str:
                print("Rate limit error detected!")
                # Extract retry-after time if available
                retry_after = None
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    retry_after = e.response.headers.get('Retry-After')
                    print(f"Retry-After header: {retry_after}")
                
                if retry_after:
                    error_message = f"The GitHub AI API rate limit has been exceeded. Please try again in {retry_after} seconds."
                else:
                    error_message = "The GitHub AI API rate limit has been exceeded. Please try again later."
            elif "timeout" in error_str:
                error_message = "The request timed out. This might be due to high server load or network issues."
            elif "token" in error_str or "authentication" in error_str or "auth" in error_str:
                error_message = "There seems to be an authentication issue. Please check your GitHub token."
                print(f"Token error. Token length: {len(TOKEN) if TOKEN else 0}")
            else:
                error_message = f"Error calling GitHub AI model: {str(e)}"
            
            # Log the error for debugging
            print(f"Error encountered: {str(e)}")
            
            # Create and yield error message
            error_response = Message(
                role="agent",  # Changed from 'assistant' to 'agent' to match ACP SDK requirements
                parts=[MessagePart(content=error_message, content_type="text/plain")]
            )
            
            yield error_response


# TODO: Enhance this agent to be a specialized knowledge assistant
# 1. Update the system prompt to focus on a specific knowledge domain
# 2. Improve conversation history management
# 3. Add better error handling


# Run the server
server.run()
