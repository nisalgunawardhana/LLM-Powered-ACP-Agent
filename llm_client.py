import asyncio
import json
import uuid
from datetime import datetime

from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart


# Custom JSON encoder to handle UUID and datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            # Convert UUID to string
            return str(obj)
        elif isinstance(obj, datetime):
            # Convert datetime to ISO format string
            return obj.isoformat()
        # Let the base class handle other types or raise TypeError
        return super().default(obj)


async def example() -> None:
    """
    Example client that interacts with the LLM-powered agent.
    """
    async with Client(base_url="http://localhost:8000") as client:
        # Ask a question to the LLM agent
        print("Sending first question to the LLM agent...")
        first_run = await client.run_sync(
            agent="llm_assistant",
            input=[
                Message(
                    parts=[MessagePart(content="What are the key concepts of the Agent Communication Protocol?", content_type="text/plain")]
                )
            ],
        )
        
        # Print debug information
        print(f"\nResponse status: {first_run.status}")
        print(f"Run ID: {first_run.run_id}")
        print(f"Output messages count: {len(first_run.output)}")
        
        # Print the full response for debugging
        print("\nFull response object:")
        try:
            # Use model_dump() instead of dict() (Pydantic v2 method) and the custom encoder
            response_dict = first_run.model_dump() if hasattr(first_run, 'model_dump') else first_run.dict()
            print(json.dumps(response_dict, indent=2, cls=CustomJSONEncoder))
        except Exception as e:
            print(f"Error serializing response: {str(e)}")
            print(f"Response type: {type(first_run)}")
        
        # Print the response content
        print("\nResponse from LLM agent:")
        response_found = False
        for message in first_run.output:
            print(f"Message role: {message.role}")
            # Check if role contains 'agent' (either exact match or contains it)
            if message.role == "agent" or "agent" in message.role:
                for part in message.parts:
                    print(f"Part content type: {part.content_type}")
                    if part.content_type == "text/plain":
                        print(f"\n{part.content}")
                        response_found = True
        
        if not response_found:
            print("No response content found. This could be due to rate limiting or an error.")
            if hasattr(first_run, 'error') and first_run.error:
                print(f"Error: {first_run.error}")
        
        # For the follow-up question, let's not try to maintain session
        # Instead, we'll ask a standalone question that references the previous one
        print("\nSending follow-up question...")
        follow_up_run = await client.run_sync(
            agent="llm_assistant",
            input=[
                Message(
                    parts=[MessagePart(content="Based on your previous explanation of ACP, can you give a specific example of how it's implemented in code?", content_type="text/plain")]
                )
            ],
        )
        
        # Print debug information for follow-up
        print(f"\nFollow-up response status: {follow_up_run.status}")
        print(f"Follow-up output messages count: {len(follow_up_run.output)}")
        
        # Print the follow-up response
        print("\nFollow-up response:")
        response_found = False
        for message in follow_up_run.output:
            print(f"Message role: {message.role}")
            # Check if role contains 'agent' (either exact match or contains it)
            if message.role == "agent" or "agent" in message.role:
                for part in message.parts:
                    print(f"Part content type: {part.content_type}")
                    if part.content_type == "text/plain":
                        print(f"\n{part.content}")
                        response_found = True
        
        if not response_found:
            print("No follow-up response content found.")
            if hasattr(follow_up_run, 'error') and follow_up_run.error:
                print(f"Error: {follow_up_run.error}")


if __name__ == "__main__":
    asyncio.run(example())
