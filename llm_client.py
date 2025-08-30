import asyncio

from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart


async def example() -> None:
    """
    Example client that interacts with the LLM-powered agent.
    """
    async with Client(base_url="http://localhost:8000") as client:
        # Ask a question to the LLM agent
        run = await client.run_sync(
            agent="llm_assistant",
            input=[
                Message(
                    parts=[MessagePart(content="What are the key concepts of the Agent Communication Protocol?", content_type="text/plain")]
                )
            ],
        )
        
        # Print the response
        print("\nResponse from LLM agent:")
        for message in run.output:
            if message.role == "assistant":
                for part in message.parts:
                    if part.content_type == "text/plain":
                        print(f"\n{part.content}")
        
        # Ask a follow-up question to test conversation history
        run = await client.run_sync(
            agent="llm_assistant",
            input=[
                Message(
                    parts=[MessagePart(content="Can you give an example of how it's implemented?", content_type="text/plain")]
                )
            ],
            run_id=run.run_id,  # Reuse the same run ID to continue the conversation
        )
        
        # Print the response
        print("\nFollow-up response:")
        for message in run.output:
            if message.role == "assistant":
                for part in message.parts:
                    if part.content_type == "text/plain":
                        print(f"\n{part.content}")


if __name__ == "__main__":
    asyncio.run(example())
