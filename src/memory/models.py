from typing import Optional
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field


class Memory(SQLModel, table=True):
    """
    SQLModel table for storing agent memories.

    Columns:
      - id: Auto-increment primary key
      - agent_id: Identifier of the agent (e.g., "ScraperAgent:parse")
      - step: Name of the action or step taken
      - timestamp: When the memory was stored
      - content: The serialized result or note
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str = Field(index=True, description="Agent unique identifier")
    step: str = Field(description="Action or step name")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp"
    )
    content: str = Field(description="Result or note to remember")
