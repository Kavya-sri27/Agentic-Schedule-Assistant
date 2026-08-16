
import os
import json
import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai

# =========================================================
# Configuration
# =========================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "schedule.json"
)

CHROMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "chroma_db"
)

# =========================================================
# Gemini
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

client = genai.Client(api_key=GEMINI_API_KEY)

# =========================================================
# Embeddings + ChromaDB
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name="schedule"
)

# =========================================================
# Load schedule
# =========================================================

with open(DATA_FILE, "r") as f:
    events = json.load(f)


def rebuild_chroma():

    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    documents = []
    ids = []
    metadatas = []

    for event in events:

        document = (
            f"Event: {event['title']}. "
            f"Type: {event['type']}. "
            f"Date: {event['date']}. "
            f"Time: {event['start_time']} "
            f"to {event['end_time']}. "
            f"Description: {event.get('description', '')}"
        )

        documents.append(document)
        ids.append(event["id"])

        metadatas.append({
            "event_id": event["id"],
            "title": event["title"],
            "type": event["type"],
            "date": event["date"],
            "start_time": event["start_time"],
            "end_time": event["end_time"]
        })

    if documents:

        embeddings = embedding_model.encode(
            documents
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )


# Populate ChromaDB if empty
if collection.count() == 0:
    rebuild_chroma()


# =========================================================
# Tool 1: get_schedule
# =========================================================

def get_schedule(query: str):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(5, collection.count())
    )

    found = []

    if results["metadatas"]:

        for metadata in results["metadatas"][0]:

            found.append({
                "event_id": metadata["event_id"],
                "title": metadata["title"],
                "type": metadata["type"],
                "date": metadata["date"],
                "start_time": metadata["start_time"],
                "end_time": metadata["end_time"]
            })

    return found


# =========================================================
# Tool 2: update_schedule
# =========================================================

def update_schedule(
    action,
    event_id=None,
    title=None,
    event_type=None,
    date=None,
    start_time=None,
    end_time=None,
    description=None
):

    global events

    # -------------------------
    # ADD
    # -------------------------

    if action == "add":

        numbers = []

        for event in events:

            try:
                numbers.append(
                    int(event["id"].split("_")[1])
                )
            except:
                pass

        next_id = max(numbers, default=0) + 1

        new_event = {
            "id": f"event_{next_id:03d}",
            "title": title or "Untitled Event",
            "type": event_type or "task",
            "date": date,
            "start_time": start_time or "00:00",
            "end_time": end_time or "01:00",
            "description": description or ""
        }

        events.append(new_event)

    # -------------------------
    # UPDATE
    # -------------------------

    elif action == "update":

        target = None

        for event in events:

            if event["id"] == event_id:
                target = event
                break

        if target is None:
            return {
                "success": False,
                "message": "Event not found."
            }

        if title is not None:
            target["title"] = title

        if event_type is not None:
            target["type"] = event_type

        if date is not None:
            target["date"] = date

        if start_time is not None:
            target["start_time"] = start_time

        if end_time is not None:
            target["end_time"] = end_time

        if description is not None:
            target["description"] = description

    # -------------------------
    # DELETE
    # -------------------------

    elif action == "delete":

        original_length = len(events)

        events = [
            event
            for event in events
            if event["id"] != event_id
        ]

        if len(events) == original_length:

            return {
                "success": False,
                "message": "Event not found."
            }

    else:

        return {
            "success": False,
            "message": "Invalid action."
        }

    # Save
    with open(DATA_FILE, "w") as f:
        json.dump(events, f, indent=2)

    # Synchronize vector database
    rebuild_chroma()

    return {
        "success": True,
        "action": action,
        "message": "Schedule updated successfully."
    }


# =========================================================
# Gemini tools
# =========================================================

SYSTEM_PROMPT = """
You are an Agentic RAG Schedule Assistant.

Manage the user's schedule for the next 30 days.

Use get_schedule whenever the user asks about:
- events
- meetings
- appointments
- workshops
- tasks
- dates
- times
- availability

Use update_schedule when the user wants to:
- add an event
- update an event
- move an event
- delete an event

Never invent schedule information.

Keep responses concise and useful.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_schedule",
        "description": "Retrieve relevant schedule information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string"
                }
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "update_schedule",
        "description": "Add, update, or delete a schedule event.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "delete"]
                },
                "event_id": {
                    "type": "string"
                },
                "title": {
                    "type": "string"
                },
                "event_type": {
                    "type": "string"
                },
                "date": {
                    "type": "string"
                },
                "start_time": {
                    "type": "string"
                },
                "end_time": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                }
            },
            "required": ["action"]
        }
    }
]


# =========================================================
# Agent
# =========================================================

def schedule_agent(user_query):

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=user_query,
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS
    )

    calls = [
        step
        for step in interaction.steps
        if step.type == "function_call"
    ]

    if not calls:
        return interaction.output_text

    results = []

    for call in calls:

        args = call.arguments

        if call.name == "get_schedule":

            result = get_schedule(
                args.get("query", "")
            )

        elif call.name == "update_schedule":

            result = update_schedule(
                action=args.get("action"),
                event_id=args.get("event_id"),
                title=args.get("title"),
                event_type=args.get("event_type"),
                date=args.get("date"),
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
                description=args.get("description")
            )

        else:

            result = {
                "success": False,
                "message": "Unknown tool"
            }

        results.append({
            "type": "function_result",
            "name": call.name,
            "call_id": call.id,
            "result": [
                {
                    "type": "text",
                    "text": json.dumps(
                        result,
                        default=str
                    )
                }
            ]
        })

    final = client.interactions.create(
        model="gemini-3.6-flash",
        previous_interaction_id=interaction.id,
        input=results
    )

    return final.output_text


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant"
)


class Query(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html>
<head>
<title>Agentic RAG Schedule Assistant</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: #f4f7fb;
    margin: 0;
}

.container {
    max-width: 850px;
    margin: 60px auto;
    background: white;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
}

h1 {
    margin-bottom: 8px;
}

.subtitle {
    color: #666;
    margin-bottom: 30px;
}

textarea {
    width: 100%;
    height: 100px;
    padding: 15px;
    box-sizing: border-box;
    border: 1px solid #ddd;
    border-radius: 10px;
    font-size: 16px;
}

button {
    margin-top: 15px;
    padding: 12px 25px;
    border: none;
    border-radius: 9px;
    background: #111827;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    opacity: 0.9;
}

#answer {
    margin-top: 25px;
    padding: 20px;
    background: #f8fafc;
    border-radius: 10px;
    white-space: pre-wrap;
    min-height: 50px;
}

.examples {
    margin-top: 25px;
}

.example {
    display: inline-block;
    padding: 8px 12px;
    margin: 5px;
    background: #eef2ff;
    border-radius: 20px;
    cursor: pointer;
}
</style>
</head>

<body>

<div class="container">

<h1>📅 Agentic RAG Schedule Assistant</h1>

<div class="subtitle">
Your AI-powered 30-day schedule manager
</div>

<textarea
id="query"
placeholder="Ask something like: What meetings do I have?"
></textarea>

<br>

<button onclick="askAgent()">
Ask Assistant
</button>

<div id="answer">
Your answer will appear here.
</div>

<div class="examples">

<strong>Try:</strong>

<div
class="example"
onclick="setQuery('What meetings do I have?')">
Meetings
</div>

<div
class="example"
onclick="setQuery('What do I have scheduled on August 20?')">
August 20
</div>

<div
class="example"
onclick="setQuery('Add a meeting on August 28 at 3 PM called Project Review')">
Add meeting
</div>

</div>

</div>

<script>

function setQuery(text) {
    document.getElementById("query").value = text;
}

async function askAgent() {

    const query =
        document.getElementById("query").value;

    const answer =
        document.getElementById("answer");

    if (!query) {
        answer.innerText =
            "Please enter a question.";
        return;
    }

    answer.innerText =
        "Thinking...";

    try {

        const response = await fetch(
            "/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    query: query
                })
            }
        );

        const data =
            await response.json();

        answer.innerText =
            data.answer || data.error;

    } catch (error) {

        answer.innerText =
            "Unable to connect to the assistant.";
    }
}

</script>

</body>
</html>
"""


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Agentic RAG Schedule Assistant"
    }


@app.post("/ask")
def ask(request: Query):

    try:

        answer = schedule_agent(
            request.query
        )

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:

        error = str(e)

        if "quota" in error.lower() or "429" in error:

            return {
                "success": False,
                "error":
                    "Gemini API quota is temporarily exceeded. "
                    "Please try again later."
            }

        return {
            "success": False,
            "error": error
        }
