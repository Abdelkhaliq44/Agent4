from crewai import LLM, Agent, Task, Process, Crew
from crewai.tools import tool
from pydantic import BaseModel
from typing import List
from firebase_admin import credentials, firestore
import firebase_admin
import os
import json
from dotenv import load_dotenv
import math

# ── إعداد Firebase ─────────────────────────────────────────────
cred = credentials.Certificate("transport-assistant-2c59e-firebase-adminsdk-fbsvc-1d85cc11f1.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)
load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv('api_key_llm1'),
    temperature=0.5,
)

# ── Tools ──────────────────────────────────────────────────────

@tool
def get_all_lines_with_markers() -> str:
    """
    Fetches all transport lines from Firebase 'routes' collection.
    For each line, also fetches its markers from 'markers' collection.
    Returns JSON string with line names and their station markers.
    """
    # جلب أسماء الخطوط من routes
    route_docs = db.collection("routes").stream()
    line_names = [doc.id for doc in route_docs]

    result = []
    for line_name in line_names:
        # جلب markers لكل خط
        marker_doc = db.collection("markers").document(line_name).get()
        markers = []
        if marker_doc.exists:
            data = marker_doc.to_dict()
            points = data.get("points", [])
            for p in points:
                markers.append({
                    "lat": p.get("latitude") or p.get("lat", 0),
                    "lng": p.get("longitude") or p.get("lng", 0),
                })
        result.append({
            "line_name": line_name,
            "markers": markers
        })

    return json.dumps(result)


# ── Schemas ────────────────────────────────────────────────────

class LineEntry(BaseModel):
    line_name: str

class FilteredLines(BaseModel):
    routes: List[LineEntry]

class RankedLineEntry(BaseModel):
    line_name: str
    score: float

class RankedLines(BaseModel):
    routes: List[RankedLineEntry]


# ── Crew ───────────────────────────────────────────────────────

def init_agents_and_crew():

    # ── Agent 1: جلب الخطوط وفلترتها ──────────────────────────
    agent_filter = Agent(
        name="agent_filter",
        role="Transport line filter",
        goal=(
            "Fetch all transport lines with their markers from Firebase, "
            "then keep only lines where at least one marker is close to "
            "the starting point AND at least one marker is close to the arrival point."
        ),
        backstory=(
            "You are stage 1. You decide which transport lines are physically "
            "reachable for the user based on proximity of stations."
        ),
        llm=llm,
        tools=[get_all_lines_with_markers],
        verbose=True,
    )

    task_filter = Task(
        description="""
        Call the tool get_all_lines_with_markers() to get all lines and their station markers.

        User positions:
        - starting_point: ({lat1}, {long1})
        - arrival_point:  ({lat2}, {long2})

        For each line:
        1. Check if ANY marker is within 0.02 degrees of starting_point ({lat1}, {long1})
        2. Check if ANY marker is within 0.02 degrees of arrival_point ({lat2}, {long2})
        3. Keep the line ONLY if BOTH conditions are true.

        Distance formula: sqrt((lat1 - marker_lat)^2 + (long1 - marker_lng)^2) < 0.02

        Return ONLY valid JSON:
        {{
          "routes": [
            {{"line_name": "L12"}},
            {{"line_name": "tram"}}
          ]
        }}
        """,
        expected_output="Valid JSON object only.",
        output_json=FilteredLines,
        output_file=os.path.join(output_dir, "step1_filtered.json"),
        agent=agent_filter,
    )

    # ── Agent 2: ترتيب الخطوط حسب تفضيلات المستخدم ────────────
    agent_ranker = Agent(
        name="agent_ranker",
        role="Transport line ranker",
        goal=(
            "Rank the filtered transport lines based on user preferences "
            "for cost, time, and comfort. Return them sorted best-first."
        ),
        backstory=(
            "You are stage 2. You receive filtered lines and sort them "
            "using a scoring formula based on user priorities."
        ),
        llm=llm,
        verbose=True,
    )

    task_rank = Task(
        description="""
        You receive filtered lines from the previous agent (agent_filter).

        User preferences:
        - Cost:    {Cost}    (low / medium / high)
        - Time:    {time}    (low / medium / high)
        - Comfort: {Comfort} (low / medium / high)

        Transport type scoring table (lower = better):
        | Type            | cost_rank | time_rank | comfort_rank |
        |-----------------|-----------|-----------|--------------|
        | bus, L12, L36,  |     1     |     4     |      4       |
        | L58, L89A,L608A |           |           |              |
        | tram            |     2     |     3     |      3       |
        | metro           |     3     |     2     |      2       |
        | teleferik       |     2     |     3     |      3       |
        | taxi            |     4     |     1     |      1       |

        Priority weight:
        - "high"   → 3
        - "medium" → 2
        - "low"    → 1

        Formula:
        score = (cost_weight * cost_rank) + (time_weight * time_rank) + (comfort_weight * comfort_rank)

        Lower score = better → put first.

        Return ONLY valid JSON sorted by score ascending:
        {{
          "routes": [
            {{"line_name": "metro",    "score": 7.0}},
            {{"line_name": "tram",     "score": 9.0}},
            {{"line_name": "L12",      "score": 11.0}}
          ]
        }}
        """,
        expected_output="Valid JSON object only.",
        output_json=RankedLines,
        output_file=os.path.join(output_dir, "step2_ranked.json"),
        context=[task_filter],
        agent=agent_ranker,
    )

    # ── الـ Crew ────────────────────────────────────────────────
    crew = Crew(
        agents=[agent_filter, agent_ranker],
        tasks=[task_filter, task_rank],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def run_selector(inputs: dict):
    crew = init_agents_and_crew()
    result = crew.kickoff(inputs=inputs)
    return json.loads(result.raw)