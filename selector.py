from crewai import LLM, Agent, Task, Process, Crew
from crewai.tools import tool
from pydantic import BaseModel
from typing import List
from firebase_admin import credentials, firestore
import firebase_admin
import os
import json
import math
from dotenv import load_dotenv

# ── إعداد Firebase ─────────────────────────────────────────────
cred = credentials.Certificate("transport-assistant-2c59e-firebase-adminsdk-fbsvc-1d85cc11f1.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)
load_dotenv()
os.environ["CREWAI_STORAGE_DIR"] = os.path.join(output_dir, "crewai_db")
os.makedirs(os.environ["CREWAI_STORAGE_DIR"], exist_ok=True)
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv('api_key_llm1'),
    temperature=0.5,
)

# ── الفلترة تتم هنا في Python مباشرة ─────────────────────────
def filter_lines_by_proximity(lat1, long1, lat2, long2, threshold=0.05):
    """
    تجلب كل الخطوط من Firebase وتفلترها حسب القرب من النقطتين.
    threshold: بالدرجات (~5 كم)
    """
    route_docs = db.collection("routes").stream()
    filtered = []

    for doc in route_docs:
        data = doc.to_dict()
        points = data.get("points", [])

        if not points:
            continue

        has_start = False
        has_end = False

        for p in points:
            plat = p.get("lat") or p.get("latitude", 0)
            plng = p.get("lng") or p.get("longitude", 0)

            dist_start = math.sqrt((lat1 - plat)**2 + (long1 - plng)**2)
            dist_end   = math.sqrt((lat2 - plat)**2 + (long2 - plng)**2)

            if dist_start < threshold:
                has_start = True
            if dist_end < threshold:
                has_end = True

        if has_start and has_end:
            filtered.append(doc.id)

    return filtered


# ── Schemas ────────────────────────────────────────────────────
class RankedLineEntry(BaseModel):
    line_name: str
    score: float

class RankedLines(BaseModel):
    routes: List[RankedLineEntry]


# ── Tool للـ Agent ─────────────────────────────────────────────
def make_filtered_lines_tool(filtered_lines: list):
    @tool
    def get_filtered_lines() -> str:
        """Returns the pre-filtered list of transport lines available for the user."""
        return json.dumps(filtered_lines)
    return get_filtered_lines


# ── Crew ───────────────────────────────────────────────────────
def init_agents_and_crew(filtered_lines: list):

    get_filtered_lines = make_filtered_lines_tool(filtered_lines)

    agent_ranker = Agent(
        name="agent_ranker",
        role="Transport line ranker",
        goal="Rank the given transport lines based on user preferences for cost, time, and comfort.",
        backstory="You receive a filtered list of lines and sort them using a scoring formula.",
        llm=llm,
        tools=[get_filtered_lines],
        verbose=True,
    )

    task_rank = Task(
        description="""
        Call get_filtered_lines() to get the available lines.

        User preferences:
        - Cost:    {Cost}    (low / medium / high)
        - Time:    {time}    (low / medium / high)
        - Comfort: {Comfort} (low / medium / high)

        Scoring table (lower = better):
        | Type                              | cost_rank | time_rank | comfort_rank |
        |-----------------------------------|-----------|-----------|--------------|
        | bus, L12, L36, L58, L89A, L608A  |     1     |     4     |      4       |
        | tram                              |     2     |     3     |      3       |
        | metro                             |     3     |     2     |      2       |
        | teleferik                         |     2     |     3     |      3       |
        | taxi                              |     4     |     1     |      1       |

        Priority weight:
        - "high"   → 3
        - "medium" → 2
        - "low"    → 1

        score = (cost_weight * cost_rank) + (time_weight * time_rank) + (comfort_weight * comfort_rank)
        Lower score = better → put first.

        Return ONLY valid JSON:
        {{
          "routes": [
            {{"line_name": "L12",   "score": 9.0}},
            {{"line_name": "tram",  "score": 11.0}}
          ]
        }}
        """,
        expected_output="Valid JSON object only.",
        output_json=RankedLines,
        output_file=os.path.join(output_dir, "step_ranked.json"),
        agent=agent_ranker,
    )

    crew = Crew(
        agents=[agent_ranker],
        tasks=[task_rank],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def run_selector(inputs: dict):
    # الفلترة في Python مباشرة
    filtered = filter_lines_by_proximity(
        inputs['lat1'], inputs['long1'],
        inputs['lat2'], inputs['long2'],
        threshold=0.05
    )

    print(f"✅ Filtered lines: {filtered}")

    if not filtered:
        return {"routes": [], "message": "No lines found near the given points"}

    # الـ Agent فقط يرتب
    crew = init_agents_and_crew(filtered)
    result = crew.kickoff(inputs=inputs)
    return json.loads(result.raw)