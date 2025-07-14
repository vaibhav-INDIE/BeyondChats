import os
import json
import pickle
import argparse
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
NUM_CLUSTERS = 10
MAX_ITEMS_PER_CLUSTER_FOR_ANALYSIS = 50
REDDIT_BASE_URL = "https://www.reddit.com"

# --- PERSONA SCHEMA ---

class PersonaPoint(BaseModel):
    point: str = Field(description="A specific, evidence-based insight about the user.")
    evidence: List[str] = Field(description="Direct quotes or summaries of posts that support the insight, including a citation.", default=[])

class Demographics(BaseModel):
    age: Optional[str] = Field(default="Unknown", description="Estimated age range (e.g., 25-30).")
    gender: Optional[str] = Field(default="Unknown", description="Inferred gender.")
    location: Optional[str] = Field(default="Unknown", description="Inferred country, state, or city.")
    occupation: Optional[str] = Field(default="Unknown", description="Inferred profession or field of study.")
    education: Optional[str] = Field(default="Unknown", description="Inferred educational level (e.g., College, PhD).")

class Psychographics(BaseModel):
    mbti_type: Optional[str] = Field(default="Unknown", description="Estimated Myers-Briggs type (e.g., INTP).")
    archetype: Optional[str] = Field(default="Unknown", description="Primary archetype (e.g., Sage, Explorer, Rebel).")
    tech_adoption_tier: Optional[str] = Field(default="Unknown", description="e.g., Innovator, Early Adopter, Laggard.")

class UserPersona(BaseModel):
    username: str = Field(description="The user's Reddit username.")
    summary_bio: str = Field(description="A 2-3 sentence narrative biography summarizing the user's core identity.")
    demographics: Demographics
    psychographics: Psychographics
    interests_and_hobbies: List[PersonaPoint] = Field(description="Key interests, hobbies, and passions.")
    personality_traits: List[PersonaPoint] = Field(description="Core personality characteristics (e.g., Analytical, Empathetic).")
    communication_style: List[PersonaPoint] = Field(description="How the user communicates (e.g., Formal, Sarcastic, Uses emojis).")
    values_and_beliefs: List[PersonaPoint] = Field(description="What the user holds as important (e.g., Honesty, Community).")
    goals_and_motivations: List[PersonaPoint] = Field(description="What drives the user's behavior and comments.")
    pain_points_and_frustrations: List[PersonaPoint] = Field(description="Topics or situations that frustrate the user.")

# --- CORE FUNCTIONS ---

def load_vector_db(username: str):
    index_file = f"{username}_reddit.faiss"
    meta_file = f"{username}_reddit_meta.pkl"
    if not all(os.path.exists(f) for f in [index_file, meta_file]):
        print(f"Error: Database files not found for '{username}'. Please run the vector DB creation script first.")
        return None, None
    print("Loading vector database and metadata...")
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def find_clusters_kmeans(index: faiss.Index, num_clusters: int) -> List[List[int]]:
    print(f"Running K-Means to create {num_clusters} thematic clusters...")
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)]).astype("float32")
    kmeans = faiss.Kmeans(d=vectors.shape[1], k=num_clusters, niter=20, verbose=False)
    kmeans.train(vectors)
    _, labels = kmeans.index.search(vectors, 1)

    clusters = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels.flatten()):
        clusters[label].append(i)
    return [cluster for cluster in clusters if cluster]

def synthesize_insight(texts: str, prompt_template: str, client: OpenAI) -> str:
    prompt = prompt_template.format(texts=texts)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM synthesis error: {e}")
        return ""

def generate_raw_insights(clusters: List[List[int]], metadata: List[Dict], client: OpenAI, username: str):
    insights_file = f"{username}_insights_raw.jsonl"
    if os.path.exists(insights_file):
        os.remove(insights_file)

    prompt_templates = {
        "interests": "Based on these comments, what is one specific interest or hobby? Example: 'Playing vintage guitars'.\n\nTexts:\n{texts}\n\nInterest:",
        "personality": "Analyze the tone. What is one key personality trait? Example: 'Deeply analytical and skeptical'.\n\nTexts:\n{texts}\n\nTrait:",
        "communication": "Describe one aspect of the user's communication style. Example: 'Uses technical jargon frequently'.\n\nTexts:\n{texts}\n\nStyle:",
        "values": "What is one core value or belief expressed here? Example: 'Believes in open-source collaboration'.\n\nTexts:\n{texts}\n\nValue:",
        "goals": "What is a primary motivation for these comments? Example: 'To find solutions to a technical problem'.\n\nTexts:\n{texts}\n\nGoal:",
        "pain_points": "What is a source of frustration or a pain point for this user? Example: 'Frustrated with inefficient software'.\n\nTexts:\n{texts}\n\nPain Point:",
    }

    print("Generating raw insights from comment clusters...")
    for i, cluster_indices in enumerate(clusters):
        print(f"  Processing Cluster {i+1}/{len(clusters)}")
        cluster_metadata = [metadata[idx] for idx in cluster_indices][:MAX_ITEMS_PER_CLUSTER_FOR_ANALYSIS]
        cluster_texts = "\n---\n".join([item['original_content'] for item in cluster_metadata])
        
        cluster_evidence = []
        for item in cluster_metadata[:3]:
            content = item.get('original_content', '[Content not available]')
            url = item.get('source_url', '')
            citation = f"'{content[:150].strip()}...' (Source: {REDDIT_BASE_URL}{url})"
            cluster_evidence.append(citation)

        for key, template in prompt_templates.items():
            insight = synthesize_insight(cluster_texts, template, client)
            if insight and insight.lower() not in ["none", "unknown", "n/a"]:
                insight_entry = {"category": key, "insight": insight, "evidence": cluster_evidence}
                with open(insights_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(insight_entry) + "\n")

def build_final_persona(username: str, client: OpenAI) -> Optional[UserPersona]:
    insights_file = f"{username}_insights_raw.jsonl"
    if not os.path.exists(insights_file):
        print("Raw insights file not found. Cannot build final persona.")
        return None

    with open(insights_file, "r", encoding="utf-8") as f:
        raw_insights = [json.loads(line) for line in f]

    final_prompt = (
        f"You are a master psychological and behavioral analyst. Your task is to synthesize the following raw data points, extracted from Reddit user '{username}'s comments, into a complete and coherent JSON persona. "
        "Adhere STRICTLY to the provided JSON schema. Consolidate related points, eliminate duplicates, infer demographic and psychographic data from the overall context, and write a compelling summary bio.\n\n"
        f"**JSON Schema to follow:**\n{json.dumps(UserPersona.model_json_schema(), indent=2)}\n\n"
        f"**Raw Insights Data:**\n{json.dumps(raw_insights, indent=2)}\n\n"
        "Generate the complete JSON object now. Do not include any text or markdown outside of the JSON object itself."
    )

    print("\nSynthesizing final persona with GPT-4o...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        persona_json = response.choices[0].message.content
        return UserPersona.model_validate_json(persona_json)
    except Exception as e:
        print(f"Final persona generation failed: {e}")
        return None

# --- MAIN EXECUTION LOGIC ---

def run_profiler(username: str):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    index, metadata = load_vector_db(username)
    if not index:
        print(f"Failed to load vector DB for {username}")
        return

    num_points = index.ntotal
    max_clusters_allowed = max(1, num_points // 39)
    num_clusters_to_use = min(NUM_CLUSTERS, max_clusters_allowed)

    if num_points < num_clusters_to_use:
        print(f"Insufficient data ({num_points}) to create {num_clusters_to_use} clusters.")
        return

    clusters = find_clusters_kmeans(index, num_clusters_to_use)
    generate_raw_insights(clusters, metadata, client, username)
    final_persona = build_final_persona(username, client)

    if final_persona:
        json_filename = f"{username}_deep_persona.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(final_persona.model_dump_json(indent=2))
        print(f"Saved persona to {json_filename}")
    else:
        print("Failed to generate final persona.")