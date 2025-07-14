# builder.py

import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# PDF Generation imports from your provided code
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# --- Schemas (Must match the schemas in your profiler.py) ---

class PersonaPoint(BaseModel):
    point: str
    citations: List[str] = []

class Demographics(BaseModel):
    age: Optional[str] = Field(default="Unknown")
    location: Optional[str] = Field(default="Unknown")
    marital_status: Optional[str] = Field(default="Unknown")
    occupation: Optional[str] = Field(default="Unknown")

class Psychographics(BaseModel):
    archetype: Optional[str] = Field(default="Unknown")
    tier: Optional[str] = Field(default="Unknown")
    introvert_extrovert: Optional[str] = Field(default="Unknown")
    intuitive_sensing: Optional[str] = Field(default="Unknown")
    thinking_feeling: Optional[str] = Field(default="Unknown")
    judging_perceiving: Optional[str] = Field(default="Unknown")

class UserPersona(BaseModel):
    name: str # Add name field for the title
    summary_bio: str
    demographics: Demographics
    psychographics: Psychographics
    interests_and_hobbies: List[PersonaPoint]
    personality_traits: List[PersonaPoint]
    values: List[PersonaPoint]
    motivations: List[PersonaPoint]
    communication_style: List[PersonaPoint]


def load_raw_insights(filepath: str) -> List[Dict]:
    """Loads a JSONL file containing raw insights."""
    if not os.path.exists(filepath):
        print(f"Error: Raw insights file not found at '{filepath}'")
        return []
    insights = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                insights.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line}")
    return insights

def synthesize_final_persona(raw_insights: List[Dict], username: str, client: OpenAI) -> Optional[Dict]:
    """Summarizes insight categories to stay under token limits, then builds a final persona."""

    from collections import defaultdict

    grouped = defaultdict(list)
    for item in raw_insights:
        for k, v in item.items():
            if isinstance(v, list):
                grouped[k].extend(v)
            elif isinstance(v, str):
                grouped[k].append(v)

    summarized_insights = {}

    print(f"Summarizing grouped insights (across {len(grouped)} categories)...")

    for category, texts in grouped.items():
        print(f"Summarizing: {category} ({len(texts)} items)")
        if not texts:
            continue
        chunk = "\n".join(f"- {t}" for t in texts[:25])  # Cap to 25 items per category

        prompt = (
            f"As an expert in behavioral analysis, summarize the following user-related insights into a coherent paragraph that captures patterns, themes, or traits for the category: '{category}'. "
            f"Focus on merging repetitive points and infer meaning. Skip citations.\n\n"
            f"Insights:\n{chunk}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            summarized_insights[category] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Failed to summarize category '{category}': {e}")
            summarized_insights[category] = "Unknown"

    # ---- Final Persona Construction Prompt ----
    print("\nComposing final persona prompt from summaries...")
    formatted_summary = json.dumps(summarized_insights, indent=2)

    final_prompt = (
        f"You are a master persona synthesis expert. Below are summarized user insight categories derived from Reddit comment analysis for '{username}'. "
        "Use these summaries to construct a complete user persona object in JSON format that conforms exactly to this schema:\n\n"
        f"{json.dumps(UserPersona.model_json_schema(), indent=2)}\n\n"
        "Summarized Insight Data:\n"
        f"{formatted_summary}\n\n"
        "Output only a valid JSON object. Do not include explanations or markdown formatting."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}]
        )
        raw_json = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        validated_persona = UserPersona.model_validate_json(raw_json)
        return validated_persona.model_dump()
    except Exception as e:
        print(f"🚫 Final persona generation failed: {e}")
        return None
    """Uses an LLM to synthesize raw insights into a final, coherent persona JSON."""
    
    # Format the raw insights for the prompt
    formatted_insights = json.dumps(raw_insights, indent=2)
    
    final_prompt = (
        f"You are a master persona synthesis expert. Your task is to analyze the following collection of raw, fragmented insights that were extracted from the user '{username}'s comments. These insights are grouped by thematic clusters.\n\n"
        "Your goal is to consolidate all this information into a single, well-written, and coherent JSON object that represents the user's final persona.\n\n"
        "Follow these rules carefully:\n"
        "1. **Synthesize, Don't Just List:** Read all insights for a given field (e.g., all demographic clues) and create a single, definitive summary. For narrative sections like the 'summary_bio', weave together information from all relevant categories.\n"
        "2. **Resolve Contradictions:** If different clusters provide conflicting information (e.g., 'Introvert' vs. 'Extrovert'), use your judgment to determine the most likely trait or classify it as 'Balanced' or 'Ambivert'.\n"
        "3. **Adhere Strictly to the Schema:** The final output MUST be a valid JSON object that perfectly matches this schema: \n"
        f"{json.dumps(UserPersona.model_json_schema(), indent=2)}\n\n"
        f"Here are the raw insights for user '{username}':\n{formatted_insights}\n\n"
        "Generate the complete, final, and valid JSON object now."
    )
    
    print("Sending raw insights to AI for final synthesis...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a more powerful model for the final, complex synthesis is better
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        raw_json = response.choices[0].message.content.strip()
        # Validate the JSON against our Pydantic model
        validated_persona = UserPersona.model_validate_json(raw_json)
        return validated_persona.model_dump() # Return as a dictionary
    except Exception as e:
        print(f"An error occurred during final persona synthesis: {e}")
        return None

# --- PDF Generation (Adapted from your provided code) ---

def render_persona_to_pdf(user_data: dict, output_file="persona_resume_style.pdf"):
    """Renders the final persona dictionary to a PDF file."""
    c = canvas.Canvas(output_file, pagesize=A4)
    c.setTitle("User Persona")

    name = user_data.get("name", "Unknown User")
    
    # Draw Title
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.HexColor("#ec6608"))
    c.drawString(320, 770, name)

    # --- Left Column ---
    x_left = 50
    
    # Demographics Block
    y_current = 700
    demo = user_data.get("demographics", {})
    psych = user_data.get("psychographics", {})
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_left, y_current, "DETAILS")
    c.line(x_left, y_current - 5, x_left + 230, y_current - 5)
    y_current -= 20
    c.setFont("Helvetica", 9)
    details = [
        f"AGE: {demo.get('age', 'N/A')}",
        f"OCCUPATION: {demo.get('occupation', 'N/A')}",
        f"STATUS: {demo.get('marital_status', 'N/A')}",
        f"LOCATION: {demo.get('location', 'N/A')}",
        f"TIER: {psych.get('tier', 'N/A')}",
        f"ARCHETYPE: {psych.get('archetype', 'N/A')}",
    ]
    for detail in details:
        c.drawString(x_left, y_current, detail)
        y_current -= 14

    # Motivations Block
    y_current = 560
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_left, y_current, "MOTIVATIONS")
    c.line(x_left, y_current - 5, x_left + 230, y_current - 5)
    y_current -= 20
    c.setFont("Helvetica", 9)
    for motivation in user_data.get("motivations", []):
        c.drawString(x_left, y_current, f"• {motivation['point']}")
        y_current -= 13
        if y_current < 450: break # Stop if it gets too long

    # Personality Block
    y_current = 420
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_left, y_current, "PERSONALITY")
    c.line(x_left, y_current - 5, x_left + 230, y_current - 5)
    y_current -= 20
    c.setFont("Helvetica", 9)
    personality = [
        f"Introvert/Extrovert: {psych.get('introvert_extrovert', 'N/A')}",
        f"Intuitive/Sensing: {psych.get('intuitive_sensing', 'N/A')}",
        f"Thinking/Feeling: {psych.get('thinking_feeling', 'N/A')}",
        f"Judging/Perceiving: {psych.get('judging_perceiving', 'N/A')}"
    ]
    for trait in personality:
        c.drawString(x_left, y_current, trait)
        y_current -= 13


    # --- Right Column ---
    x_right = 320
    def draw_right_column_section(canvas, title, points_list, y_start):
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(colors.HexColor("#000000"))
        canvas.drawString(x_right, y_start, title.upper())
        canvas.line(x_right, y_start - 5, x_right + 230, y_start - 5)
        y = y_start - 20
        canvas.setFont("Helvetica", 9)
        for item in points_list:
            # Simple wrapping logic
            text = f"• {item['point']}"
            lines = [text[i:i+60] for i in range(0, len(text), 60)]
            for line in lines:
                if y < 100: return y # Stop if we run out of page
                canvas.drawString(x_right, y, line)
                y -= 12
            y -= 3 # Extra space between points
        return y
    
    y_right = 700
    y_right = draw_right_column_section(c, "Behaviour & Habits", user_data.get("communication_style", []), y_right)
    
    y_right -= 20 # Add space between sections
    y_right = draw_right_column_section(c, "Frustrations", user_data.get("values", []), y_right)

    y_right -= 20
    draw_right_column_section(c, "Goals & Needs", user_data.get("motivations", []), y_right)

    c.save()
    print(f"✅ PDF resume saved to {output_file}")


# --- Main Execution Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a final persona from raw insights and generate a PDF.")
    parser.add_argument("username", help="The Reddit username for which to build the persona.")
    args = parser.parse_args()

    # Load environment variables and initialize client
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        exit()
    client = OpenAI(api_key=api_key)

    # 1. Load the raw insights
    raw_insights_file = f"{args.username}_insights_raw.jsonl"
    insights = load_raw_insights(raw_insights_file)
    if not insights:
        print("Exiting.")
        exit()

    # 2. Synthesize into a final persona
    final_persona_dict = synthesize_final_persona(insights, args.username, client)

    if final_persona_dict:
        # 3. Save the final clean JSON for review
        final_json_path = f"{args.username}_final_persona.json"
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_persona_dict, f, indent=4)
        print(f"✓ Final synthesized persona saved to '{final_json_path}'")

        # 4. Render the PDF
        pdf_output_file = f"{args.username}_persona_resume.pdf"
        render_persona_to_pdf(final_persona_dict, output_file=pdf_output_file)