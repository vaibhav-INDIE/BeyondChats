import os
import json
import argparse
from typing import List, Optional
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- PERSONA SCHEMA ---

class PersonaPoint(BaseModel):
    point: str
    evidence: List[str]

class Demographics(BaseModel):
    age: Optional[str]
    gender: Optional[str]
    location: Optional[str]
    occupation: Optional[str]
    education: Optional[str]

class Psychographics(BaseModel):
    mbti_type: Optional[str]
    archetype: Optional[str]
    tech_adoption_tier: Optional[str]

class UserPersona(BaseModel):
    username: str
    summary_bio: str
    demographics: Demographics
    psychographics: Psychographics
    interests_and_hobbies: List[PersonaPoint]
    personality_traits: List[PersonaPoint]
    communication_style: List[PersonaPoint]
    values_and_beliefs: List[PersonaPoint]
    goals_and_motivations: List[PersonaPoint]
    pain_points_and_frustrations: List[PersonaPoint]

# --- PDF GENERATION FUNCTION ---

def generate_pdf_report(persona: UserPersona, filename: str):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Colors and Styles
    COLOR_PRIMARY = colors.HexColor("#2C3E50")
    COLOR_SECONDARY = colors.HexColor("#3498DB")
    COLOR_TEXT = colors.HexColor("#34495E")
    COLOR_LIGHT_TEXT = colors.HexColor("#7F8C8D")
    COLOR_BACKGROUND = colors.HexColor("#ECF0F1")

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle('title', parent=styles['h1'], fontName='Helvetica-Bold', fontSize=24, textColor=COLOR_PRIMARY, leading=30)
    style_h2 = ParagraphStyle('h2', parent=styles['h2'], fontName='Helvetica-Bold', fontSize=15, textColor=COLOR_PRIMARY, spaceBefore=20, spaceAfter=8, leading=18)
    style_h3 = ParagraphStyle('h3', parent=styles['h3'], fontName='Helvetica-Bold', fontSize=12, textColor=COLOR_SECONDARY, spaceBefore=14, spaceAfter=6)
    style_body = ParagraphStyle('body', parent=styles['BodyText'], fontName='Helvetica', fontSize=10, textColor=COLOR_TEXT, leading=14, spaceAfter=10)
    style_bullet_point = ParagraphStyle('bullet_point', parent=style_body, leftIndent=inch * 0.2, spaceAfter=2)
    style_bullet_evidence = ParagraphStyle('bullet_evidence', parent=style_body, fontName='Helvetica-Oblique', fontSize=9, textColor=COLOR_LIGHT_TEXT, leftIndent=inch * 0.35, spaceAfter=10)

    margin = inch
    content_x = margin
    content_width = width - 2 * margin

    def draw_footer(page_number):
        c.saveState()
        c.setFont('Helvetica', 8)
        c.setFillColor(COLOR_LIGHT_TEXT)
        c.drawString(margin, margin * 0.5, f"Deep Persona Report: u/{persona.username}")
        c.drawRightString(width - margin, margin * 0.5, f"Page {page_number}")
        c.restoreState()

    def check_page_break(current_y, required_space):
        if current_y < margin + required_space:
            draw_footer(c.getPageNumber())
            c.showPage()
            return height - margin
        return current_y

    def draw_paragraph(text, y, max_width, style):
        p = Paragraph(text, style)
        p_width, p_height = p.wrapOn(c, max_width, height)
        y = check_page_break(y, p_height)
        p.drawOn(c, content_x, y - p_height)
        return y - p_height
    
    def draw_section_header(text, y, style):
        header_y = draw_paragraph(text, y, content_width, style)
        return header_y


    current_y = height - margin

    # --- Page Header ---
    current_y = draw_paragraph(f"Deep Persona Report", current_y, content_width, style_title)
    current_y = draw_paragraph(f"u/{persona.username}", current_y - 5, content_width, style_h2)
    c.setStrokeColor(COLOR_SECONDARY)
    c.setLineWidth(2)
    c.line(margin, current_y - 5, width - margin, current_y - 5)
    current_y -= 30

    # --- Summary ---
    current_y = draw_section_header("Summary & Profile", current_y, style_h2)
    current_y = draw_paragraph(persona.summary_bio, current_y, content_width, style_body)

    # --- Demographics ---
    current_y = draw_section_header("Demographics", current_y, style_h2)
    demo_text = f"""
    <b>Age:</b> {persona.demographics.age or 'N/A'}<br/>
    <b>Gender:</b> {persona.demographics.gender or 'N/A'}<br/>
    <b>Location:</b> {persona.demographics.location or 'N/A'}<br/>
    <b>Occupation:</b> {persona.demographics.occupation or 'N/A'}<br/>
    <b>Education:</b> {persona.demographics.education or 'N/A'}
    """
    current_y = draw_paragraph(demo_text, current_y, content_width, style_body)

    # --- Psychographics ---
    current_y = draw_section_header("Psychographics", current_y, style_h2)
    psycho_text = f"""
    <b>MBTI Type:</b> {persona.psychographics.mbti_type or 'N/A'}<br/>
    <b>Archetype:</b> {persona.psychographics.archetype or 'N/A'}<br/>
    <b>Tech Adoption:</b> {persona.psychographics.tech_adoption_tier or 'N/A'}
    """
    current_y = draw_paragraph(psycho_text, current_y, content_width, style_body)

    # --- Detailed Insights ---
    category_map = {
        "Interests & Hobbies": persona.interests_and_hobbies,
        "Personality Traits": persona.personality_traits,
        "Values & Beliefs": persona.values_and_beliefs,
        "Communication Style": persona.communication_style,
        "Goals & Motivations": persona.goals_and_motivations,
        "Pain Points & Frustrations": persona.pain_points_and_frustrations
    }

    current_y = draw_section_header("Detailed Insights", current_y, style_h2)

    for title, points in category_map.items():
        if points:
            current_y = check_page_break(current_y, 80)
            current_y = draw_paragraph(title, current_y, content_width, style_h3)

            for point in points:
                bullet_text = f"• {point.point}"
                current_y = draw_paragraph(bullet_text, current_y, content_width, style_bullet_point)

                if point.evidence:
                    first_evidence = point.evidence[0].strip()
                    # --- CLEAN LINK ---
                    if first_evidence.startswith("http") or first_evidence.startswith("www"):
                        evidence_link = first_evidence.replace('"', '').replace('“', '').replace('”', '').strip().rstrip('/.')  # Basic cleaning
                        evidence_para = f'<a href="{evidence_link}" color="{COLOR_SECONDARY}">{evidence_link}</a>'
                    else:
                        evidence_para = f"<i>E.g., \"{first_evidence}\"</i>"

                    current_y = draw_paragraph(evidence_para, current_y, content_width, style_bullet_evidence)
                else:
                    current_y -= 4

    # Finalize the last page
    draw_footer(c.getPageNumber())
    c.save()
    print(f"\n✅ Successfully generated single-column PDF report: {filename}")


def run_report(username: str):
    json_filename = f"{username}_deep_persona.json"
    if not os.path.exists(json_filename):
        print(f"Persona JSON not found for {username}")
        return

    with open(json_filename, "r", encoding="utf-8") as f:
        persona_data = json.load(f)
        final_persona = UserPersona(**persona_data)

    pdf_filename = f"{username}_persona_report.pdf"
    generate_pdf_report(final_persona, pdf_filename)

