"""
========================================================
report_generator.py — PDF Coaching Report Generator
========================================================

WHAT THIS FILE DOES:
--------------------
Generates a multi-page PDF coaching report after a session,
including overall scores, SHAP AI tips, and metric breakdowns.

AUTHOR: Cricket Biomechanics Analyzer V2
"""

from fpdf import FPDF
import os
import datetime

class CoachingReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(0, 100, 0)
        self.cell(0, 10, 'CRICKET BIOMECHANICS AI', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Performance & Technique Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(player_name, session_scores, top_shot, top_phase, ai_tips):
    """
    Generates a PDF report and returns the file path.
    """
    pdf = CoachingReport()
    pdf.add_page()
    
    # Date & Player Info
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Player: {player_name}", 0, 1)
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"Date: {date_str}", 0, 1)
    
    pdf.ln(5)
    
    # Overall Score Box
    pdf.set_font('Helvetica', 'B', 16)
    ov = session_scores.get('overall_score', 0)
    pdf.cell(0, 15, f"OVERALL SCORE: {ov} / 100", border=1, ln=1, align='C', fill=False)
    
    pdf.ln(10)
    
    # Core Metrics
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, "Core Mechanics:", 0, 1)
    pdf.set_font('Helvetica', '', 12)
    
    metrics = [
        ("Balance", session_scores.get('balance_score', 0)),
        ("Stability", session_scores.get('stability_score', 0)),
        ("Power", session_scores.get('power_score', 0)),
        ("Timing", session_scores.get('timing_score', 0))
    ]
    
    for name, score in metrics:
        pdf.cell(50, 10, f"{name}:", 0, 0)
        pdf.cell(50, 10, f"{score}/100", 0, 1)
        
    pdf.ln(10)
    
    # Analysis
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, "Session Analysis:", 0, 1)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"Primary Shot Detected: {top_shot}", 0, 1)
    pdf.cell(0, 10, f"Dominant Phase: {top_phase}", 0, 1)
    
    pdf.ln(10)
    
    # AI Coaching Tips
    if ai_tips:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, "AI Coach Insights (SHAP Explanation):", 0, 1)
        pdf.set_font('Helvetica', '', 12)
        
        for tip in ai_tips:
            # Handle unicode arrows/emojis by replacing or removing
            safe_tip = tip.encode('ascii', 'ignore').decode('ascii')
            pdf.write(8, f"* {safe_tip}\n")
            
    pdf.ln(5)
    
    # Injury Risks
    if session_scores.get('injury_flags'):
        pdf.set_text_color(200, 0, 0)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, "Injury Risk Warnings:", 0, 1)
        pdf.set_font('Helvetica', '', 12)
        for flag in session_scores['injury_flags']:
            safe_flag = flag.encode('ascii', 'ignore').decode('ascii')
            pdf.write(8, f"! {safe_flag}\n")
            
    # Save PDF
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    safe_name = "".join(c for c in player_name if c.isalnum() or c in " _-").strip()
    file_path = os.path.join(reports_dir, f"{safe_name}_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pdf")
    
    pdf.output(file_path)
    return file_path
