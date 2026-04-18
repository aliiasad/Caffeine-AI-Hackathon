import os
from groq import Groq

def generate_cover_letter(opportunity: dict, profile: dict) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = f"""Write a professional, warm, and highly personalized cover letter for a Pakistani university student.
Student Profile:
- Name: {profile.get('name', 'Student')}
- Degree: {profile.get('degree', '')} in {profile.get('program', '')}
- Semester: {profile.get('semester', '')}
- CGPA: {profile.get('cgpa', '')}
- Skills: {', '.join(profile.get('skills', []))}
- Past Experience: {profile.get('past_experience', 'None mentioned')}
- Location: {profile.get('location_preference', 'Pakistan')}
Opportunity:
- Title: {opportunity.get('subject', '')}
- Type: {opportunity.get('type', '')}
- Organization: {opportunity.get('sender_email', '')}
- Eligibility: {opportunity.get('eligibility', '')}
- Summary: {opportunity.get('summary', '')}
- Required Skills: {', '.join(opportunity.get('required_skills', []))}
Write a 3-paragraph cover letter:
Paragraph 1: Enthusiastic opening — express specific interest in this opportunity and introduce the student naturally.
Paragraph 2: Highlight 2-3 most relevant skills, CGPA achievement, and any past experience that directly matches this opportunity.
Paragraph 3: Explain why this opportunity fits the student's career goals + a confident, polite call to action.
Tone: Professional but warm. Sound like a real motivated student, not a template.
Length: Under 300 words.
Salutation: "Dear Selection Committee,"
Sign-off: "Sincerely, {profile.get('name', 'Student')}"
Do NOT include any placeholder text like [Your Name] or [Date]. Fill everything in."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()