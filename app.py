import os
import json
from flask import Flask, request, jsonify, render_template
from groq import Groq
from dotenv import load_dotenv
from scorer import score_opportunity
from skills_engine import infer_skills
from cover_letter import generate_cover_letter

load_dotenv()
app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_content(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096
    )
    return response.choices[0].message.content

EXTRACTION_PROMPT = """You are an expert email analyst for university students in Pakistan.
Analyze each email in the batch below and return a JSON array.

For each email return an object with these exact keys:
- "subject": string
- "sender_email": string (or "unknown")
- "is_opportunity": boolean
- "rejection_reason": string (if not opportunity, else "")
- "type": one of: "scholarship","internship","fellowship","competition","admission","research","job","other","not_opportunity"
- "deadline": string YYYY-MM-DD format if found, else "unknown"
- "eligibility": string
- "min_cgpa": number or null
- "required_skills": array of strings
- "required_docs": array of strings
- "application_link": string (or "")
- "contact_info": string (or "")
- "is_funded": boolean
- "location": string (or "unknown")
- "experience_required": string (or "")
- "evidence_quote": string (one key phrase max 20 words)
- "summary": string (2 sentences)
- "similar_opportunity_hint": string (if expired or unlikely to qualify, suggest next steps, else "")
- "readiness_tip": string (one practical sentence for right now, else "")

Return ONLY a raw JSON array. No markdown. No explanation. Start with [ end with ]."""


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html", active_page="home")

@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html", active_page="analyze")

@app.route("/results")
def results_page():
    return render_template("results.html", active_page="results")


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/infer_skills", methods=["POST"])
def infer_skills_route():
    data = request.json
    result = infer_skills(
        data.get("degree", ""),
        data.get("program", ""),
        data.get("semester", 1)
    )
    return jsonify(result)


@app.route("/cover_letter", methods=["POST"])
def cover_letter_route():
    data = request.json
    opp = data.get("opportunity", {})
    profile = data.get("profile", {})
    try:
        letter = generate_cover_letter(opp, profile)
        return jsonify({"cover_letter": letter})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    emails_text = data.get("emails", "")
    profile = data.get("profile", {})

    if not emails_text.strip():
        return jsonify({"error": "No emails provided"}), 400

    try:
        full_prompt = EXTRACTION_PROMPT + f"\n\nEmails to analyze:\n\n{emails_text}"
        raw = generate_content(full_prompt).strip()

        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    raw = part
                    break

        if not raw.startswith("["):
            start = raw.find("[")
            if start != -1:
                raw = raw[start:]

        opportunities = json.loads(raw)
        scored = []
        rejected = []
        expired_with_hints = []

        for opp in opportunities:
            if opp.get("is_opportunity"):
                opp = score_opportunity(opp, profile)
                if opp.get("heat") == "expired":
                    expired_with_hints.append(opp)
                else:
                    scored.append(opp)
            else:
                rejected.append(opp)

        scored.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        for i, opp in enumerate(scored):
            opp["rank"] = i + 1

        # WOW #1 — One Task Today
        one_task_today = None
        if scored:
            top = scored[0]
            one_task_today = {
                "subject": top.get("subject", ""),
                "urgency_label": top.get("urgency_label", ""),
                "urgency_color": top.get("urgency_color", "#27ae60"),
                "days_left": top.get("days_left"),
                "heat": top.get("heat", "green"),
                "application_link": top.get("application_link", ""),
                "fit_score": top.get("fit_score", 0),
                "total_score": top.get("total_score", 0),
                "readiness_tip": top.get("readiness_tip", ""),
                "next_step": (
                    f"Apply now at: {top.get('application_link')}" if top.get("application_link")
                    else f"Contact: {top.get('contact_info')}" if top.get("contact_info")
                    else "Prepare your documents and find the application link"
                ),
                "required_docs": top.get("required_docs", []),
                "rank": 1
            }

        # WOW #3 — If Not This Then What
        for opp in scored:
            if opp.get("fit_score", 0) < 40 and len(scored) > 1:
                alternates = [o for o in scored if o.get("rank") != opp.get("rank") and o.get("fit_score", 0) > opp.get("fit_score", 0)]
                if alternates:
                    best = max(alternates, key=lambda x: x.get("fit_score", 0))
                    opp["if_not_this"] = {
                        "subject": best.get("subject", ""),
                        "rank": best.get("rank"),
                        "fit_score": best.get("fit_score")
                    }

        return jsonify({
            "opportunities": scored,
            "rejected": rejected,
            "expired": expired_with_hints,
            "one_task_today": one_task_today,
            "total_emails": len(opportunities),
            "opportunities_found": len(scored),
            "rejected_count": len(rejected),
            "expired_count": len(expired_with_hints)
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI parse error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)