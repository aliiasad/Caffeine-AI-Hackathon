from datetime import datetime, date

def score_opportunity(opp: dict, profile: dict) -> dict:
    # --- URGENCY SCORE (35%) ---
    urgency = 0
    heat = "green"
    urgency_label = "No deadline"
    urgency_color = "#27ae60"
    deadline_str = opp.get("deadline", "")
    days_left = None

    if deadline_str and deadline_str.lower() not in ["unknown", "not specified", "none", ""]:
        try:
            deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            days_left = (deadline_date - date.today()).days
            if days_left < 0:
                urgency = 100
                heat = "expired"
                urgency_label = f"Expired {abs(days_left)}d ago"
                urgency_color = "#7f8c8d"
            elif days_left == 0:
                urgency = 100
                heat = "red"
                urgency_label = "Closes TODAY"
                urgency_color = "#e74c3c"
            elif days_left <= 2:
                urgency = 95
                heat = "red"
                urgency_label = f"{days_left}d left — Act NOW"
                urgency_color = "#e74c3c"
            elif days_left <= 5:
                urgency = 80
                heat = "red"
                urgency_label = f"{days_left}d left — Apply today"
                urgency_color = "#e74c3c"
            elif days_left <= 7:
                urgency = 65
                heat = "yellow"
                urgency_label = f"{days_left}d left — This week"
                urgency_color = "#f39c12"
            elif days_left <= 14:
                urgency = 50
                heat = "yellow"
                urgency_label = f"{days_left}d left — Plan soon"
                urgency_color = "#f39c12"
            elif days_left <= 30:
                urgency = 30
                heat = "green"
                urgency_label = f"{days_left}d left — Can wait"
                urgency_color = "#27ae60"
            else:
                urgency = 15
                heat = "green"
                urgency_label = f"{days_left}d left — Plenty of time"
                urgency_color = "#27ae60"
        except:
            urgency = 30
            heat = "yellow"
            urgency_label = "Deadline unclear"
            urgency_color = "#f39c12"
    else:
        urgency = 20
        urgency_label = "No deadline found"
        urgency_color = "#27ae60"

    # --- PROFILE FIT SCORE (40%) ---
    fit = 0
    fit_reasons = []
    fit_gaps = []

    pref_types = [t.lower() for t in profile.get("preferred_types", [])]
    opp_type = opp.get("type", "").lower()
    if any(p in opp_type for p in pref_types) or any(opp_type in p for p in pref_types):
        fit += 25
        fit_reasons.append("Matches your preferred opportunity type")
    else:
        fit_gaps.append({"field": "opportunity type", "detail": f"You prefer {', '.join(pref_types)} but this is {opp_type}", "severity": "minor"})

    min_cgpa = opp.get("min_cgpa")
    student_cgpa = profile.get("cgpa")
    if min_cgpa and student_cgpa:
        try:
            diff = float(student_cgpa) - float(min_cgpa)
            if diff >= 0:
                fit += 20
                fit_reasons.append(f"CGPA {student_cgpa} meets requirement of {min_cgpa}")
            elif diff >= -0.3:
                fit += 5
                fit_gaps.append({"field": "CGPA", "detail": f"You need {abs(diff):.1f} more CGPA points (have {student_cgpa}, need {min_cgpa})", "severity": "minor"})
            else:
                fit_gaps.append({"field": "CGPA", "detail": f"Required {min_cgpa}, you have {student_cgpa} — gap of {abs(diff):.1f}", "severity": "blocker"})
        except:
            pass

    opp_skills = [s.lower().strip() for s in opp.get("required_skills", [])]
    student_skills = [s.lower().strip() for s in profile.get("skills", [])]
    matched_skills = [s for s in opp_skills if any(s in ss or ss in s for ss in student_skills)]
    missing_skills = [s for s in opp_skills if s not in matched_skills]
    if opp_skills:
        ratio = len(matched_skills) / len(opp_skills)
        fit += int(ratio * 20)
        if matched_skills:
            fit_reasons.append(f"Skills match: {', '.join(matched_skills)}")
        if missing_skills:
            fit_gaps.append({"field": "skills", "detail": f"Missing skills: {', '.join(missing_skills)}", "severity": "preparable"})

    is_funded = opp.get("is_funded", False)
    needs_funding = profile.get("financial_need", False)
    if is_funded and needs_funding:
        fit += 15
        fit_reasons.append("Fully/partially funded — matches your financial need")
    elif needs_funding and not is_funded:
        fit_gaps.append({"field": "funding", "detail": "No funding mentioned — you indicated financial need", "severity": "minor"})

    opp_location = opp.get("location", "").lower()
    pref_location = profile.get("location_preference", "").lower()
    if pref_location and (pref_location in opp_location or opp_location in ["remote", "online", "virtual"]):
        fit += 10
        fit_reasons.append(f"Location compatible: {opp_location}")
    elif opp_location in ["remote", "online", "virtual"]:
        fit += 5
        fit_reasons.append("Remote — accessible from anywhere")

    past_exp = profile.get("past_experience", "").lower()
    opp_exp = opp.get("experience_required", "").lower()
    if opp_exp and past_exp and any(word in past_exp for word in opp_exp.split()):
        fit += 10
        fit_reasons.append("Your past experience is relevant")

    fit = max(0, min(100, fit))

    gaps = fit_gaps[:]
    docs_required = opp.get("required_docs", [])
    if docs_required:
        gaps.append({"field": "documents", "detail": f"Required docs: {', '.join(docs_required)}", "severity": "preparable"})

    # --- CREDIBILITY SCORE (15%) ---
    credibility = 50
    sender = opp.get("sender_email", "").lower()
    if any(d in sender for d in [".edu", ".gov", ".org", ".ac", ".pk"]):
        credibility = 90
    elif "@gmail" in sender or "@yahoo" in sender or "@hotmail" in sender:
        credibility = 35
    if opp.get("application_link"):
        credibility = min(100, credibility + 10)
    if not opp.get("application_link") and not opp.get("contact_info"):
        credibility = max(0, credibility - 20)

    # --- COMPLETENESS SCORE (10%) ---
    fields = ["type", "deadline", "eligibility", "required_docs", "application_link"]
    filled = sum(1 for f in fields if opp.get(f) and str(opp.get(f)).lower() not in ["unknown", "none", "not specified", "", "[]"])
    completeness = int((filled / len(fields)) * 100)

    total = int((urgency * 0.35) + (fit * 0.40) + (credibility * 0.15) + (completeness * 0.10))

    opp.update({
        "urgency_score": urgency,
        "fit_score": fit,
        "fit_reasons": fit_reasons,
        "credibility_score": credibility,
        "completeness_score": completeness,
        "total_score": total,
        "gaps": gaps,
        "urgency_label": urgency_label,
        "urgency_color": urgency_color,
        "heat": heat,
        "days_left": days_left
    })
    return opp