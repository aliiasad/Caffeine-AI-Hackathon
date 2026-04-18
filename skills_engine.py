PROGRAM_SKILLS = {
    "computer science": ["Python", "C++", "Data Structures", "Algorithms", "OOP", "Linux", "Git"],
    "cs": ["Python", "C++", "Data Structures", "Algorithms", "OOP", "Linux", "Git"],
    "software engineering": ["Python", "Java", "Git", "Agile", "OOP", "Web Development", "Testing"],
    "se": ["Python", "Java", "Git", "Agile", "OOP", "Web Development"],
    "electrical engineering": ["MATLAB", "Circuit Design", "Embedded Systems", "Signal Processing", "Arduino"],
    "ee": ["MATLAB", "Circuit Design", "Embedded Systems", "Signal Processing"],
    "data science": ["Python", "Machine Learning", "Statistics", "SQL", "Data Analysis", "Pandas", "NumPy"],
    "business": ["Excel", "Communication", "Management", "Marketing", "Finance", "PowerPoint"],
    "economics": ["Excel", "Statistics", "Research", "Data Analysis", "Finance", "Econometrics"],
    "mathematics": ["MATLAB", "Statistics", "LaTeX", "Research", "Python", "Proof Writing"],
    "mechanical engineering": ["AutoCAD", "SolidWorks", "MATLAB", "Thermodynamics", "Manufacturing"],
    "civil engineering": ["AutoCAD", "Structural Analysis", "Project Management", "Surveying"],
    "bioinformatics": ["Python", "R", "Statistics", "Biology", "Data Analysis", "Genomics"],
    "artificial intelligence": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "NLP", "Computer Vision"],
    "ai": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "NLP"],
    "cybersecurity": ["Networking", "Linux", "Python", "Ethical Hacking", "Cryptography"],
}

SEMESTER_SKILLS = {
    1: ["Communication", "Time Management", "Teamwork"],
    2: ["Communication", "Time Management", "Teamwork", "Research Basics"],
    3: ["Research", "Academic Writing", "Problem Solving"],
    4: ["Research", "Academic Writing", "Problem Solving", "Project Management"],
    5: ["Leadership", "Project Management", "Technical Writing", "Presentation"],
    6: ["Leadership", "Project Management", "Technical Writing", "Presentation", "Critical Thinking"],
    7: ["Advanced Research", "Industry Knowledge", "Mentoring", "Proposal Writing"],
    8: ["Advanced Research", "Industry Knowledge", "Mentoring", "Professional Networking", "Report Writing"],
}

def infer_skills(degree: str, program: str, semester: int) -> dict:
    suggested = []
    combined = f"{degree} {program}".lower()
    for key, skills in PROGRAM_SKILLS.items():
        if key in combined:
            suggested.extend(skills)
            break
    sem_skills = SEMESTER_SKILLS.get(int(semester) if semester else 1, [])
    suggested.extend(sem_skills)
    suggested = list(dict.fromkeys(suggested))
    return {
        "suggested_skills": suggested[:12],
        "reason": f"Based on your {program} program at semester {semester}"
    }