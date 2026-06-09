def format_qa(questions, answers):
    dialogue = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
    return dialogue
