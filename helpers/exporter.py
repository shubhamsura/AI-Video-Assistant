import json
from datetime import datetime


def export_markdown_report(result: dict) -> str:
    """Generate structured Markdown documentation report."""
    title = result.get("title", "Meeting Report")
    summary = result.get("summary", "")
    action_items = result.get("action_items", "")
    key_decisions = result.get("key_decisions", "")
    open_questions = result.get("open_questions", "")
    transcript = result.get("transcript", "")
    word_count = len(transcript.split()) if transcript else 0

    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    return f"""# 🎬 {title}

**Generated Date:** {timestamp}  
**Platform Engine:** AI Video Assistant  
**Transcript Statistics:** {word_count} total words  

---

## 📋 Executive Summary
{summary}

---

## ✅ Action Items & Task Assignments
{action_items}

---

## 🔑 Key Strategic Decisions
{key_decisions}

---

## ❓ Open Questions & Follow-ups
{open_questions}

---

## 📜 Complete Transcript
```text
{transcript}
```
"""


def export_json_report(result: dict) -> str:
    """Generate JSON report with structured meeting metrics."""
    transcript = result.get("transcript", "")
    word_count = len(transcript.split()) if transcript else 0

    export_payload = {
        "title": result.get("title", "Meeting Report"),
        "created_at": datetime.now().isoformat(),
        "engine": "AI Video Assistant",
        "word_count": word_count,
        "summary": result.get("summary", ""),
        "action_items": result.get("action_items", ""),
        "key_decisions": result.get("key_decisions", ""),
        "open_questions": result.get("open_questions", ""),
        "transcript": transcript,
    }
    return json.dumps(export_payload, indent=2)
