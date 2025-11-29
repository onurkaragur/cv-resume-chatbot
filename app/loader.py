from docx import Document

def load_csv_text(path="data/replace-with-your-cv.docx"):
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text