from pypdf import PdfReader

def extract_pdf_text(uploaded_files):
    documents = []
    for file in uploaded_files:
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    {
                        "text": text,
                        "page": page_num + 1,
                        "source": file.name
                    }
                )
    return documents