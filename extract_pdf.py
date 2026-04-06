import pdfplumber

with pdfplumber.open('methodologies/methodology-sp-ss-index-series.pdf') as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            print(f'=== PAGE {i+1} ===')
            print(text)
