
import pypdf

def load_pdf(file_path):
    with open(file_path, 'rb') as f:
        t=""
        pdf_reader = pypdf.PdfReader(f)
        num_pages = pdf_reader._get_num_pages()
        # You can access each page like this:
        for page_num in range(num_pages):
            page = pdf_reader._get_page(page_num)
            # Do something with the page
            text = page.extract_text()
            t=t+text.strip()
    return t
    