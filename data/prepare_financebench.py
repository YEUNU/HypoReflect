import json
import requests
import os
import re
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# FinanceBench GitHub raw URLs
FINANCEBENCH_QUESTIONS_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
FINANCEBENCH_DOCS_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_document_information.jsonl"


def sanitize_filename(name: str) -> str:
    """파일명으로 사용할 수 있게 정규화."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def download_financebench():
    """FinanceBench JSONL 파일들을 다운로드."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    questions_path = data_dir / "financebench_open_source.jsonl"
    docs_path = data_dir / "financebench_document_information.jsonl"
    
    # Questions 다운로드
    if not questions_path.exists():
        print(f"Downloading FinanceBench questions from {FINANCEBENCH_QUESTIONS_URL}...")
        resp = requests.get(FINANCEBENCH_QUESTIONS_URL)
        resp.raise_for_status()
        questions_path.write_text(resp.text, encoding="utf-8")
        print(f"Downloaded: {questions_path}")
    else:
        print(f"Questions already exist: {questions_path}")
    
    # Document metadata 다운로드
    if not docs_path.exists():
        print(f"Downloading document metadata from {FINANCEBENCH_DOCS_URL}...")
        resp = requests.get(FINANCEBENCH_DOCS_URL)
        resp.raise_for_status()
        docs_path.write_text(resp.text, encoding="utf-8")
        print(f"Downloaded: {docs_path}")
    else:
        print(f"Document metadata already exist: {docs_path}")
    
    return questions_path, docs_path


def process_financebench(questions_path: Path, docs_path: Path, build_corpus: bool = False):
    """
    FinanceBench 데이터를 현재 시스템 포맷으로 변환.
    
    1. evidence_text_full_page를 코퍼스 파일로 저장
    2. 쿼리 JSON 생성
    """
    # 1. JSONL 파일 로드
    questions = []
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    doc_info = {}
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_info[doc["doc_name"]] = doc
    
    print(f"Loaded {len(questions)} questions, {len(doc_info)} documents")
    
    # 2. 코퍼스 생성(옵션) - evidence_text_full_page를 파일로 저장
    corpus_dir = Path("data/finance_corpus")
    if build_corpus:
        if corpus_dir.exists():
            import shutil
            shutil.rmtree(corpus_dir)
        corpus_dir.mkdir(parents=True, exist_ok=True)
    
    saved_pages = set()  # (doc_name, page_num) 중복 방지
    
    for q in questions:
        doc_name = q.get("doc_name", "")
        evidences = q.get("evidence", [])
        
        for ev in evidences:
            ev_doc = ev.get("evidence_doc_name", ev.get("doc_name", doc_name))
            page_num = ev.get("evidence_page_num", 0)
            full_page = ev.get("evidence_text_full_page", "")
            
            if not full_page:
                continue
            
            page_key = (ev_doc, page_num)
            if page_key in saved_pages:
                continue
            saved_pages.add(page_key)
            
            if build_corpus:
                # 파일명: {doc_name}_page_{page_num}.txt
                filename = f"{sanitize_filename(ev_doc)}_page_{page_num:03d}.txt"
                file_path = corpus_dir / filename
                
                # 메타데이터 포함하여 저장
                content = f"Document: {ev_doc}\nPage: {page_num}\n\n{full_page}"
                file_path.write_text(content, encoding="utf-8")
    
    if build_corpus:
        print(f"Created {len(saved_pages)} corpus files in {corpus_dir}")
    else:
        print(f"Corpus generation disabled. Collected metadata for {len(saved_pages)} evidence pages.")
    
    # 3. 쿼리 파일 생성
    queries = []
    for q in questions:
        # evidence에서 첫 번째 항목 사용
        evidences = q.get("evidence", [])
        first_ev = evidences[0] if evidences else {}
        
        query_item = {
            "_id": q.get("financebench_id", ""),
            "query": q.get("question", ""),
            "ground_truth": q.get("answer", ""),
            "justification": q.get("justification", ""),
            # FinanceBench 특화 필드
            "evidence_doc": first_ev.get("evidence_doc_name", q.get("doc_name", "")),
            "evidence_page": first_ev.get("evidence_page_num", 0),
            "evidence_text": first_ev.get("evidence_text", ""),
            # 메타데이터
            "company": q.get("company", ""),
            "question_type": q.get("question_type", ""),
            "question_reasoning": q.get("question_reasoning", ""),
            "dataset": "financebench"  # 데이터셋 식별자
        }
        queries.append(query_item)
    
    queries_path = Path("data/financebench_queries.json")
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(queries)} queries in {queries_path}")
    
    # 4. 통계 출력
    print("\n=== FinanceBench Statistics ===")
    print(f"Total questions: {len(queries)}")
    print(f"Total evidence pages: {len(saved_pages)}")
    if build_corpus:
        print(f"Total corpus files created: {len(saved_pages)}")
    else:
        print("Total corpus files created: 0 (build disabled)")
    
    # 질문 유형별 통계
    type_counts = {}
    for q in queries:
        qt = q.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    print("\nQuestion types:")
    for qt, count in sorted(type_counts.items()):
        print(f"  - {qt}: {count}")
    
    # 회사별 통계
    company_counts = {}
    for q in queries:
        c = q.get("company", "unknown")
        company_counts[c] = company_counts.get(c, 0) + 1
    print(f"\nUnique companies: {len(company_counts)}")
    
    return doc_info  # PDF 다운로드에 사용


EDGAR_HEADERS = {
    "User-Agent": "research@example.com",
    "Accept": "text/html,application/xhtml+xml,application/pdf",
}

# Adobe의 pdf-page.html 래퍼 URL 패턴
_ADOBE_WRAPPER_RE = re.compile(r"adobe\.com/pdf-page\.html\?pdfTarget=(.+)$")

# SEC EDGAR 회사 CIK 매핑 (doc_link로 찾기 어려운 경우 사용)
_COMPANY_CIK = {
    "johnsonandjohnson": "200406",
    "kraftheinz": "1637459",
    "amd": "2488",
}


def _decode_adobe_url(doc_link: str) -> str | None:
    """Adobe pdf-page.html 래퍼에서 실제 PDF URL을 추출."""
    m = _ADOBE_WRAPPER_RE.search(doc_link)
    if not m:
        return None
    try:
        return base64.b64decode(m.group(1)).decode()
    except Exception:
        return None


def _get_edgar_cik_from_link(doc_link: str) -> str | None:
    """doc_link URL에서 SEC EDGAR CIK를 추출."""
    # https://www.sec.gov/Archives/edgar/data/{CIK}/...
    m = re.search(r"sec\.gov/Archives/edgar/data/(\d+)/", doc_link)
    if m:
        return m.group(1)
    # IR 사이트 URL에서 회사명으로 추정
    for keyword, cik in _COMPANY_CIK.items():
        if keyword in doc_link.lower():
            return cik
    return None


def _find_edgar_pdf(cik: str, doc_link: str) -> str | None:
    """
    SEC EDGAR에서 해당 doc_link와 가장 가까운 10-K/8-K의 PDF URL을 탐색.
    accession number를 doc_link에서 직접 파싱하거나, 없으면 None 반환.
    """
    # accession number 직접 파싱: .../edgar/data/{CIK}/{acc_nodash}/...
    m = re.search(r"edgar/data/\d+/(\d{18})/", doc_link)
    if not m:
        return None
    acc_nodash = m.group(1)
    acc = f"{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}"
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
    try:
        r = requests.get(idx_url, headers=EDGAR_HEADERS, timeout=15)
        pdfs = re.findall(r'href="(/Archives/edgar/data/[^"]+\.pdf)"', r.text, re.IGNORECASE)
        if pdfs:
            return "https://www.sec.gov" + pdfs[0]
    except Exception:
        pass
    return None


def _find_edgar_main_htm(cik: str, acc_nodash: str) -> str | None:
    """EDGAR 파일 인덱스에서 메인 10-K/8-K HTML 문서 URL을 반환."""
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
    try:
        r = requests.get(idx_url, headers=EDGAR_HEADERS, timeout=15)
        hrefs = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', r.text, re.IGNORECASE)
        # exhibit, R*.htm 제외하고 메인 문서만
        mains = [h for h in hrefs if not re.search(r"/ex\d|/R\d+\.htm$", h, re.IGNORECASE)]
        if mains:
            return "https://www.sec.gov" + mains[0]
    except Exception:
        pass
    return None


def _html_url_to_pdf(html_url: str, pdf_path: Path) -> bool:
    """HTML URL을 weasyprint로 PDF 변환. 성공 시 True."""
    try:
        from weasyprint import HTML
        r = requests.get(html_url, headers=EDGAR_HEADERS, timeout=60)
        base_url = html_url.rsplit("/", 1)[0] + "/"
        HTML(string=r.text, base_url=base_url).write_pdf(str(pdf_path))
        return pdf_path.exists() and pdf_path.stat().st_size > 1024
    except ImportError:
        print("    [WARN] weasyprint not installed. pip install weasyprint")
        return False
    except Exception as e:
        print(f"    [WARN] HTML→PDF failed: {e}")
        return False


def download_single_pdf(doc, pdf_dir):
    """
    단일 PDF 파일을 다운로드하는 헬퍼 함수.

    전략:
    1. doc_link에서 직접 PDF 시도
    2. Adobe 래퍼면 base64 디코딩 후 직접 URL 시도 (SEC EDGAR unofficial PDF 포함)
    3. EDGAR 파일 인덱스에서 PDF 탐색
    4. PDF 없으면 메인 HTML 문서를 weasyprint로 변환
    """
    doc_name = doc.get("doc_name", "unknown")
    doc_link = doc.get("doc_link", "")

    if not doc_link:
        return "skipped_no_link", doc_name

    pdf_path = pdf_dir / f"{sanitize_filename(doc_name)}.pdf"

    if pdf_path.exists():
        return "skipped_exists", doc_name

    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    def _try_download(url: str) -> bool:
        try:
            resp = requests.get(url, headers=browser_headers, timeout=60, allow_redirects=True)
            if resp.headers.get("content-type", "").startswith("application/pdf") or \
               resp.content[:4] == b"%PDF":
                pdf_path.write_bytes(resp.content)
                return True
        except Exception:
            pass
        return False

    # 1. 직접 다운로드 시도
    if _try_download(doc_link):
        return "downloaded", doc_name

    # 2. Adobe 래퍼 처리: base64 디코딩 후 실제 URL 시도
    adobe_real_url = _decode_adobe_url(doc_link)
    if adobe_real_url:
        if _try_download(adobe_real_url):
            return "downloaded_adobe_decoded", doc_name
        # Adobe URL이 SEC EDGAR를 가리키면 EDGAR에서 unofficial PDF 탐색
        cik = _get_edgar_cik_from_link(adobe_real_url)
        if cik:
            edgar_pdf = _find_edgar_pdf(cik, adobe_real_url)
            if edgar_pdf and _try_download(edgar_pdf):
                return "downloaded_edgar_pdf", doc_name

    # 3. doc_link 자체가 EDGAR URL이면 해당 인덱스에서 PDF 탐색
    cik = _get_edgar_cik_from_link(doc_link)
    if cik:
        edgar_pdf = _find_edgar_pdf(cik, doc_link)
        if edgar_pdf and _try_download(edgar_pdf):
            return "downloaded_edgar_pdf", doc_name

    # 4. CIK를 알면 EDGAR 메인 HTML → weasyprint PDF 변환
    # doc_link에서 accession number 추출 시도
    m = re.search(r"edgar/data/\d+/(\d{18})/", doc_link)
    if m and cik:
        acc_nodash = m.group(1)
        html_url = _find_edgar_main_htm(cik, acc_nodash)
        if html_url and _html_url_to_pdf(html_url, pdf_path):
            return "converted_html_to_pdf", doc_name

    # 5. static-files URL (J&J, KraftHeinz IR 사이트) - CIK 매핑으로 EDGAR 탐색
    if not cik:
        for keyword, mapped_cik in _COMPANY_CIK.items():
            if keyword in doc_link.lower():
                cik = mapped_cik
                break

    return "skipped_not_pdf", doc_name


def download_pdfs(doc_info: dict, limit: int = None, max_workers: int = 10):
    """
    FinanceBench PDF 파일들을 병렬로 다운로드.
    """
    pdf_dir = Path("data/finance_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    docs_to_download = list(doc_info.values())
    if limit:
        docs_to_download = docs_to_download[:limit]
    
    total = len(docs_to_download)
    print(f"\n=== Downloading {total} PDFs (Parallel, workers={max_workers}) ===")
    
    results = {"downloaded": 0, "skipped_exists": 0, "skipped_no_link": 0, "skipped_not_pdf": 0, "failed": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(download_single_pdf, doc, pdf_dir): doc for doc in docs_to_download}
        
        count = 0
        for future in as_completed(future_to_doc):
            status, doc_name = future.result()
            count += 1
            
            if status == "downloaded":
                results["downloaded"] += 1
                msg = f"[OK] {doc_name}"
            elif status == "skipped_exists":
                results["skipped_exists"] += 1
                msg = f"[EXISTS] {doc_name}"
            elif status.startswith("failed"):
                results["failed"] += 1
                msg = f"[{status.upper()}] {doc_name}"
            else:
                results[status] += 1
                msg = f"[{status.upper()}] {doc_name}"
            
            print(f"  ({count}/{total}) {msg}")
    
    print(f"\n=== PDF Download Summary ===")
    print(f"Downloaded: {results['downloaded']}")
    print(f"Skipped: {results['skipped_exists'] + results['skipped_no_link'] + results['skipped_not_pdf']}")
    print(f"Failed: {results['failed']}")
    print(f"PDF directory: {pdf_dir.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinanceBench 데이터셋 준비")
    parser.add_argument("--download-pdfs", action="store_true", 
                        help="원본 PDF 파일 다운로드 (OCR용)")
    parser.add_argument("--pdf-limit", type=int, default=None,
                        help="다운로드할 PDF 최대 수 (테스트용)")
    parser.add_argument("--build-corpus", action="store_true",
                        help="evidence_text_full_page 기반 data/finance_corpus 생성 (기본값: 비활성)")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="코퍼스 생성 건너뛰기 (PDF만 다운로드)")
    args = parser.parse_args()

    if args.build_corpus and args.skip_corpus:
        parser.error("--build-corpus and --skip-corpus cannot be used together.")
    
    # 1. 메타데이터 다운로드
    questions_path, docs_path = download_financebench()
    
    # 2. 쿼리 생성 + (옵션) 코퍼스 생성
    doc_info = process_financebench(
        questions_path,
        docs_path,
        build_corpus=(args.build_corpus and not args.skip_corpus),
    )
    
    # 3. PDF 다운로드 (선택)
    if args.download_pdfs:
        download_pdfs(doc_info, limit=args.pdf_limit)
    
    print("\nFinanceBench data preparation complete!")
