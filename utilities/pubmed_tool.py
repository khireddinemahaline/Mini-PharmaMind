from typing import List, Dict, Any
from Bio import Entrez, Medline

Entrez.email = "khirocyber@gmail.com"  # Always use your academic email


def search_pubmed(query: str, max_results: int = 5) -> List[str]:
    """
    Search PubMed for article IDs matching a query.

    Args:
        query: Search term.
        max_results: Max number of article IDs to return.

    Returns:
        List of PubMed IDs (PMIDs).
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_full_articles(id_list: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch full article details (title, abstract, authors, etc.).

    Args:
        id_list: List of PubMed IDs.

    Returns:
        List of dictionaries with article info.
    """
    if not id_list:
        return []

    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = Medline.parse(handle)

    articles = []
    for record in records:
        articles.append(
            {
                "pmid": record.get("PMID", ""),
                "title": record.get("TI", ""),
                "abstract": record.get("AB", ""),
                "authors": record.get("AU", []),
                "journal": record.get("JT", ""),
                "year": record.get("DP", "")[:4],
            }
        )
    handle.close()
    return articles


def search_and_fetch_articles(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search PubMed and fetch full article details (abstracts).

    Args:
        query: Search query.
        max_results: Max number of articles to retrieve.

    Returns:
        List of article dictionaries.
    """
    ids = search_pubmed(query, max_results)
    return fetch_full_articles(ids)
