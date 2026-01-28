import requests
from typing import List, Dict, Any, Tuple

OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


def search_disease_by_name(disease_name: str, size: int = 4):
    """
    Search for diseases by name using the Open Targets Platform API
    and return the top results ranked by score.

    Args:
        disease_name (str): The disease keyword to search for.
        size (int): Number of top results to return.

    Returns:
        List of dicts with 'id', 'name', 'entity', and 'score'.
    """
    query = """
    query searchDisease($queryString: String!) {
      search(queryString: $queryString) {
        hits {
          id
          name
          entity
          score
        }
      }
    }
    """
    response = requests.post(
        OPEN_TARGETS_API,
        json={"query": query, "variables": {"queryString": disease_name, "size": size}},
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    hits = response.json()["data"]["search"]["hits"]

    # Keep only diseases and sort by score
    diseases = [r for r in hits if r["entity"] == "disease"]
    diseases.sort(key=lambda x: x["score"], reverse=True)

    return diseases


def get_targets_by_disease(
    efo_id: str, size: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve target genes associated with a disease by EFO ID.

    Args:
        efo_id: EFO identifier of the disease.
        size: Number of targets to retrieve.

    Returns:
        Tuple of disease name and list of target information.
    """
    query = """
    query diseaseTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        name
        associatedTargets(page: { index: 0, size: $size }) {
          rows {
            score
            target {
              id
              approvedSymbol
              approvedName
              biotype
              functionDescriptions
              tractability {
                label
                modality
                value
              }
              knownDrugs {
                count
                rows {
                  drugType
                  phase
                  mechanismOfAction
                }
              }
            }
          }
        }
      }
    }
    """
    response = requests.post(
        OPEN_TARGETS_API,
        json={"query": query, "variables": {"efoId": efo_id, "size": size}},
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    data = response.json()["data"]["disease"]
    return data["name"], data["associatedTargets"]["rows"]


def extract_top_targets_summary(efo_id: str, size: int = 4) -> List[Dict[str, Any]]:
    """
    Format and summarize information about disease targets.
    """
    disease_name, targets = get_targets_by_disease(efo_id=efo_id, size=size)
    summaries = []
    for t in targets:
        gene = t["target"]
        score = round(t["score"], 3)
        entry = {
            "symbol": gene["approvedSymbol"],
            "name": gene["approvedName"],
            "score": score,
            "functions": gene.get("functionDescriptions", []),
            "tractability": [
                l["label"] for l in gene.get("tractability", []) if l["value"]
            ],
            "known_drugs": {
                "count": gene["knownDrugs"]["count"],
                "max_phase": (
                    max(
                        (
                            d["phase"]
                            for d in gene["knownDrugs"]["rows"]
                            if d["phase"] is not None
                        ),
                        default="N/A",
                    )
                    if gene.get("knownDrugs")
                    else "N/A"
                ),
            },
        }
        summaries.append(entry)

    return summaries
