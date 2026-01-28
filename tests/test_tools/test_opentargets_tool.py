
from unittest.mock import Mock, patch
from utilities.opentargets_tool import search_disease_by_name



mock_response_data = {
        "data": {
            "search": {
                "hits": [
                    {"id": "EFO_0000311", "name": "Breast carcinoma", "entity": "disease", "score": 0.95},
                    {"id": "EFO_0000408", "name": "Diabetes mellitus", "entity": "disease", "score": 0.90},
                    {"id": "EFO_0000270", "name": "Hypertension", "entity": "disease", "score": 0.85},
                    {"id": "EFO_0000305", "name": "Asthma", "entity": "disease", "score": 0.80},
                ]
            }
        }
    }

@patch("utilities.opentargets_tool.requests.post")

def test_search_disease_by_name(mock_post):
    mock_post.return_value = Mock(status_code=200)
    mock_post.return_value.json.return_value = mock_response_data

    disease_name = "breast cancer"
    size = 4
    results = search_disease_by_name(disease_name, size)
    assert len(results) == size
    assert results[0]["name"] == "Breast carcinoma"
    assert results[1]["name"] == "Diabetes mellitus"
    assert results[2]["name"] == "Hypertension"
    assert results[3]["name"] == "Asthma"
    mock_post.assert_called_once()