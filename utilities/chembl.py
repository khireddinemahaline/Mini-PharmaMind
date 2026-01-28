"""
ChEMBL Database Tools for Pharmaceutical Research
Provides direct access to ChEMBL API for drug discovery and bioactivity queries.
"""

import requests
from typing import Optional, List, Dict, Any
import json
from datetime import datetime



class ChEMBLClient:
    """Client for interacting with ChEMBL API"""
    
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    TIMEOUT = 30
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChEMBL-Python-Client/1.0.0',
            'Accept': 'application/json',
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to ChEMBL API
        
        Args:
            endpoint: API endpoint path (e.g., '/molecule/search.json')
            params: Query parameters
        
        Returns:
            JSON response as dictionary
        
        Raises:
            requests.RequestException: If API request fails
        """
        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = self.session.get(url, params=params, timeout=self.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"ChEMBL API Error: {str(e)}")


# Initialize global client
_chembl_client = ChEMBLClient()


# ============================================================================
# CORE CHEMICAL SEARCH & RETRIEVAL TOOLS (3 tools)
# ============================================================================
def search_compounds(
    query: str,
    limit: int = 25,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Search the ChEMBL database for pharmaceutical compounds.
    
    This is the FIRST STEP when searching for compounds. Returns ChEMBL IDs 
    which are required for other detailed queries.
    
    Args:
        query (str): Search term - drug name (e.g., "Aspirin"), brand name (e.g., "Lipitor"), 
                     chemical name (e.g., "acetylsalicylic acid"), or partial ChEMBL ID.
                     Case-insensitive, supports wildcards and partial matches.
        limit (int): Maximum number of results to return. Range: 1-1000, Default: 25.
                     Use 5-10 for focused searches, 20-50 for exploratory, 100+ for comprehensive.
        offset (int): Number of results to skip for pagination. Default: 0.
                      Example: limit=25, offset=25 gets results 26-50.
    
    Returns:
        Dict containing search results with ChEMBL IDs and basic compound info
    
    Example:
        >>> results = search_compounds("aspirin", limit=10)
        >>> for compound in results['molecules']:
        ...     print(compound['molecule_chembl_id'], compound['pref_name'])
    """
    if not isinstance(query, str) or len(query) == 0:
        raise ValueError("Query must be a non-empty string")
    if not (1 <= limit <= 1000):
        raise ValueError("Limit must be between 1 and 1000")
    if not (offset >= 0):
        raise ValueError("Offset must be >= 0")
    
    params = {
        'q': query,
        'limit': limit,
        'offset': offset,
    }
    
    response = _chembl_client._make_request('/molecule/search.json', params=params)
    return {
        'query': query,
        'limit': limit,
        'offset': offset,
        'result_count': response.get('page_meta', {}).get('total_count', 0),
        'molecules': response.get('molecules', []),
        'timestamp': datetime.now().isoformat()
    }

def get_compound_info(chembl_id: str) -> Dict[str, Any]:
    """
    Retrieve comprehensive detailed information for a SPECIFIC compound.
    
    IMPORTANT: You must have a valid ChEMBL ID from search_compounds before using this.
    
    Returns:
        - Molecular properties (molecular weight, LogP, H-bond donors/acceptors)
        - Drug classifications and approval status
        - Therapeutic indications and mechanism of action
        - Trade names and cross-references (PubChem, DrugBank)
    
    Args:
        chembl_id (str): ChEMBL compound identifier in format CHEMBLxxxx
                         Examples: "CHEMBL59" (aspirin), "CHEMBL25" (atorvastatin), "CHEMBL192" (caffeine)
                         Must be exact match, case-sensitive.
    
    Returns:
        Dict containing complete compound information including properties, 
        drug data, and cross-references
    
    Example:
        >>> compound = get_compound_info("CHEMBL59")
        >>> print(f"Name: {compound['pref_name']}")
        >>> print(f"Molecular Weight: {compound['molecule_properties']['mw_freebase']}")
        >>> print(f"Drug Type: {compound.get('therapeutic_flag', False)}")
    """
    if not isinstance(chembl_id, str) or len(chembl_id) == 0:
        raise ValueError("ChEMBL ID must be a non-empty string")
    
    response = _chembl_client._make_request(f'/molecule/{chembl_id}.json')
    return {
        'chembl_id': response.get('molecule_chembl_id'),
        'pref_name': response.get('pref_name'),
        'molecular_properties': response.get('molecule_properties', {}),
        'drug_data': {
            'therapeutic_flag': response.get('therapeutic_flag'),
            'drug_type': response.get('drug_type'),
            'usan_stem': response.get('usan_stem'),
            'first_approval': response.get('first_approval'),
            'withdrawn_flag': response.get('withdrawn_flag'),
        },
        'cross_references': {
            'pubchem_sid': response.get('pubchem_sid'),
            'pubchem_cid': response.get('pubchem_cid'),
            'external_ids': response.get('external_ids', []),
        },
        'synonyms': response.get('molecule_synonyms', []),
        'indication': response.get('indication_class'),
        'timestamp': datetime.now().isoformat()
    }

def get_compound_structure(
    chembl_id: str,
    format: str = 'smiles'
) -> Dict[str, Any]:
    """
    Retrieve chemical structure representations in standard computational chemistry formats.
    
    Essential for: structure-based drug design, similarity/substructure searches, 
    molecular docking, visualization, and computational analysis.
    
    Args:
        chembl_id (str): ChEMBL compound ID (e.g., "CHEMBL59"). Must be valid and exist in database.
        format (str): Chemical structure format to retrieve. Options:
            - "smiles" (default): Simplified Molecular Input Line Entry System - compact text format, 
                                  best for similarity searches and quick viewing
            - "inchi": International Chemical Identifier - canonical format, 
                       best for exact structure matching and database lookups
            - "molfile": MDL Molfile format - includes 2D coordinates, good for visualization software
            - "sdf": Structure Data File - includes properties and 3D coordinates, 
                     best for computational chemistry tools
    
    Returns:
        Dict containing structure data in requested format plus canonical representations
    
    Example:
        >>> structure = get_compound_structure("CHEMBL59", format="smiles")
        >>> print(f"SMILES: {structure['smiles']}")
        >>> print(f"InChI: {structure['inchi']}")
    """
    if not isinstance(chembl_id, str) or len(chembl_id) == 0:
        raise ValueError("ChEMBL ID must be a non-empty string")
    
    valid_formats = ['smiles', 'inchi', 'molfile', 'sdf']
    if format not in valid_formats:
        raise ValueError(f"Format must be one of {valid_formats}")
    
    response = _chembl_client._make_request(f'/molecule/{chembl_id}.json')
    
    structures = response.get('molecule_structures', {})
    
    return {
        'chembl_id': response.get('molecule_chembl_id'),
        'pref_name': response.get('pref_name'),
        'structures': {
            'canonical_smiles': structures.get('canonical_smiles'),
            'standard_inchi': structures.get('standard_inchi'),
            'standard_inchi_key': structures.get('standard_inchi_key'),
            'molfile_2d': structures.get('molfile'),
            'molfile_3d': response.get('3d_structure'),
        },
        'requested_format': format,
        'molecular_formula': response.get('molecule_properties', {}).get('ro5_violations'),
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# BIOACTIVITY & ASSAY DATA TOOLS (2 tools)
# ============================================================================
def search_activities(
    target_chembl_id: Optional[str] = None,
    assay_chembl_id: Optional[str] = None,
    molecule_chembl_id: Optional[str] = None,
    activity_type: Optional[str] = None,
    limit: int = 25
) -> Dict[str, Any]:
    """
    Search bioactivity data from high-throughput screening and binding assays.
    
    Returns experimental measurements including IC50, Ki, EC50, Kd with units, 
    standard deviations, and assay details.
    
    CRITICAL for:
    - Finding potency data for compounds
    - Identifying all compounds tested against a target protein
    - Comparing compound activities
    
    NOTE: At least ONE filter parameter must be provided.
    
    Args:
        target_chembl_id (str, optional): Filter by biological target ChEMBL ID
            Example: "CHEMBL1824" (beta-2 adrenergic receptor), 
                     "CHEMBL1862" (EGFR)
            Use when searching for drug candidates for a specific protein/enzyme/receptor.
        
        assay_chembl_id (str, optional): Filter by specific assay ChEMBL ID
            Example: "CHEMBL1217643"
            Returns all activity measurements from this experimental protocol.
            Use when comparing compounds tested under identical conditions.
        
        molecule_chembl_id (str, optional): Filter by compound ChEMBL ID
            Example: "CHEMBL59"
            Returns all bioactivity measurements for this compound across all targets/assays.
            Use to profile a compound's complete activity spectrum and selectivity.
        
        activity_type (str, optional): Filter by measurement type. Common values (case-sensitive):
            - "IC50": Half-maximal inhibitory concentration (lower = more potent inhibitor)
            - "Ki": Inhibition constant (binding affinity)
            - "EC50": Half-maximal effective concentration (agonist potency)
            - "Kd": Dissociation constant (binding affinity)
            - "GI50": 50% growth inhibition (cytotoxicity)
            - "MIC": Minimum inhibitory concentration (antimicrobials)
        
        limit (int): Maximum results to return. Range: 1-1000, Default: 25.
                     Note: Bioactivity searches can return thousands of results.
                     Start with 25-50 for initial exploration, use higher with specific filters.
    
    Returns:
        Dict containing bioactivity measurements with complete metadata
    
    Example:
        >>> activities = search_activities(target_chembl_id="CHEMBL1862", activity_type="IC50", limit=50)
        >>> for act in activities['activities']:
        ...     print(f"Compound: {act['molecule_chembl_id']}, IC50: {act['standard_value']} {act['standard_units']}")
    """
    # Validate at least one filter is provided
    if not any([target_chembl_id, assay_chembl_id, molecule_chembl_id, activity_type]):
        raise ValueError("At least one filter parameter must be provided (target, assay, molecule, or activity_type)")
    
    if not (1 <= limit <= 1000):
        raise ValueError("Limit must be between 1 and 1000")
    
    params = {'limit': limit}
    
    if target_chembl_id:
        params['target_chembl_id'] = target_chembl_id
    if assay_chembl_id:
        params['assay_chembl_id'] = assay_chembl_id
    if molecule_chembl_id:
        params['molecule_chembl_id'] = molecule_chembl_id
    if activity_type:
        params['standard_type'] = activity_type
    
    response = _chembl_client._make_request('/activity.json', params=params)
    
    return {
        'filters': {
            'target_chembl_id': target_chembl_id,
            'assay_chembl_id': assay_chembl_id,
            'molecule_chembl_id': molecule_chembl_id,
            'activity_type': activity_type,
        },
        'limit': limit,
        'result_count': response.get('page_meta', {}).get('total_count', 0),
        'activities': response.get('activities', []),
        'timestamp': datetime.now().isoformat()
    }

def get_assay_info(chembl_id: str) -> Dict[str, Any]:
    """
    Retrieve complete information about a specific biological assay.
    
    Returns detailed experimental protocol including:
    - Assay type (binding/functional/ADME/toxicity)
    - Target organism and species
    - Cell line and tissue type used
    - Subcellular fraction
    - Detection method and measurement endpoints
    - Data validity comments and confidence scores
    - Literature references (PubMed IDs)
    
    Use this to:
    - Understand the experimental context and quality of bioactivity data
    - Assess data reliability
    - Compare assay protocols
    
    Args:
        chembl_id (str): ChEMBL assay identifier in format CHEMBLxxxxxxx
                         (typically 7-10 digits)
                         Examples: "CHEMBL1217643", "CHEMBL829152"
                         Obtain from search_activities results.
    
    Returns:
        Dict containing complete assay metadata and experimental details
    
    Example:
        >>> assay = get_assay_info("CHEMBL1217643")
        >>> print(f"Assay Type: {assay['assay_type']}")
        >>> print(f"Target: {assay['assay_organism']}")
        >>> print(f"Detection Method: {assay['detection_method']}")
    """
    if not isinstance(chembl_id, str) or len(chembl_id) == 0:
        raise ValueError("ChEMBL ID must be a non-empty string")
    
    response = _chembl_client._make_request(f'/assay/{chembl_id}.json')
    
    return {
        'chembl_id': response.get('chembl_id'),
        'description': response.get('description'),
        'assay_type': response.get('assay_type'),
        'experimental_details': {
            'assay_organism': response.get('assay_organism'),
            'assay_tissue': response.get('assay_tissue'),
            'assay_cell_type': response.get('assay_cell_type'),
            'assay_subcellular_fraction': response.get('assay_subcellular_fraction'),
            'detection_method': response.get('detection_method'),
            'detection_method_comment': response.get('detection_method_comment'),
        },
        'data_quality': {
            'confidence_score': response.get('confidence_score'),
            'data_validity_comment': response.get('data_validity_comment'),
            'chi_test_result': response.get(' chi_test_result'),
        },
        'literature': {
            'pubmed_ids': response.get('doc_id'),
            'journal': response.get('journal'),
        },
        'target_info': {
            'target_chembl_id': response.get('target_chembl_id'),
            'target_type': response.get('target_type'),
        },
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# BATCH OPERATIONS TOOL (1 tool)
# ============================================================================
def batch_compound_lookup(chembl_ids: List[str]) -> Dict[str, Any]:
    """
    Efficiently retrieve basic information for multiple compounds in a single operation.
    
    Returns essential data for each compound: ChEMBL ID, preferred name, molecular weight, 
    LogP, drug status, and basic properties.
    
    Useful for:
    - Processing hit lists from virtual screening
    - Batch validation of compound collections
    - Quick property comparison across compound series
    - Generating compound datasets
    
    IMPORTANT: Limited to 50 compounds per request for optimal performance.
    For detailed information on specific compounds, use get_compound_info individually.
    
    Args:
        chembl_ids (List[str]): Array of ChEMBL compound IDs to retrieve
            Examples: ["CHEMBL59", "CHEMBL25", "CHEMBL192"]
            All IDs must be in valid CHEMBLxxxx format. Results returned in same order as input.
            Individual lookup failures don't stop the batch - errors reported per compound.
            Max: 50 compounds per request.
    
    Returns:
        Dict containing batch results with success/error status for each compound
    
    Example:
        >>> compounds_to_check = ["CHEMBL59", "CHEMBL25", "CHEMBL192"]
        >>> batch_results = batch_compound_lookup(compounds_to_check)
        >>> for result in batch_results['results']:
        ...     if result['success']:
        ...         print(f"{result['chembl_id']}: {result['pref_name']}, MW: {result['mw']}")
        ...     else:
        ...         print(f"{result['chembl_id']}: ERROR - {result['error']}")
    """
    if not isinstance(chembl_ids, list) or len(chembl_ids) == 0:
        raise ValueError("chembl_ids must be a non-empty list")
    
    if len(chembl_ids) > 50:
        raise ValueError("Maximum 50 compounds per batch request")
    
    if not all(isinstance(id, str) and len(id) > 0 for id in chembl_ids):
        raise ValueError("All chembl_ids must be non-empty strings")
    
    results = []
    
    for chembl_id in chembl_ids:
        try:
            response = _chembl_client._make_request(f'/molecule/{chembl_id}.json')
            
            mol_properties = response.get('molecule_properties', {})
            
            results.append({
                'chembl_id': chembl_id,
                'pref_name': response.get('pref_name'),
                'mw': mol_properties.get('mw_freebase'),
                'logp': mol_properties.get('alogp'),
                'hbd': mol_properties.get('hbd'),
                'hba': mol_properties.get('hba'),
                'rotatable_bonds': mol_properties.get('rtb'),
                'drug_type': response.get('drug_type'),
                'therapeutic_flag': response.get('therapeutic_flag'),
                'success': True,
            })
        except Exception as e:
            results.append({
                'chembl_id': chembl_id,
                'error': str(e),
                'success': False,
            })
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    return {
        'batch_size': len(chembl_ids),
        'successful': successful,
        'failed': failed,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# HELPER/UTILITY FUNCTIONS
# ============================================================================
def validate_chembl_id(chembl_id: str) -> bool:
    """
    Validate if a string is in valid ChEMBL ID format.
    
    Args:
        chembl_id (str): ID to validate
    
    Returns:
        bool: True if valid ChEMBL ID format, False otherwise
    """
    if not isinstance(chembl_id, str):
        return False
    return chembl_id.startswith('CHEMBL') and len(chembl_id) > 6


def format_activity_value(standard_value: float, standard_units: str) -> str:
    """
    Format bioactivity measurement value with units for display.
    
    Args:
        standard_value (float): The numeric value
        standard_units (str): The unit of measurement
    
    Returns:
        str: Formatted string representation
    """
    if standard_value is None:
        return "N/A"
    return f"{standard_value} {standard_units}" if standard_units else str(standard_value)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get definitions of all available tools for agent integration.
    
    Returns:
        List of tool definitions compatible with autogen agents
    """
    return [
        {
            'type': 'function',
            'function': {
                'name': 'search_compounds',
                'description': 'Search ChEMBL database for pharmaceutical compounds by name, brand, or ID',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Search term (drug name, brand, or ID)'},
                        'limit': {'type': 'integer', 'description': 'Max results (1-1000)', 'default': 25},
                        'offset': {'type': 'integer', 'description': 'Results to skip for pagination', 'default': 0},
                    },
                    'required': ['query'],
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_compound_info',
                'description': 'Get detailed information for a specific compound by ChEMBL ID',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'chembl_id': {'type': 'string', 'description': 'ChEMBL compound ID (e.g., CHEMBL59)'},
                    },
                    'required': ['chembl_id'],
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_compound_structure',
                'description': 'Get chemical structure data in various formats (SMILES, InChI, MOL)',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'chembl_id': {'type': 'string', 'description': 'ChEMBL compound ID'},
                        'format': {'type': 'string', 'enum': ['smiles', 'inchi', 'molfile', 'sdf'], 'default': 'smiles'},
                    },
                    'required': ['chembl_id'],
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_activities',
                'description': 'Search bioactivity data (IC50, Ki, EC50, etc.) with flexible filtering',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'target_chembl_id': {'type': 'string', 'description': 'Filter by target ID'},
                        'molecule_chembl_id': {'type': 'string', 'description': 'Filter by compound ID'},
                        'assay_chembl_id': {'type': 'string', 'description': 'Filter by assay ID'},
                        'activity_type': {'type': 'string', 'description': 'Filter by activity type (IC50, Ki, etc)'},
                        'limit': {'type': 'integer', 'description': 'Max results', 'default': 25},
                    },
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_assay_info',
                'description': 'Get detailed assay protocol and experimental metadata',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'chembl_id': {'type': 'string', 'description': 'ChEMBL assay ID'},
                    },
                    'required': ['chembl_id'],
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'batch_compound_lookup',
                'description': 'Retrieve basic info for multiple compounds (max 50)',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'chembl_ids': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of ChEMBL IDs'},
                    },
                    'required': ['chembl_ids'],
                }
            }
        },
    ]
