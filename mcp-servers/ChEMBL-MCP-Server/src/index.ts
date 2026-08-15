#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import axios, { AxiosInstance } from 'axios';

// ChEMBL API interfaces
interface CompoundSearchResult {
  molecule_chembl_id: string;
  pref_name?: string;
  molecule_type: string;
  molecule_structures: {
    canonical_smiles?: string;
    standard_inchi?: string;
    standard_inchi_key?: string;
  };
  molecule_properties: {
    molecular_weight?: number;
    alogp?: number;
    hbd?: number;
    hba?: number;
    psa?: number;
    rtb?: number;
    ro3_pass?: string;
    num_ro5_violations?: number;
  };
}

interface TargetInfo {
  target_chembl_id: string;
  pref_name: string;
  target_type: string;
  organism: string;
  species_group_flag: boolean;
  target_components?: Array<{
    component_id: number;
    component_type: string;
    accession?: string;
    sequence?: string;
  }>;
}

interface ActivityData {
  activity_id: number;
  assay_chembl_id: string;
  molecule_chembl_id: string;
  target_chembl_id: string;
  standard_type?: string;
  standard_value?: number;
  standard_units?: string;
  standard_relation?: string;
  activity_comment?: string;
}

interface AssayInfo {
  assay_chembl_id: string;
  description: string;
  assay_type: string;
  assay_organism?: string;
  assay_strain?: string;
  assay_tissue?: string;
  assay_cell_type?: string;
  assay_subcellular_fraction?: string;
  target_chembl_id?: string;
  confidence_score?: number;
}

// Type guards and validation functions
const isValidCompoundSearchArgs = (
  args: any
): args is { query: string; limit?: number; offset?: number } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.query === 'string' &&
    args.query.length > 0 &&
    (args.limit === undefined || (typeof args.limit === 'number' && args.limit > 0 && args.limit <= 1000)) &&
    (args.offset === undefined || (typeof args.offset === 'number' && args.offset >= 0))
  );
};

const isValidChemblIdArgs = (
  args: any
): args is { chembl_id: string } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.chembl_id === 'string' &&
    args.chembl_id.length > 0
  );
};

const isValidSimilaritySearchArgs = (
  args: any
): args is { smiles: string; similarity?: number; limit?: number } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.smiles === 'string' &&
    args.smiles.length > 0 &&
    (args.similarity === undefined || (typeof args.similarity === 'number' && args.similarity >= 0 && args.similarity <= 1)) &&
    (args.limit === undefined || (typeof args.limit === 'number' && args.limit > 0 && args.limit <= 1000))
  );
};

const isValidSubstructureSearchArgs = (
  args: any
): args is { smiles: string; limit?: number } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.smiles === 'string' &&
    args.smiles.length > 0 &&
    (args.limit === undefined || (typeof args.limit === 'number' && args.limit > 0 && args.limit <= 1000))
  );
};

const isValidActivitySearchArgs = (
  args: any
): args is { target_chembl_id?: string; assay_chembl_id?: string; molecule_chembl_id?: string; activity_type?: string; limit?: number } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    (args.target_chembl_id === undefined || typeof args.target_chembl_id === 'string') &&
    (args.assay_chembl_id === undefined || typeof args.assay_chembl_id === 'string') &&
    (args.molecule_chembl_id === undefined || typeof args.molecule_chembl_id === 'string') &&
    (args.activity_type === undefined || typeof args.activity_type === 'string') &&
    (args.limit === undefined || (typeof args.limit === 'number' && args.limit > 0 && args.limit <= 1000)) &&
    (args.target_chembl_id !== undefined || args.assay_chembl_id !== undefined || args.molecule_chembl_id !== undefined)
  );
};

const isValidPropertyFilterArgs = (
  args: any
): args is {
    min_mw?: number;
    max_mw?: number;
    min_logp?: number;
    max_logp?: number;
    max_hbd?: number;
    max_hba?: number;
    limit?: number
  } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    (args.min_mw === undefined || (typeof args.min_mw === 'number' && args.min_mw >= 0)) &&
    (args.max_mw === undefined || (typeof args.max_mw === 'number' && args.max_mw >= 0)) &&
    (args.min_logp === undefined || typeof args.min_logp === 'number') &&
    (args.max_logp === undefined || typeof args.max_logp === 'number') &&
    (args.max_hbd === undefined || (typeof args.max_hbd === 'number' && args.max_hbd >= 0)) &&
    (args.max_hba === undefined || (typeof args.max_hba === 'number' && args.max_hba >= 0)) &&
    (args.limit === undefined || (typeof args.limit === 'number' && args.limit > 0 && args.limit <= 1000))
  );
};

const isValidBatchArgs = (
  args: any
): args is { chembl_ids: string[] } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    Array.isArray(args.chembl_ids) &&
    args.chembl_ids.length > 0 &&
    args.chembl_ids.length <= 50 &&
    args.chembl_ids.every((id: any) => typeof id === 'string' && id.length > 0)
  );
};

class ChEMBLServer {
  private server: Server;
  private apiClient: AxiosInstance;

  constructor() {
    this.server = new Server(
      {
        name: 'chembl-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize ChEMBL API client
    this.apiClient = axios.create({
      baseURL: 'https://www.ebi.ac.uk/chembl/api/data',
      timeout: 30000,
      headers: {
        'User-Agent': 'ChEMBL-MCP-Server/1.0.0',
        'Accept': 'application/json',
      },
    });

    this.setupResourceHandlers();
    this.setupToolHandlers();

    // Error handling
    this.server.onerror = (error: any) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupResourceHandlers() {
    // List available resource templates
    this.server.setRequestHandler(
      ListResourceTemplatesRequestSchema,
      async () => ({
        resourceTemplates: [
          {
            uriTemplate: 'chembl://compound/{chembl_id}',
            name: 'ChEMBL compound entry',
            mimeType: 'application/json',
            description: 'Complete compound information for a ChEMBL ID',
          },
          {
            uriTemplate: 'chembl://target/{chembl_id}',
            name: 'ChEMBL target entry',
            mimeType: 'application/json',
            description: 'Complete target information for a ChEMBL target ID',
          },
          {
            uriTemplate: 'chembl://assay/{chembl_id}',
            name: 'ChEMBL assay entry',
            mimeType: 'application/json',
            description: 'Complete assay information for a ChEMBL assay ID',
          },
          {
            uriTemplate: 'chembl://activity/{activity_id}',
            name: 'ChEMBL activity entry',
            mimeType: 'application/json',
            description: 'Bioactivity measurement data for an activity ID',
          },
          {
            uriTemplate: 'chembl://search/{query}',
            name: 'ChEMBL search results',
            mimeType: 'application/json',
            description: 'Search results for compounds matching the query',
          },
        ],
      })
    );

    // Handle resource requests
    this.server.setRequestHandler(
      ReadResourceRequestSchema,
      async (request: any) => {
        const uri = request.params.uri;

        // Handle compound info requests
        const compoundMatch = uri.match(/^chembl:\/\/compound\/([A-Z0-9]+)$/);
        if (compoundMatch) {
          const chemblId = compoundMatch[1];
          try {
            const response = await this.apiClient.get(`/molecule/${chemblId}.json`);
            return {
              contents: [
                {
                  uri: request.params.uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(response.data, null, 2),
                },
              ],
            };
          } catch (error) {
            throw new McpError(
              ErrorCode.InternalError,
              `Failed to fetch compound ${chemblId}: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }

        // Handle target info requests
        const targetMatch = uri.match(/^chembl:\/\/target\/([A-Z0-9]+)$/);
        if (targetMatch) {
          const chemblId = targetMatch[1];
          try {
            const response = await this.apiClient.get(`/target/${chemblId}.json`);
            return {
              contents: [
                {
                  uri: request.params.uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(response.data, null, 2),
                },
              ],
            };
          } catch (error) {
            throw new McpError(
              ErrorCode.InternalError,
              `Failed to fetch target ${chemblId}: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }

        // Handle assay info requests
        const assayMatch = uri.match(/^chembl:\/\/assay\/([A-Z0-9]+)$/);
        if (assayMatch) {
          const chemblId = assayMatch[1];
          try {
            const response = await this.apiClient.get(`/assay/${chemblId}.json`);
            return {
              contents: [
                {
                  uri: request.params.uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(response.data, null, 2),
                },
              ],
            };
          } catch (error) {
            throw new McpError(
              ErrorCode.InternalError,
              `Failed to fetch assay ${chemblId}: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }

        // Handle activity info requests
        const activityMatch = uri.match(/^chembl:\/\/activity\/([0-9]+)$/);
        if (activityMatch) {
          const activityId = activityMatch[1];
          try {
            const response = await this.apiClient.get(`/activity/${activityId}.json`);
            return {
              contents: [
                {
                  uri: request.params.uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(response.data, null, 2),
                },
              ],
            };
          } catch (error) {
            throw new McpError(
              ErrorCode.InternalError,
              `Failed to fetch activity ${activityId}: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }

        // Handle search requests
        const searchMatch = uri.match(/^chembl:\/\/search\/(.+)$/);
        if (searchMatch) {
          const query = decodeURIComponent(searchMatch[1]);
          try {
            const response = await this.apiClient.get('/molecule/search.json', {
              params: {
                q: query,
                limit: 25,
              },
            });

            return {
              contents: [
                {
                  uri: request.params.uri,
                  mimeType: 'application/json',
                  text: JSON.stringify(response.data, null, 2),
                },
              ],
            };
          } catch (error) {
            throw new McpError(
              ErrorCode.InternalError,
              `Failed to search compounds: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }

        throw new McpError(
          ErrorCode.InvalidRequest,
          `Invalid URI format: ${uri}`
        );
      }
    );
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        // Core Chemical Search & Retrieval (5 tools)
        {
          name: 'search_compounds',
          description: 'Search ChEMBL database for compounds by name, synonym, or identifier',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Search query (compound name, synonym, or identifier)' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
              offset: { type: 'number', description: 'Number of results to skip (default: 0)', minimum: 0 },
            },
            required: ['query'],
          },
        },
        {
          name: 'get_compound_info',
          description: 'Get detailed information for a specific compound by ChEMBL ID',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID (e.g., CHEMBL59)' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'search_by_inchi',
          description: 'Search for compounds by InChI key or InChI string',
          inputSchema: {
            type: 'object',
            properties: {
              inchi: { type: 'string', description: 'InChI key or InChI string' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['inchi'],
          },
        },
        {
          name: 'get_compound_structure',
          description: 'Retrieve chemical structure information in various formats',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
              format: { type: 'string', enum: ['smiles', 'inchi', 'molfile', 'sdf'], description: 'Structure format (default: smiles)' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'search_similar_compounds',
          description: 'Find chemically similar compounds using Tanimoto similarity',
          inputSchema: {
            type: 'object',
            properties: {
              smiles: { type: 'string', description: 'SMILES string of the query molecule' },
              similarity: { type: 'number', description: 'Similarity threshold (0-1, default: 0.7)', minimum: 0, maximum: 1 },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['smiles'],
          },
        },
        // Target Analysis & Drug Discovery (5 tools)
        {
          name: 'search_targets',
          description: 'Search for biological targets by name or type',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Target name or search query' },
              target_type: { type: 'string', description: 'Target type filter (e.g., SINGLE PROTEIN, PROTEIN COMPLEX)' },
              organism: { type: 'string', description: 'Organism filter' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['query'],
          },
        },
        {
          name: 'get_target_info',
          description: 'Get detailed information for a specific target by ChEMBL target ID',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL target ID (e.g., CHEMBL2095173)' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'get_target_compounds',
          description: 'Get compounds tested against a specific target',
          inputSchema: {
            type: 'object',
            properties: {
              target_chembl_id: { type: 'string', description: 'ChEMBL target ID' },
              activity_type: { type: 'string', description: 'Activity type filter (e.g., IC50, Ki, Kd)' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['target_chembl_id'],
          },
        },
        {
          name: 'search_by_uniprot',
          description: 'Find ChEMBL targets by UniProt accession',
          inputSchema: {
            type: 'object',
            properties: {
              uniprot_id: { type: 'string', description: 'UniProt accession number' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['uniprot_id'],
          },
        },
        {
          name: 'get_target_pathways',
          description: 'Get biological pathways associated with a target',
          inputSchema: {
            type: 'object',
            properties: {
              target_chembl_id: { type: 'string', description: 'ChEMBL target ID' },
            },
            required: ['target_chembl_id'],
          },
        },
        // Bioactivity & Assay Data (5 tools)
        {
          name: 'search_activities',
          description: 'Search bioactivity measurements and assay results',
          inputSchema: {
            type: 'object',
            properties: {
              target_chembl_id: { type: 'string', description: 'ChEMBL target ID filter' },
              assay_chembl_id: { type: 'string', description: 'ChEMBL assay ID filter' },
              molecule_chembl_id: { type: 'string', description: 'ChEMBL compound ID filter' },
              activity_type: { type: 'string', description: 'Activity type (e.g., IC50, Ki, EC50)' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: [],
          },
        },
        {
          name: 'get_assay_info',
          description: 'Get detailed information for a specific assay by ChEMBL assay ID',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL assay ID (e.g., CHEMBL1217643)' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'search_by_activity_type',
          description: 'Find bioactivity data by specific activity type and value range',
          inputSchema: {
            type: 'object',
            properties: {
              activity_type: { type: 'string', description: 'Activity type (e.g., IC50, Ki, EC50, Kd)' },
              min_value: { type: 'number', description: 'Minimum activity value' },
              max_value: { type: 'number', description: 'Maximum activity value' },
              units: { type: 'string', description: 'Units filter (e.g., nM, uM)' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['activity_type'],
          },
        },
        {
          name: 'get_dose_response',
          description: 'Get dose-response data and activity profiles for compounds',
          inputSchema: {
            type: 'object',
            properties: {
              molecule_chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
              target_chembl_id: { type: 'string', description: 'ChEMBL target ID (optional filter)' },
            },
            required: ['molecule_chembl_id'],
          },
        },
        {
          name: 'compare_activities',
          description: 'Compare bioactivity data across multiple compounds or targets',
          inputSchema: {
            type: 'object',
            properties: {
              molecule_chembl_ids: { type: 'array', items: { type: 'string' }, description: 'Array of ChEMBL compound IDs (2-10)', minItems: 2, maxItems: 10 },
              target_chembl_id: { type: 'string', description: 'ChEMBL target ID for comparison' },
              activity_type: { type: 'string', description: 'Activity type for comparison' },
            },
            required: ['molecule_chembl_ids'],
          },
        },
        // Drug Development & Clinical Data (4 tools)
        {
          name: 'search_drugs',
          description: 'Search for approved drugs and clinical candidates',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Drug name or search query' },
              development_phase: { type: 'string', description: 'Development phase filter (e.g., Approved, Phase III)' },
              therapeutic_area: { type: 'string', description: 'Therapeutic area filter' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['query'],
          },
        },
        {
          name: 'get_drug_info',
          description: 'Get drug development status and clinical trial information',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'search_drug_indications',
          description: 'Search for therapeutic indications and disease areas',
          inputSchema: {
            type: 'object',
            properties: {
              indication: { type: 'string', description: 'Disease or indication search term' },
              drug_type: { type: 'string', description: 'Drug type filter (e.g., Small molecule, Antibody)' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['indication'],
          },
        },
        {
          name: 'get_mechanism_of_action',
          description: 'Get mechanism of action and target interaction data',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
            },
            required: ['chembl_id'],
          },
        },
        // Chemical Property Analysis (4 tools)
        {
          name: 'analyze_admet_properties',
          description: 'Analyze ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity)',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'calculate_descriptors',
          description: 'Calculate molecular descriptors and physicochemical properties',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
              smiles: { type: 'string', description: 'SMILES string (alternative to ChEMBL ID)' },
            },
            required: [],
          },
        },
        {
          name: 'predict_solubility',
          description: 'Predict aqueous solubility and permeability properties',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
              smiles: { type: 'string', description: 'SMILES string (alternative to ChEMBL ID)' },
            },
            required: [],
          },
        },
        {
          name: 'assess_drug_likeness',
          description: 'Assess drug-likeness using Lipinski Rule of Five and other metrics',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound ID' },
              smiles: { type: 'string', description: 'SMILES string (alternative to ChEMBL ID)' },
            },
            required: [],
          },
        },
        // Advanced Search & Cross-References (4 tools)
        {
          name: 'substructure_search',
          description: 'Find compounds containing specific substructures',
          inputSchema: {
            type: 'object',
            properties: {
              smiles: { type: 'string', description: 'SMILES string of the substructure query' },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: ['smiles'],
          },
        },
        {
          name: 'batch_compound_lookup',
          description: 'Process multiple ChEMBL IDs efficiently',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_ids: { type: 'array', items: { type: 'string' }, description: 'Array of ChEMBL compound IDs (1-50)', minItems: 1, maxItems: 50 },
            },
            required: ['chembl_ids'],
          },
        },
        {
          name: 'get_external_references',
          description: 'Get links to external databases (PubChem, DrugBank, PDB, etc.)',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { type: 'string', description: 'ChEMBL compound or target ID' },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'advanced_search',
          description: 'Complex queries with multiple chemical and biological filters',
          inputSchema: {
            type: 'object',
            properties: {
              min_mw: { type: 'number', description: 'Minimum molecular weight (Da)', minimum: 0 },
              max_mw: { type: 'number', description: 'Maximum molecular weight (Da)', minimum: 0 },
              min_logp: { type: 'number', description: 'Minimum LogP value' },
              max_logp: { type: 'number', description: 'Maximum LogP value' },
              max_hbd: { type: 'number', description: 'Maximum hydrogen bond donors', minimum: 0 },
              max_hba: { type: 'number', description: 'Maximum hydrogen bond acceptors', minimum: 0 },
              limit: { type: 'number', description: 'Number of results to return (1-1000, default: 25)', minimum: 1, maximum: 1000 },
            },
            required: [],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request: any) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          // Core Chemical Search & Retrieval
          case 'search_compounds':
            return await this.handleSearchCompounds(args);
          case 'get_compound_info':
            return await this.handleGetCompoundInfo(args);
          case 'search_by_inchi':
            return await this.handleSearchByInchi(args);
          case 'get_compound_structure':
            return await this.handleGetCompoundStructure(args);
          case 'search_similar_compounds':
            return await this.handleSearchSimilarCompounds(args);
          // Target Analysis & Drug Discovery
          case 'search_targets':
            return await this.handleSearchTargets(args);
          case 'get_target_info':
            return await this.handleGetTargetInfo(args);
          case 'get_target_compounds':
            return await this.handleGetTargetCompounds(args);
          case 'search_by_uniprot':
            return await this.handleSearchByUniprot(args);
          case 'get_target_pathways':
            return await this.handleGetTargetPathways(args);
          // Bioactivity & Assay Data
          case 'search_activities':
            return await this.handleSearchActivities(args);
          case 'get_assay_info':
            return await this.handleGetAssayInfo(args);
          case 'search_by_activity_type':
            return await this.handleSearchByActivityType(args);
          case 'get_dose_response':
            return await this.handleGetDoseResponse(args);
          case 'compare_activities':
            return await this.handleCompareActivities(args);
          // Drug Development & Clinical Data
          case 'search_drugs':
            return await this.handleSearchDrugs(args);
          case 'get_drug_info':
            return await this.handleGetDrugInfo(args);
          case 'search_drug_indications':
            return await this.handleSearchDrugIndications(args);
          case 'get_mechanism_of_action':
            return await this.handleGetMechanismOfAction(args);
          // Chemical Property Analysis
          case 'analyze_admet_properties':
            return await this.handleAnalyzeAdmetProperties(args);
          case 'calculate_descriptors':
            return await this.handleCalculateDescriptors(args);
          case 'predict_solubility':
            return await this.handlePredictSolubility(args);
          case 'assess_drug_likeness':
            return await this.handleAssessDrugLikeness(args);
          // Advanced Search & Cross-References
          case 'substructure_search':
            return await this.handleSubstructureSearch(args);
          case 'batch_compound_lookup':
            return await this.handleBatchCompoundLookup(args);
          case 'get_external_references':
            return await this.handleGetExternalReferences(args);
          case 'advanced_search':
            return await this.handleAdvancedSearch(args);
          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing tool ${name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  // Core Chemical Search & Retrieval handlers
  private async handleSearchCompounds(args: any) {
    if (!isValidCompoundSearchArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid compound search arguments');
    }

    try {
      const response = await this.apiClient.get('/molecule/search.json', {
        params: {
          q: args.query,
          limit: args.limit || 25,
          offset: args.offset || 0,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search compounds: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleGetCompoundInfo(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid ChEMBL ID arguments');
    }

    try {
      const response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get compound info: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  // Simplified placeholder implementations for the remaining tools
  private async handleSearchByInchi(args: any) {
    if (!args || typeof args.inchi !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid InChI arguments');
    }

    try {
      // ChEMBL supports InChI and InChI key searches
      const response = await this.apiClient.get('/molecule/search.json', {
        params: {
          q: args.inchi,
          limit: args.limit || 25,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search by InChI: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleGetCompoundStructure(args: any) {
    if (!args || typeof args.chembl_id !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid arguments');
    }

    try {
      const response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
      const compound = response.data;

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              chembl_id: compound.molecule_chembl_id,
              structures: compound.molecule_structures || {},
              requested_format: args.format || 'smiles'
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to get structure: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async handleSearchSimilarCompounds(args: any) {
    if (!isValidSimilaritySearchArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid similarity search arguments');
    }

    try {
      // ChEMBL similarity search using SMILES
      const similarity = args.similarity !== undefined ? Math.round(args.similarity * 100) : 70;
      const response = await this.apiClient.get('/similarity/' + encodeURIComponent(args.smiles) + '/' + similarity + '.json', {
        params: {
          limit: args.limit || 25,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search similar compounds: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleSearchTargets(args: any) {
    try {
      const response = await this.apiClient.get('/target/search.json', {
        params: { q: args.query, limit: args.limit || 25 },
      });
      return { content: [{ type: 'text', text: JSON.stringify(response.data, null, 2) }] };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to search targets: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async handleGetTargetInfo(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid arguments');
    }

    try {
      const response = await this.apiClient.get(`/target/${args.chembl_id}.json`);
      return { content: [{ type: 'text', text: JSON.stringify(response.data, null, 2) }] };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to get target info: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Placeholder implementations for remaining tools
  private async handleGetTargetCompounds(args: any) {
    if (!args || typeof args.target_chembl_id !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid target compounds arguments');
    }

    try {
      const params: any = {
        target_chembl_id: args.target_chembl_id,
        limit: args.limit || 25,
      };

      if (args.activity_type) {
        params.standard_type = args.activity_type;
      }

      const response = await this.apiClient.get('/activity.json', { params });

      // Extract unique compounds from activities
      const activities = response.data.activities || [];
      const compoundIds = [...new Set(activities.map((a: any) => a.molecule_chembl_id))];

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              target_chembl_id: args.target_chembl_id,
              total_activities: activities.length,
              unique_compounds: compoundIds.length,
              compound_ids: compoundIds.slice(0, 100),
              activities: activities.slice(0, 50),
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get target compounds: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleSearchByUniprot(args: any) {
    if (!args || typeof args.uniprot_id !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid UniProt arguments');
    }

    try {
      const response = await this.apiClient.get('/target/search.json', {
        params: {
          q: args.uniprot_id,
          limit: args.limit || 25,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search by UniProt: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleGetTargetPathways(args: any) {
    if (!args || typeof args.target_chembl_id !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid target pathways arguments');
    }

    try {
      // Get target information which includes pathway data
      const targetResponse = await this.apiClient.get(`/target/${args.target_chembl_id}.json`);
      const target = targetResponse.data;

      // Extract pathway information from target data
      const pathways = {
        target_chembl_id: args.target_chembl_id,
        target_name: target.pref_name,
        target_type: target.target_type,
        cross_references: target.cross_references || [],
        pathways: (target.cross_references || []).filter((ref: any) =>
          ref.xref_src === 'Reactome' || ref.xref_src === 'KEGG' || ref.xref_src === 'WikiPathways'
        ),
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(pathways, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get target pathways: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleSearchActivities(args: any) {
    try {
      const params: any = { limit: args.limit || 25 };
      if (args.target_chembl_id) params.target_chembl_id = args.target_chembl_id;
      if (args.molecule_chembl_id) params.molecule_chembl_id = args.molecule_chembl_id;
      if (args.activity_type) params.standard_type = args.activity_type;

      const response = await this.apiClient.get('/activity.json', { params });
      return { content: [{ type: 'text', text: JSON.stringify(response.data, null, 2) }] };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to search activities: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async handleGetAssayInfo(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid arguments');
    }

    try {
      const response = await this.apiClient.get(`/assay/${args.chembl_id}.json`);
      return { content: [{ type: 'text', text: JSON.stringify(response.data, null, 2) }] };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to get assay info: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Remaining placeholder implementations
  private async handleSearchByActivityType(args: any) {
    if (!args || typeof args.activity_type !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid activity type arguments');
    }

    try {
      const params: any = {
        standard_type: args.activity_type,
        limit: args.limit || 25,
      };

      if (args.min_value !== undefined) {
        params.standard_value__gte = args.min_value;
      }
      if (args.max_value !== undefined) {
        params.standard_value__lte = args.max_value;
      }
      if (args.units) {
        params.standard_units = args.units;
      }

      const response = await this.apiClient.get('/activity.json', { params });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search by activity type: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleGetDoseResponse(args: any) {
    if (!args || typeof args.molecule_chembl_id !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid dose response arguments');
    }

    try {
      const params: any = {
        molecule_chembl_id: args.molecule_chembl_id,
        limit: 100,
      };

      if (args.target_chembl_id) {
        params.target_chembl_id = args.target_chembl_id;
      }

      const response = await this.apiClient.get('/activity.json', { params });
      const activities = response.data.activities || [];

      // Group activities by assay and extract dose-response data
      const doseResponseData = activities
        .filter((a: any) => a.standard_value !== null && a.standard_type)
        .map((a: any) => ({
          assay_chembl_id: a.assay_chembl_id,
          target_chembl_id: a.target_chembl_id,
          activity_type: a.standard_type,
          value: a.standard_value,
          units: a.standard_units,
          relation: a.standard_relation,
        }));

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              molecule_chembl_id: args.molecule_chembl_id,
              total_measurements: doseResponseData.length,
              dose_response_data: doseResponseData,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get dose response: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleCompareActivities(args: any) {
    if (!args || !Array.isArray(args.molecule_chembl_ids) || args.molecule_chembl_ids.length < 2) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid activity comparison arguments');
    }

    try {
      const comparisonResults = [];

      for (const chemblId of args.molecule_chembl_ids.slice(0, 10)) {
        const params: any = {
          molecule_chembl_id: chemblId,
          limit: 50,
        };

        if (args.target_chembl_id) {
          params.target_chembl_id = args.target_chembl_id;
        }
        if (args.activity_type) {
          params.standard_type = args.activity_type;
        }

        try {
          const response = await this.apiClient.get('/activity.json', { params });
          const activities = response.data.activities || [];

          comparisonResults.push({
            molecule_chembl_id: chemblId,
            activity_count: activities.length,
            activities: activities.slice(0, 20),
          });
        } catch (error) {
          comparisonResults.push({
            molecule_chembl_id: chemblId,
            error: error instanceof Error ? error.message : 'Unknown error',
          });
        }
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              comparison_results: comparisonResults,
              target_filter: args.target_chembl_id,
              activity_type_filter: args.activity_type,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to compare activities: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleSearchDrugs(args: any) {
    if (!args || typeof args.query !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid drug search arguments');
    }

    try {
      // Search for drugs using molecule endpoint with max_phase filter
      const params: any = {
        q: args.query,
        limit: args.limit || 25,
      };

      const response = await this.apiClient.get('/molecule/search.json', { params });
      const molecules = response.data.molecules || [];

      // Filter for drugs (molecules with max_phase >= 1)
      const drugs = molecules.filter((m: any) => m.max_phase && m.max_phase >= 1);

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              query: args.query,
              total_results: drugs.length,
              drugs: drugs,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search drugs: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleGetDrugInfo(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid drug info arguments');
    }

    try {
      // Get molecule information
      const moleculeResponse = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
      const molecule = moleculeResponse.data;

      // Get drug indication data if available
      let indications = [];
      try {
        const indicationResponse = await this.apiClient.get('/drug_indication.json', {
          params: { molecule_chembl_id: args.chembl_id, limit: 50 },
        });
        indications = indicationResponse.data.drug_indications || [];
      } catch (e) {
        // Indications may not be available for all compounds
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              chembl_id: args.chembl_id,
              molecule_info: molecule,
              development_phase: molecule.max_phase,
              indications: indications,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get drug info: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleSearchDrugIndications(args: any) {
    if (!args || typeof args.indication !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid drug indications arguments');
    }

    try {
      const response = await this.apiClient.get('/drug_indication.json', {
        params: {
          q: args.indication,
          limit: args.limit || 25,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to search drug indications: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleGetMechanismOfAction(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid mechanism of action arguments');
    }

    try {
      const response = await this.apiClient.get('/mechanism.json', {
        params: {
          molecule_chembl_id: args.chembl_id,
          limit: 50,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get mechanism of action: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleAnalyzeAdmetProperties(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid ADMET analysis arguments');
    }

    try {
      const response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
      const molecule = response.data;
      const props = molecule.molecule_properties || {};

      // Analyze ADMET-related properties from ChEMBL data
      const admetAnalysis = {
        chembl_id: args.chembl_id,
        absorption: {
          molecular_weight: props.full_mwt || props.molecular_weight,
          alogp: props.alogp,
          hbd: props.hbd,
          hba: props.hba,
          psa: props.psa,
          ro3_pass: props.ro3_pass,
          assessment: this.assessAbsorption(props),
        },
        distribution: {
          logp: props.alogp,
          psa: props.psa,
          assessment: this.assessDistribution(props),
        },
        drug_likeness: {
          lipinski_violations: props.num_ro5_violations,
          rotatable_bonds: props.rtb,
          aromatic_rings: props.aromatic_rings,
          assessment: this.assessDrugLikeness(props),
        },
        molecular_properties: props,
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(admetAnalysis, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to analyze ADMET properties: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private assessAbsorption(props: any): string {
    const mw = props.full_mwt || props.molecular_weight || 0;
    const hbd = props.hbd || 0;
    const hba = props.hba || 0;
    const psa = props.psa || 0;

    if (mw > 500 || hbd > 5 || hba > 10 || psa > 140) {
      return 'Poor oral absorption predicted';
    } else if (mw < 400 && hbd <= 3 && hba <= 7 && psa < 100) {
      return 'Good oral absorption predicted';
    }
    return 'Moderate oral absorption predicted';
  }

  private assessDistribution(props: any): string {
    const logp = props.alogp || 0;
    const psa = props.psa || 0;

    if (logp > 5) {
      return 'High lipophilicity - may accumulate in tissues';
    } else if (logp < 0) {
      return 'Low lipophilicity - limited tissue distribution';
    } else if (psa < 90 && logp > 0 && logp < 3) {
      return 'Good CNS penetration predicted';
    }
    return 'Moderate distribution predicted';
  }

  private assessDrugLikeness(props: any): string {
    const violations = props.num_ro5_violations || 0;
    if (violations === 0) {
      return 'Excellent drug-likeness (Lipinski compliant)';
    } else if (violations === 1) {
      return 'Good drug-likeness (1 Lipinski violation)';
    }
    return `Poor drug-likeness (${violations} Lipinski violations)`;
  }

  private async handleCalculateDescriptors(args: any) {
    if (!args || (!args.chembl_id && !args.smiles)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid descriptor calculation arguments');
    }

    try {
      let molecule;
      if (args.chembl_id) {
        const response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
        molecule = response.data;
      } else {
        // For SMILES input, we can only provide limited info
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: 'SMILES-based descriptor calculation requires ChEMBL ID',
                smiles: args.smiles,
              }, null, 2),
            },
          ],
        };
      }

      const props = molecule.molecule_properties || {};
      const structures = molecule.molecule_structures || {};

      const descriptors = {
        chembl_id: molecule.molecule_chembl_id,
        basic_properties: {
          molecular_weight: props.full_mwt || props.molecular_weight,
          exact_mass: props.full_mwt,
          molecular_formula: props.molecular_formula,
        },
        lipophilicity: {
          alogp: props.alogp,
          logp: props.alogp,
        },
        hydrogen_bonding: {
          hbd: props.hbd,
          hba: props.hba,
        },
        polar_surface_area: {
          psa: props.psa,
          tpsa: props.psa,
        },
        complexity: {
          rotatable_bonds: props.rtb,
          aromatic_rings: props.aromatic_rings,
          heavy_atoms: props.heavy_atoms,
          num_atoms: props.num_atoms,
        },
        drug_likeness_metrics: {
          ro5_violations: props.num_ro5_violations,
          ro3_pass: props.ro3_pass,
          cx_logp: props.cx_logp,
          cx_logd: props.cx_logd,
        },
        structures: {
          smiles: structures.canonical_smiles,
          inchi: structures.standard_inchi,
          inchi_key: structures.standard_inchi_key,
        },
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(descriptors, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to calculate descriptors: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handlePredictSolubility(args: any) {
    if (!args || (!args.chembl_id && !args.smiles)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid solubility prediction arguments');
    }

    try {
      let molecule;
      if (args.chembl_id) {
        const response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
        molecule = response.data;
      } else {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: 'SMILES-based solubility prediction requires ChEMBL ID',
                smiles: args.smiles,
              }, null, 2),
            },
          ],
        };
      }

      const props = molecule.molecule_properties || {};

      // Predict solubility based on molecular properties
      const logp = props.alogp || 0;
      const psa = props.psa || 0;
      const mw = props.full_mwt || props.molecular_weight || 0;
      const hbd = props.hbd || 0;
      const hba = props.hba || 0;

      // Simple solubility prediction model
      let solubilityClass = 'Moderate';
      let permeability = 'Moderate';

      if (logp < 0 && psa > 100) {
        solubilityClass = 'High';
        permeability = 'Low';
      } else if (logp > 5 || psa < 40) {
        solubilityClass = 'Low';
        permeability = 'High';
      } else if (logp > 3 && psa < 70) {
        solubilityClass = 'Low-Moderate';
        permeability = 'High';
      } else if (logp < 2 && psa > 80) {
        solubilityClass = 'Moderate-High';
        permeability = 'Low-Moderate';
      }

      const solubilityPrediction = {
        chembl_id: molecule.molecule_chembl_id,
        aqueous_solubility: {
          predicted_class: solubilityClass,
          logp: logp,
          psa: psa,
          factors: {
            lipophilicity: logp > 3 ? 'High (reduces solubility)' : 'Moderate',
            polar_surface_area: psa > 100 ? 'High (increases solubility)' : 'Moderate',
            hydrogen_bonding: `${hbd} donors, ${hba} acceptors`,
          },
        },
        permeability: {
          predicted_class: permeability,
          assessment: this.assessPermeability(props),
        },
        molecular_properties: {
          molecular_weight: mw,
          alogp: logp,
          psa: psa,
          hbd: hbd,
          hba: hba,
        },
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(solubilityPrediction, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to predict solubility: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private assessPermeability(props: any): string {
    const psa = props.psa || 0;
    const logp = props.alogp || 0;

    if (psa < 90 && logp > 0 && logp < 5) {
      return 'Good membrane permeability predicted';
    } else if (psa > 140 || logp < -1) {
      return 'Poor membrane permeability predicted';
    }
    return 'Moderate membrane permeability predicted';
  }

  private async handleAssessDrugLikeness(args: any) {
    if (!args || (!args.chembl_id && !args.smiles)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid drug-likeness assessment arguments');
    }

    try {
      let molecule;
      if (args.chembl_id) {
        const response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
        molecule = response.data;
      } else {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: 'SMILES-based drug-likeness assessment requires ChEMBL ID',
                smiles: args.smiles,
              }, null, 2),
            },
          ],
        };
      }

      const props = molecule.molecule_properties || {};

      // Lipinski Rule of Five
      const mw = props.full_mwt || props.molecular_weight || 0;
      const logp = props.alogp || 0;
      const hbd = props.hbd || 0;
      const hba = props.hba || 0;

      const lipinskiViolations = [
        mw > 500 ? 'Molecular weight > 500 Da' : null,
        logp > 5 ? 'LogP > 5' : null,
        hbd > 5 ? 'H-bond donors > 5' : null,
        hba > 10 ? 'H-bond acceptors > 10' : null,
      ].filter(v => v !== null);

      // Veber rules
      const rtb = props.rtb || 0;
      const psa = props.psa || 0;
      const veberPass = rtb <= 10 && psa <= 140;

      // Overall assessment
      const drugLikenessAssessment = {
        chembl_id: molecule.molecule_chembl_id,
        lipinski_rule_of_five: {
          violations: lipinskiViolations.length,
          details: lipinskiViolations.length > 0 ? lipinskiViolations : ['All criteria met'],
          pass: lipinskiViolations.length === 0,
          criteria: {
            molecular_weight: { value: mw, limit: 500, pass: mw <= 500 },
            logp: { value: logp, limit: 5, pass: logp <= 5 },
            hbd: { value: hbd, limit: 5, pass: hbd <= 5 },
            hba: { value: hba, limit: 10, pass: hba <= 10 },
          },
        },
        veber_rules: {
          pass: veberPass,
          criteria: {
            rotatable_bonds: { value: rtb, limit: 10, pass: rtb <= 10 },
            psa: { value: psa, limit: 140, pass: psa <= 140 },
          },
        },
        overall_assessment: {
          drug_likeness: lipinskiViolations.length === 0 ? 'Excellent' : lipinskiViolations.length === 1 ? 'Good' : 'Poor',
          oral_bioavailability: veberPass && lipinskiViolations.length <= 1 ? 'Likely' : 'Uncertain',
          recommendation: this.getDrugLikenessRecommendation(lipinskiViolations.length, veberPass),
        },
        molecular_properties: props,
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(drugLikenessAssessment, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to assess drug-likeness: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private getDrugLikenessRecommendation(violations: number, veberPass: boolean): string {
    if (violations === 0 && veberPass) {
      return 'Excellent drug-like properties - suitable for oral administration';
    } else if (violations <= 1 && veberPass) {
      return 'Good drug-like properties - likely suitable for development';
    } else if (violations <= 2) {
      return 'Moderate drug-like properties - may require optimization';
    }
    return 'Poor drug-like properties - significant optimization needed';
  }

  private async handleSubstructureSearch(args: any) {
    if (!isValidSubstructureSearchArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid substructure search arguments');
    }

    try {
      // ChEMBL substructure search using SMILES
      const response = await this.apiClient.get('/substructure/' + encodeURIComponent(args.smiles) + '.json', {
        params: {
          limit: args.limit || 25,
        },
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(response.data, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to perform substructure search: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private async handleBatchCompoundLookup(args: any) {
    if (!isValidBatchArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid batch arguments');
    }

    try {
      const results = [];
      for (const chemblId of args.chembl_ids.slice(0, 10)) { // Limit to 10 for demo
        try {
          const response = await this.apiClient.get(`/molecule/${chemblId}.json`);
          results.push({ chembl_id: chemblId, data: response.data, success: true });
        } catch (error) {
          results.push({ chembl_id: chemblId, error: error instanceof Error ? error.message : 'Unknown error', success: false });
        }
      }

      return { content: [{ type: 'text', text: JSON.stringify({ batch_results: results }, null, 2) }] };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Batch lookup failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async handleGetExternalReferences(args: any) {
    if (!isValidChemblIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid external references arguments');
    }

    try {
      // Try to get molecule data first
      let response;
      let entityType = 'molecule';

      try {
        response = await this.apiClient.get(`/molecule/${args.chembl_id}.json`);
      } catch (e) {
        // If not a molecule, try target
        try {
          response = await this.apiClient.get(`/target/${args.chembl_id}.json`);
          entityType = 'target';
        } catch (e2) {
          throw new McpError(ErrorCode.InvalidParams, 'ChEMBL ID not found as molecule or target');
        }
      }

      const entity = response.data;
      const crossRefs = entity.cross_references || [];

      // Organize external references by database
      const externalReferences = {
        chembl_id: args.chembl_id,
        entity_type: entityType,
        databases: {} as any,
      };

      // Group references by source
      crossRefs.forEach((ref: any) => {
        const source = ref.xref_src || ref.xref_name;
        if (!externalReferences.databases[source]) {
          externalReferences.databases[source] = [];
        }
        externalReferences.databases[source].push({
          id: ref.xref_id,
          name: ref.xref_name,
          url: this.getExternalUrl(source, ref.xref_id),
        });
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(externalReferences, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to get external references: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  private getExternalUrl(source: string, id: string): string {
    const urlMap: { [key: string]: string } = {
      'PubChem': `https://pubchem.ncbi.nlm.nih.gov/compound/${id}`,
      'DrugBank': `https://www.drugbank.ca/drugs/${id}`,
      'PDB': `https://www.rcsb.org/structure/${id}`,
      'UniProt': `https://www.uniprot.org/uniprot/${id}`,
      'Wikipedia': `https://en.wikipedia.org/wiki/${id}`,
      'KEGG': `https://www.genome.jp/entry/${id}`,
      'Reactome': `https://reactome.org/content/detail/${id}`,
    };
    return urlMap[source] || `https://www.ebi.ac.uk/chembl/`;
  }

  private async handleAdvancedSearch(args: any) {
    if (!isValidPropertyFilterArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid advanced search arguments');
    }

    try {
      // Build filter query for ChEMBL API
      const filters: string[] = [];

      if (args.min_mw !== undefined) {
        filters.push(`molecule_properties__mw_freebase__gte=${args.min_mw}`);
      }
      if (args.max_mw !== undefined) {
        filters.push(`molecule_properties__mw_freebase__lte=${args.max_mw}`);
      }
      if (args.min_logp !== undefined) {
        filters.push(`molecule_properties__alogp__gte=${args.min_logp}`);
      }
      if (args.max_logp !== undefined) {
        filters.push(`molecule_properties__alogp__lte=${args.max_logp}`);
      }
      if (args.max_hbd !== undefined) {
        filters.push(`molecule_properties__hbd__lte=${args.max_hbd}`);
      }
      if (args.max_hba !== undefined) {
        filters.push(`molecule_properties__hba__lte=${args.max_hba}`);
      }

      const filterString = filters.join('&');
      const response = await this.apiClient.get(`/molecule.json?${filterString}&limit=${args.limit || 25}`);

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              filters: args,
              results: response.data,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to perform advanced search: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('ChEMBL MCP server running on stdio');
  }
}

const server = new ChEMBLServer();
server.run().catch(console.error);
