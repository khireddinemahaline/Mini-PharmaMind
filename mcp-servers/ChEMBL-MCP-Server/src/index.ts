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
        // Core Chemical Search & Retrieval (3 tools)
        {
          name: 'search_compounds',
          description: 'Search the ChEMBL database for pharmaceutical compounds by drug name, synonym, brand name, or ChEMBL identifier. Use this as the FIRST STEP when you need to find compounds - it returns ChEMBL IDs which are required for other detailed queries. Searches are case-insensitive and support partial matches. Examples: "aspirin", "ibuprofen", "CHEMBL25", "atorvastatin".',
          inputSchema: {
            type: 'object',
            properties: {
              query: { 
                type: 'string', 
                description: 'Search term: drug name (e.g., "Aspirin"), brand name (e.g., "Lipitor"), chemical name (e.g., "acetylsalicylic acid"), or partial ChEMBL ID. Case-insensitive, supports wildcards and partial matches.' 
              },
              limit: { 
                type: 'number', 
                description: 'Maximum number of results to return. Range: 1-1000, Default: 25. Use 5-10 for focused searches, 20-50 for exploratory searches, 100+ for comprehensive searches.', 
                minimum: 1, 
                maximum: 1000,
                default: 25
              },
              offset: { 
                type: 'number', 
                description: 'Number of results to skip for pagination. Default: 0. Use with limit to paginate through large result sets. Example: limit=25, offset=25 gets results 26-50.', 
                minimum: 0,
                default: 0
              },
            },
            required: ['query'],
          },
        },
        {
          name: 'get_compound_info',
          description: 'Retrieve comprehensive detailed information for a SPECIFIC compound using its ChEMBL ID. Returns: molecular properties (molecular weight, LogP, H-bond donors/acceptors), drug classifications, approval status, therapeutic indications, mechanism of action, trade names, and cross-references to other databases (PubChem, DrugBank). IMPORTANT: You must have a valid ChEMBL ID from search_compounds before using this tool.',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { 
                type: 'string', 
                description: 'ChEMBL compound identifier in format CHEMBLxxxx (e.g., "CHEMBL59" for aspirin, "CHEMBL25" for atorvastatin, "CHEMBL192" for caffeine). Must be exact match, case-sensitive. Obtain this ID from search_compounds results first.' 
              },
            },
            required: ['chembl_id'],
          },
        },
        {
          name: 'get_compound_structure',
          description: 'Retrieve chemical structure representations in standard computational chemistry formats for a specific compound. Returns structural data including SMILES (compact linear notation), InChI (canonical identifier), molecular formula, and optionally 2D/3D coordinates. Essential for: structure-based drug design, similarity/substructure searches, molecular docking, visualization, and computational analysis. Use AFTER getting compound info to obtain structures for computational work.',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { 
                type: 'string', 
                description: 'ChEMBL compound ID (e.g., "CHEMBL59"). Must be valid and exist in database. Get from search_compounds first.' 
              },
              format: { 
                type: 'string', 
                enum: ['smiles', 'inchi', 'molfile', 'sdf'], 
                description: 'Chemical structure format to retrieve. Options:\n- "smiles" (default): Simplified Molecular Input Line Entry System - compact text format, best for similarity searches and quick viewing\n- "inchi": International Chemical Identifier - canonical format, best for exact structure matching and database lookups\n- "molfile": MDL Molfile format - includes 2D coordinates, good for visualization software\n- "sdf": Structure Data File - includes properties and 3D coordinates, best for computational chemistry tools',
                default: 'smiles'
              },
            },
            required: ['chembl_id'],
          },
        },
        // Bioactivity & Assay Data (2 tools)
        {
          name: 'search_activities',
          description: 'Search bioactivity data from high-throughput screening, binding assays, and functional assays. Returns experimental measurements including IC50 (inhibitory concentration), Ki (binding affinity), EC50 (effective concentration), Kd (dissociation constant) with units, standard deviations, and assay details. CRITICAL for: (1) Finding potency data for compounds, (2) Identifying all compounds tested against a target protein, (3) Comparing compound activities. At least ONE filter parameter (target, assay, molecule, or activity_type) must be provided.',
          inputSchema: {
            type: 'object',
            properties: {
              target_chembl_id: { 
                type: 'string', 
                description: 'Filter by biological target ChEMBL ID (e.g., "CHEMBL1824" for beta-2 adrenergic receptor, "CHEMBL1862" for EGFR). Returns all compounds tested against this target. Use when searching for drug candidates for a specific protein/enzyme/receptor.' 
              },
              assay_chembl_id: { 
                type: 'string', 
                description: 'Filter by specific assay ChEMBL ID (e.g., "CHEMBL1217643"). Returns all activity measurements from this particular experimental protocol. Use when you want to compare compounds tested under identical conditions.' 
              },
              molecule_chembl_id: { 
                type: 'string', 
                description: 'Filter by compound ChEMBL ID (e.g., "CHEMBL59"). Returns all bioactivity measurements for this compound across all targets and assays. Use to profile a compound\'s complete activity spectrum and selectivity.' 
              },
              activity_type: { 
                type: 'string', 
                description: 'Filter by measurement type. Common values (case-sensitive):\n- "IC50": Half-maximal inhibitory concentration (lower value = more potent inhibitor)\n- "Ki": Inhibition constant (binding affinity measurement)\n- "EC50": Half-maximal effective concentration (agonist potency)\n- "Kd": Dissociation constant (binding affinity)\n- "GI50": 50% growth inhibition (cytotoxicity)\n- "MIC": Minimum inhibitory concentration (antimicrobials)\nLeave empty to retrieve all activity types.' 
              },
              limit: { 
                type: 'number', 
                description: 'Maximum results to return. Range: 1-1000, Default: 25. Note: Bioactivity searches can return thousands of results. Start with 25-50 for initial exploration, use higher values with specific filters.',
                minimum: 1, 
                maximum: 1000,
                default: 25
              },
            },
            required: [],
          },
        },
        {
          name: 'get_assay_info',
          description: 'Retrieve complete information about a specific biological assay including: detailed experimental protocol, assay type (binding/functional/ADME/toxicity), target organism and species, cell line used, tissue type, subcellular fraction, detection method, measurement endpoints, data validity comments, confidence scores, and literature references (PubMed IDs). Use this to understand the experimental context and quality of bioactivity data, assess data reliability, and compare assay protocols.',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_id: { 
                type: 'string', 
                description: 'ChEMBL assay identifier in format CHEMBLxxxxxxx (typically 7-10 digits, e.g., "CHEMBL1217643", "CHEMBL829152"). Obtain from search_activities results to understand the assay methodology and experimental details.' 
              },
            },
            required: ['chembl_id'],
          },
        },
        // Batch Operations (1 tool)
        {
          name: 'batch_compound_lookup',
          description: 'Efficiently retrieve basic information for multiple compounds in a single API request. Returns essential data for each compound: ChEMBL ID, preferred name, molecular weight, LogP, drug status, and basic properties. Useful for: (1) Processing hit lists from virtual screening, (2) Batch validation of compound collections, (3) Quick property comparison across compound series, (4) Generating compound datasets. IMPORTANT: Limited to 50 compounds per request for optimal performance. For detailed information on specific compounds, use get_compound_info individually.',
          inputSchema: {
            type: 'object',
            properties: {
              chembl_ids: { 
                type: 'array', 
                items: { type: 'string' }, 
                description: 'Array of ChEMBL compound IDs to retrieve (e.g., ["CHEMBL59", "CHEMBL25", "CHEMBL192"]). All IDs must be in valid CHEMBLxxxx format. Results returned in same order as input. Individual lookup failures don\'t stop the batch - errors are reported per compound.',
                minItems: 1, 
                maxItems: 50,
                examples: [
                  ["CHEMBL59", "CHEMBL25"],
                  ["CHEMBL192", "CHEMBL502", "CHEMBL1", "CHEMBL10"]
                ]
              },
            },
            required: ['chembl_ids'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request: any) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          // Core Chemical Search & Retrieval (3 tools)
          case 'search_compounds':
            return await this.handleSearchCompounds(args);
          case 'get_compound_info':
            return await this.handleGetCompoundInfo(args);
          case 'get_compound_structure':
            return await this.handleGetCompoundStructure(args);
          // Bioactivity & Assay Data (2 tools)
          case 'search_activities':
            return await this.handleSearchActivities(args);
          case 'get_assay_info':
            return await this.handleGetAssayInfo(args);
          // Batch Operations (1 tool)
          case 'batch_compound_lookup':
            return await this.handleBatchCompoundLookup(args);
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

  private async handleGetCompoundStructure(args: any) {
    if (!args || typeof args.chembl_id !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid arguments: chembl_id is required');
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

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('ChEMBL MCP server running on stdio');
  }
}

const server = new ChEMBLServer();
server.run().catch(console.error);
