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
const isValidTargetSearchArgs = (args: any): args is { query: string; size?: number; format?: string } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.query === 'string' &&
    args.query.length > 0 &&
    (args.size === undefined || (typeof args.size === 'number' && args.size > 0 && args.size <= 500)) &&
    (args.format === undefined || ['json', 'tsv'].includes(args.format))
  );
};

const isValidDiseaseSearchArgs = (args: any): args is { query: string; size?: number; format?: string } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.query === 'string' &&
    args.query.length > 0 &&
    (args.size === undefined || (typeof args.size === 'number' && args.size > 0 && args.size <= 500)) &&
    (args.format === undefined || ['json', 'tsv'].includes(args.format))
  );
};

const isValidAssociationArgs = (args: any): args is { targetId?: string; diseaseId?: string; minScore?: number; size?: number } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    (args.targetId === undefined || typeof args.targetId === 'string') &&
    (args.diseaseId === undefined || typeof args.diseaseId === 'string') &&
    (args.minScore === undefined || (typeof args.minScore === 'number' && args.minScore >= 0 && args.minScore <= 1)) &&
    (args.size === undefined || (typeof args.size === 'number' && args.size > 0 && args.size <= 500)) &&
    (args.targetId !== undefined || args.diseaseId !== undefined)
  );
};

const isValidIdArgs = (args: any): args is { id: string } => {
  return (
    typeof args === 'object' &&
    args !== null &&
    typeof args.id === 'string' &&
    args.id.length > 0
  );
};

class OpenTargetsServer {
  private server: Server;
  private apiClient: AxiosInstance;
  private graphqlClient: AxiosInstance;

  constructor() {
    this.server = new Server(
      {
        name: 'opentargets-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize Open Targets REST API client
    this.apiClient = axios.create({
      baseURL: 'https://api.platform.opentargets.org/api/v4',
      timeout: 30000,
      headers: {
        'User-Agent': 'OpenTargets-MCP-Server/0.1.0',
        'Content-Type': 'application/json',
      },
    });

    // Initialize Open Targets GraphQL API client
    this.graphqlClient = axios.create({
      baseURL: 'https://api.platform.opentargets.org/api/v4/graphql',
      timeout: 30000,
      headers: {
        'User-Agent': 'OpenTargets-MCP-Server/0.1.0',
        'Content-Type': 'application/json',
      },
    });

    this.setupResourceHandlers();
    this.setupToolHandlers();

    // Error handling
    this.server.onerror = (error: Error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupResourceHandlers() {
    this.server.setRequestHandler(
      ListResourceTemplatesRequestSchema,
      async () => ({
        resourceTemplates: [
          {
            uriTemplate: 'opentargets://target/{id}',
            name: 'Open Targets target information',
            mimeType: 'application/json',
            description: 'Complete target information for an Ensembl gene ID',
          },
          {
            uriTemplate: 'opentargets://disease/{id}',
            name: 'Open Targets disease information',
            mimeType: 'application/json',
            description: 'Complete disease information for an EFO ID',
          },
          {
            uriTemplate: 'opentargets://drug/{id}',
            name: 'Open Targets drug information',
            mimeType: 'application/json',
            description: 'Complete drug information for a ChEMBL ID',
          },
          {
            uriTemplate: 'opentargets://association/{targetId}/{diseaseId}',
            name: 'Target-disease association',
            mimeType: 'application/json',
            description: 'Target-disease association evidence and scoring',
          },
          {
            uriTemplate: 'opentargets://search/{query}',
            name: 'Search results',
            mimeType: 'application/json',
            description: 'Search results across targets, diseases, and drugs',
          },
        ],
      })
    );

    this.server.setRequestHandler(
      ReadResourceRequestSchema,
      async (request: any) => {
        const uri = request.params.uri;

        // Handle target info requests
        const targetMatch = uri.match(/^opentargets:\/\/target\/([A-Z0-9_]+)$/);
        if (targetMatch) {
          const targetId = targetMatch[1];
          try {
            const response = await this.apiClient.get(`/target/${targetId}`);
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
              `Failed to fetch target ${targetId}: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }

        // Handle disease info requests
        const diseaseMatch = uri.match(/^opentargets:\/\/disease\/([A-Z0-9_]+)$/);
        if (diseaseMatch) {
          const diseaseId = diseaseMatch[1];
          try {
            const response = await this.apiClient.get(`/disease/${diseaseId}`);
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
              `Failed to fetch disease ${diseaseId}: ${error instanceof Error ? error.message : 'Unknown error'}`
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
        {
          name: 'search_targets',
          description: 'Search for therapeutic targets by gene symbol, name, or description',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Search query (gene symbol, name, description)' },
              size: { type: 'number', description: 'Number of results to return (1-500, default: 25)', minimum: 1, maximum: 500 },
              format: { type: 'string', enum: ['json', 'tsv'], description: 'Output format (default: json)' },
            },
            required: ['query'],
          },
        },
        {
          name: 'search_diseases',
          description: 'Search for diseases by name, synonym, or description',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Search query (disease name, synonym, description)' },
              size: { type: 'number', description: 'Number of results to return (1-500, default: 25)', minimum: 1, maximum: 500 },
              format: { type: 'string', enum: ['json', 'tsv'], description: 'Output format (default: json)' },
            },
            required: ['query'],
          },
        },
        {
          name: 'get_target_disease_associations',
          description: 'Get target-disease associations with evidence scores',
          inputSchema: {
            type: 'object',
            properties: {
              targetId: { type: 'string', description: 'Target Ensembl gene ID' },
              diseaseId: { type: 'string', description: 'Disease EFO ID' },
              minScore: { type: 'number', description: 'Minimum association score (0-1)', minimum: 0, maximum: 1 },
              size: { type: 'number', description: 'Number of results to return (1-500, default: 25)', minimum: 1, maximum: 500 },
            },
            required: [],
          },
        },
        {
          name: 'get_disease_targets_summary',
          description: 'Get overview of all targets associated with a disease',
          inputSchema: {
            type: 'object',
            properties: {
              diseaseId: { type: 'string', description: 'Disease EFO ID' },
              minScore: { type: 'number', description: 'Minimum association score (0-1)', minimum: 0, maximum: 1 },
              size: { type: 'number', description: 'Number of targets to return (1-500, default: 50)', minimum: 1, maximum: 500 },
            },
            required: ['diseaseId'],
          },
        },
        {
          name: 'get_target_details',
          description: 'Get comprehensive target information',
          inputSchema: {
            type: 'object',
            properties: {
              id: { type: 'string', description: 'Target Ensembl gene ID' },
            },
            required: ['id'],
          },
        },
        {
          name: 'get_disease_details',
          description: 'Get comprehensive disease information',
          inputSchema: {
            type: 'object',
            properties: {
              id: { type: 'string', description: 'Disease EFO ID' },
            },
            required: ['id'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request: any) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'search_targets':
          return this.handleSearchTargets(args);
        case 'search_diseases':
          return this.handleSearchDiseases(args);
        case 'get_target_disease_associations':
          return this.handleGetTargetDiseaseAssociations(args);
        case 'get_disease_targets_summary':
          return this.handleGetDiseaseTargetsSummary(args);
        case 'get_target_details':
          return this.handleGetTargetDetails(args);
        case 'get_disease_details':
          return this.handleGetDiseaseDetails(args);
        default:
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${name}`
          );
      }
    });
  }

  private async handleSearchTargets(args: any) {
    if (!isValidTargetSearchArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid target search arguments');
    }

    try {
      const query = `
        query SearchTargets($queryString: String!) {
          search(queryString: $queryString, entityNames: ["target"]) {
            hits {
              id
              name
              description
              entity
            }
          }
        }
      `;

      const response = await this.graphqlClient.post('', {
        query,
        variables: {
          queryString: args.query
        }
      });

      // Limit results on client side
      const hits = response.data.data?.search?.hits || [];
      const limitedHits = hits.slice(0, args.size || 25);
      const result = {
        ...response.data,
        data: {
          search: {
            hits: limitedHits,
            total: hits.length
          }
        }
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error searching targets: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleSearchDiseases(args: any) {
    if (!isValidDiseaseSearchArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid disease search arguments');
    }

    try {
      const query = `
        query SearchDiseases($queryString: String!) {
          search(queryString: $queryString, entityNames: ["disease"]) {
            hits {
              id
              name
              description
              entity
            }
          }
        }
      `;

      const response = await this.graphqlClient.post('', {
        query,
        variables: {
          queryString: args.query
        }
      });

      // Limit results on client side
      const hits = response.data.data?.search?.hits || [];
      const limitedHits = hits.slice(0, args.size || 25);
      const result = {
        ...response.data,
        data: {
          search: {
            hits: limitedHits,
            total: hits.length
          }
        }
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error searching diseases: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleGetTargetDiseaseAssociations(args: any) {
    if (!isValidAssociationArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid association arguments');
    }

    try {
      // If only targetId provided, get associations for that target
      if (args.targetId && !args.diseaseId) {
        const query = `query GetTargetAssociations($ensemblId: String!) { target(ensemblId: $ensemblId) { id approvedSymbol associatedDiseases { rows { disease { id name } score } } } }`;

        const response = await this.graphqlClient.post('', {
          query,
          variables: {
            ensemblId: args.targetId,
            size: args.size || 25
          }
        });

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(response.data, null, 2),
            },
          ],
        };
      }

      // If only diseaseId provided, get associations for that disease
      else if (args.diseaseId && !args.targetId) {
        const query = `query GetDiseaseAssociations($efoId: String!) { disease(efoId: $efoId) { id name associatedTargets { rows { target { id approvedSymbol approvedName } score } } } }`;

        const response = await this.graphqlClient.post('', {
          query,
          variables: {
            efoId: args.diseaseId,
            size: args.size || 25
          }
        });

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(response.data, null, 2),
            },
          ],
        };
      }

      // If both provided, return the association between them
      else {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                message: "Specific target-disease pair association lookup not yet implemented",
                suggestion: "Use targetId OR diseaseId to get associations for that entity"
              }, null, 2),
            },
          ],
        };
      }
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error getting associations: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleGetDiseaseTargetsSummary(args: any) {
    if (!isValidIdArgs(args) && !args.diseaseId) {
      throw new McpError(ErrorCode.InvalidParams, 'Disease ID is required');
    }

    try {
      const diseaseId = args.diseaseId || args.id;

      const query = `query GetDiseaseTargetsSummary($efoId: String!) { disease(efoId: $efoId) { id name associatedTargets { count rows { target { id approvedSymbol approvedName } score } } } }`;

      const response = await this.graphqlClient.post('', {
        query,
        variables: {
          efoId: diseaseId,
          size: args.size || 50
        }
      });

      const diseaseData = response.data.data?.disease;
      const associations = diseaseData?.associatedTargets;

      const summary = {
        diseaseId,
        diseaseName: diseaseData?.name,
        totalTargets: associations?.count || 0,
        topTargets: associations?.rows?.slice(0, 10).map((assoc: any) => ({
          targetId: assoc.target.id,
          targetSymbol: assoc.target.approvedSymbol,
          targetName: assoc.target.approvedName,
          associationScore: assoc.score,
          datatypeScores: assoc.datatypeScores,
        })) || [],
        fullResults: response.data,
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(summary, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error getting disease targets summary: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleGetTargetDetails(args: any) {
    if (!isValidIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Target ID is required');
    }

    try {
      const query = `query GetTarget($ensemblId: String!) { target(ensemblId: $ensemblId) { id approvedName approvedSymbol biotype } }`;

      const response = await this.graphqlClient.post('', {
        query,
        variables: {
          ensemblId: args.id
        }
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
      return {
        content: [
          {
            type: 'text',
            text: `Error getting target details: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleGetDiseaseDetails(args: any) {
    if (!isValidIdArgs(args)) {
      throw new McpError(ErrorCode.InvalidParams, 'Disease ID is required');
    }

    try {
      const query = `query GetDisease($efoId: String!) { disease(efoId: $efoId) { id name description } }`;

      const response = await this.graphqlClient.post('', {
        query,
        variables: {
          efoId: args.id
        }
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
      return {
        content: [
          {
            type: 'text',
            text: `Error getting disease details: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Open Targets MCP server running on stdio');
  }
}

const server = new OpenTargetsServer();
server.run().catch(console.error);
