#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
CallToolRequestSchema,
ErrorCode,
ListResourceTemplatesRequestSchema,
ListToolsRequestSchema,
McpError,
ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import axios, {
AxiosError,
AxiosInstance,
AxiosRequestConfig,
} from 'axios';

/**

* ============================================================
* Open Targets MCP Server
* ============================================================
* 
* Design goals:
* 
* 1. Keep tool responses compact for LLM context efficiency.
* 2. Apply pagination in GraphQL BEFORE data is returned.
* 3. Never fetch thousands of associations and slice locally.
* 4. Retry transient network/API failures.
* 5. Return structured, predictable MCP responses.
* 
* GraphQL endpoint:
* https://api.platform.opentargets.org/api/v4/graphql
* ============================================================
  */

/* ============================================================

* Configuration
* ============================================================
  */

const GRAPHQL_URL =
'https://api.platform.opentargets.org/api/v4/graphql';

const DEFAULT_SEARCH_LIMIT = 4;
const DEFAULT_ASSOCIATION_LIMIT = 8;
const DEFAULT_DRUG_LIMIT = 8;

const MAX_SEARCH_LIMIT = 8;
const MAX_ASSOCIATION_LIMIT = 8;
const MAX_DRUG_LIMIT = 8;

const REQUEST_TIMEOUT_MS = 30_000;

const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 750;

/* ============================================================

* Types
* ============================================================
  */

type PaginationArgs = {
size?: number;
pageIndex?: number;
};

type SearchArgs = {
query: string;
size?: number;
format?: 'json' | 'tsv';
};

type AssociationArgs = PaginationArgs & {
targetId?: string;
diseaseId?: string;
minScore?: number;
};

type DiseaseTargetsArgs = PaginationArgs & {
diseaseId: string;
minScore?: number;
};

type DiseaseDrugsArgs = PaginationArgs & {
diseaseId: string;
};

type TargetDrugsArgs = PaginationArgs & {
targetId: string;
};

type IdArgs = {
id: string;
};

/* ============================================================

* Validation helpers
* ============================================================
  */

function isObject(value: unknown): value is Record<string, unknown> {
return typeof value === 'object' && value !== null;
}

function isNonEmptyString(value: unknown): value is string {
return (
typeof value === 'string' &&
value.trim().length > 0
);
}

function isValidSize(
value: unknown,
max: number
): value is number {
return (
typeof value === 'number' &&
Number.isInteger(value) &&
value > 0 &&
value <= max
);
}

function isValidPageIndex(
value: unknown
): value is number {
return (
typeof value === 'number' &&
Number.isInteger(value) &&
value >= 0
);
}

function isValidScore(
value: unknown
): value is number {
return (
typeof value === 'number' &&
value >= 0 &&
value <= 1
);
}

function isValidSearchArgs(
args: unknown
): args is SearchArgs {
if (!isObject(args)) return false;

if (!isNonEmptyString(args.query)) {
return false;
}

if (
args.size !== undefined &&
!isValidSize(args.size, MAX_SEARCH_LIMIT)
) {
return false;
}

if (
args.format !== undefined &&
args.format !== 'json' &&
args.format !== 'tsv'
) {
return false;
}

return true;
}

function isValidAssociationArgs(
args: unknown
): args is AssociationArgs {
if (!isObject(args)) return false;

const hasTarget =
args.targetId !== undefined;

const hasDisease =
args.diseaseId !== undefined;

if (!hasTarget && !hasDisease) {
return false;
}

if (
args.targetId !== undefined &&
!isNonEmptyString(args.targetId)
) {
return false;
}

if (
args.diseaseId !== undefined &&
!isNonEmptyString(args.diseaseId)
) {
return false;
}

if (
args.minScore !== undefined &&
!isValidScore(args.minScore)
) {
return false;
}

if (
args.size !== undefined &&
!isValidSize(args.size, MAX_ASSOCIATION_LIMIT)
) {
return false;
}

if (
args.pageIndex !== undefined &&
!isValidPageIndex(args.pageIndex)
) {
return false;
}

return true;
}

function isValidDiseaseTargetsArgs(
args: unknown
): args is DiseaseTargetsArgs {
if (!isObject(args)) return false;

if (!isNonEmptyString(args.diseaseId)) {
return false;
}

if (
args.minScore !== undefined &&
!isValidScore(args.minScore)
) {
return false;
}

if (
args.size !== undefined &&
!isValidSize(args.size, MAX_ASSOCIATION_LIMIT)
) {
return false;
}

if (
args.pageIndex !== undefined &&
!isValidPageIndex(args.pageIndex)
) {
return false;
}

return true;
}

function isValidDiseaseDrugsArgs(
args: unknown
): args is DiseaseDrugsArgs {
if (!isObject(args)) return false;

if (!isNonEmptyString(args.diseaseId)) {
return false;
}

if (
args.size !== undefined &&
!isValidSize(args.size, MAX_DRUG_LIMIT)
) {
return false;
}

if (
args.pageIndex !== undefined &&
!isValidPageIndex(args.pageIndex)
) {
return false;
}

return true;
}

function isValidTargetDrugsArgs(
args: unknown
): args is TargetDrugsArgs {
if (!isObject(args)) return false;

if (!isNonEmptyString(args.targetId)) {
return false;
}

if (
args.size !== undefined &&
!isValidSize(args.size, MAX_DRUG_LIMIT)
) {
return false;
}

if (
args.pageIndex !== undefined &&
!isValidPageIndex(args.pageIndex)
) {
return false;
}

return true;
}

function isValidIdArgs(
args: unknown
): args is IdArgs {
return (
isObject(args) &&
isNonEmptyString(args.id)
);
}

/* ============================================================

* Utility helpers
* ============================================================
  */

function getSize(
requested: number | undefined,
defaultValue: number,
maxValue: number
): number {
return Math.min(
requested ?? defaultValue,
maxValue
);
}

function getPageIndex(
requested: number | undefined
): number {
return requested ?? 0;
}

function sleep(ms: number): Promise<void> {
return new Promise(resolve =>
setTimeout(resolve, ms)
);
}

function isRetryableError(error: unknown): boolean {
if (axios.isAxiosError(error)) {
const status = error.response?.status;

// Retry network failures where there is no response.
if (!status) {
  return true;
}

// Retry temporary server-side failures.
if (status >= 500) {
  return true;
}

if (status === 429) {
  return true;
}

}

return false;
}

function getErrorMessage(error: unknown): string {
if (axios.isAxiosError(error)) {
const axiosError = error as AxiosError<any>;

if (axiosError.response?.data?.errors) {
  return JSON.stringify(
    axiosError.response.data.errors
  );
}

if (axiosError.response?.data) {
  return JSON.stringify(
    axiosError.response.data
  );
}

return axiosError.message;

}

if (error instanceof Error) {
return error.message;
}

return 'Unknown error';
}

/* ============================================================

* OpenTargetsServer
* ============================================================
  */

class OpenTargetsServer {
private server: Server;
private graphqlClient: AxiosInstance;

constructor() {
this.server = new Server(
{
name: 'opentargets-server',
version: '0.2.0',
},
{
capabilities: {
resources: {},
tools: {},
},
}
);

this.graphqlClient = axios.create({
  baseURL: GRAPHQL_URL,
  timeout: REQUEST_TIMEOUT_MS,
  headers: {
    'User-Agent': 'OpenTargets-MCP-Server/0.2.0',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

this.setupResourceHandlers();
this.setupToolHandlers();

this.server.onerror = (
  error: Error
) => {
  console.error('[MCP Error]', error);
};

process.on(
  'SIGINT',
  async () => {
    console.error(
      'Shutting down Open Targets MCP server...'
    );

    await this.server.close();
    process.exit(0);
  }
);

}

/* ==========================================================

* HTTP / GraphQL helpers
* ==========================================================
  */

private async graphqlRequest<T>(
query: string,
variables: Record<string, unknown>
): Promise<T> {

let lastError: unknown;

for (
  let attempt = 0;
  attempt < MAX_RETRIES;
  attempt++
) {
  try {
    const response =
      await this.graphqlClient.post(
        '',
        {
          query,
          variables,
        }
      );

    if (
      response.data?.errors &&
      Array.isArray(response.data.errors) &&
      response.data.errors.length > 0
    ) {
      throw new Error(
        `GraphQL errors: ${JSON.stringify(
          response.data.errors
        )}`
      );
    }

    return response.data.data as T;

  } catch (error) {
    lastError = error;

    const isLastAttempt =
      attempt === MAX_RETRIES - 1;

    if (
      isLastAttempt ||
      !isRetryableError(error)
    ) {
      throw error;
    }

    const delay =
      RETRY_BASE_DELAY_MS *
      Math.pow(2, attempt);

    console.error(
      `[OpenTargets] Request failed. ` +
      `Retry ${attempt + 1}/${MAX_RETRIES - 1} ` +
      `in ${delay}ms: ` +
      getErrorMessage(error)
    );

    await sleep(delay);
  }
}

throw lastError;

}

private textResponse(
data: unknown
) {
return {
content: [
{
type: 'text',
text: JSON.stringify(data, null, 2),
},
],
};
}

private errorResponse(
message: string,
error: unknown
) {
return {
content: [
{
type: 'text',
text: "${message}: ${getErrorMessage(error)}",
},
],
isError: true,
};
}

/* ==========================================================

* Resources
* ==========================================================
  */

private setupResourceHandlers() {

this.server.setRequestHandler(
  ListResourceTemplatesRequestSchema,
  async () => ({
    resourceTemplates: [
      {
        uriTemplate:
          'opentargets://target/{id}',
        name:
          'Open Targets target information',
        mimeType:
          'application/json',
        description:
          'Compact target information for an Ensembl gene ID',
      },

      {
        uriTemplate:
          'opentargets://disease/{id}',
        name:
          'Open Targets disease information',
        mimeType:
          'application/json',
        description:
          'Compact disease information for an Open Targets disease identifier',
      },
    ],
  })
);


this.server.setRequestHandler(
  ReadResourceRequestSchema,
  async (request: any) => {

    const uri =
      request.params.uri;

    const targetMatch =
      uri.match(
        /^opentargets:\/\/target\/([^/]+)$/
      );

    if (targetMatch) {
      const targetId =
        decodeURIComponent(targetMatch[1]);

      const result =
        await this.fetchTargetDetails(
          targetId
        );

      return {
        contents: [
          {
            uri,
            mimeType:
              'application/json',
            text:
              JSON.stringify(
                result,
                null,
                2
              ),
          },
        ],
      };
    }


    const diseaseMatch =
      uri.match(
        /^opentargets:\/\/disease\/([^/]+)$/
      );

    if (diseaseMatch) {
      const diseaseId =
        decodeURIComponent(
          diseaseMatch[1]
        );

      const result =
        await this.fetchDiseaseDetails(
          diseaseId
        );

      return {
        contents: [
          {
            uri,
            mimeType:
              'application/json',
            text:
              JSON.stringify(
                result,
                null,
                2
              ),
          },
        ],
      };
    }


    throw new McpError(
      ErrorCode.InvalidRequest,
      `Invalid URI format: ${uri}`
    );
  }
);

}

/* ==========================================================

* Tools
* ==========================================================
  */

private setupToolHandlers() {

this.server.setRequestHandler(
  ListToolsRequestSchema,
  async () => ({
    tools: [

      /* -------------------------
       * Search targets
       * ------------------------- */

      {
        name: 'search_targets',

        description:
          'Search Open Targets for therapeutic targets. Returns compact results and canonical Ensembl identifiers.',

        inputSchema: {
          type: 'object',

          properties: {
            query: {
              type: 'string',
              description:
                'Gene symbol, target name, or target description',
            },

            size: {
              type: 'integer',
              minimum: 1,
              maximum:
                MAX_SEARCH_LIMIT,

              description:
                `Default ${DEFAULT_SEARCH_LIMIT}, maximum ${MAX_SEARCH_LIMIT}`,
            },
          },

          required:
            ['query'],
        },
      },


      /* -------------------------
       * Search diseases
       * ------------------------- */

      {
        name: 'search_diseases',

        description:
          'Search Open Targets for diseases and phenotypes. Returns the canonical identifier returned by the Platform, which may be EFO or MONDO.',

        inputSchema: {
          type: 'object',

          properties: {
            query: {
              type: 'string',
              description:
                'Disease name, synonym, or phenotype',
            },

            size: {
              type: 'integer',
              minimum: 1,
              maximum:
                MAX_SEARCH_LIMIT,

              description:
                `Default ${DEFAULT_SEARCH_LIMIT}, maximum ${MAX_SEARCH_LIMIT}`,
            },
          },

          required:
            ['query'],
        },
      },


      /* -------------------------
       * Associations
       * ------------------------- */

      {
        name:
          'get_target_disease_associations',

        description:
          'Get compact, paginated target-disease associations. Provide targetId or diseaseId. Never returns an unbounded association list.',

        inputSchema: {
          type:
            'object',

          properties: {

            targetId: {
              type:
                'string',

              description:
                'Target Ensembl gene ID',
            },

            diseaseId: {
              type:
                'string',

              description:
                'Open Targets disease identifier',
            },

            minScore: {
              type:
                'number',

              minimum: 0,
              maximum: 1,

              description:
                'Optional minimum association score',
            },

            size: {
              type:
                'integer',

              minimum: 1,
              maximum:
                MAX_ASSOCIATION_LIMIT,
            },

            pageIndex: {
              type:
                'integer',

              minimum: 0,

              description:
                'Zero-based page index',
            },
          },
        },
      },


      /* -------------------------
       * Disease targets
       * ------------------------- */

      {
        name:
          'get_disease_targets_summary',

        description:
          'Get top disease-associated targets using server-side GraphQL pagination. Returns total count plus only the requested page.',

        inputSchema: {
          type:
            'object',

          properties: {

            diseaseId: {
              type:
                'string',

              description:
                'Open Targets disease identifier returned by search_diseases',
            },

            minScore: {
              type:
                'number',

              minimum: 0,
              maximum: 1,
            },

            size: {
              type:
                'integer',

              minimum: 1,
              maximum:
                MAX_ASSOCIATION_LIMIT,
            },

            pageIndex: {
              type:
                'integer',

              minimum: 0,
            },
          },

          required:
            ['diseaseId'],
        },
      },


      /* -------------------------
       * Disease drugs
       * ------------------------- */

      {
        name:
          'get_disease_drugs',

        description:
          'Get approved drugs and clinical candidates associated with a disease using paginated Open Targets data.',

        inputSchema: {
          type:
            'object',

          properties: {

            diseaseId: {
              type:
                'string',

              description:
                'Open Targets disease identifier',
            },

            size: {
              type:
                'integer',

              minimum: 1,
              maximum:
                MAX_DRUG_LIMIT,
            },

            pageIndex: {
              type:
                'integer',

              minimum: 0,
            },
          },

          required:
            ['diseaseId'],
        },
      },


      /* -------------------------
       * Target drugs
       * ------------------------- */

      {
        name:
          'get_target_drugs',

        description:
          'Get known drugs and clinical candidates that act on a target. Results are paginated and compact.',

        inputSchema: {
          type:
            'object',

          properties: {

            targetId: {
              type:
                'string',

              description:
                'Target Ensembl gene ID',
            },

            size: {
              type:
                'integer',

              minimum: 1,
              maximum:
                MAX_DRUG_LIMIT,
            },

            pageIndex: {
              type:
                'integer',

              minimum: 0,
            },
          },

          required:
            ['targetId'],
        },
      },


      /* -------------------------
       * Details
       * ------------------------- */

      {
        name:
          'get_target_details',

        description:
          'Get compact target details.',

        inputSchema: {
          type:
            'object',

          properties: {
            id: {
              type:
                'string',
            },
          },

          required:
            ['id'],
        },
      },


      {
        name:
          'get_disease_details',

        description:
          'Get compact disease details.',

        inputSchema: {
          type:
            'object',

          properties: {
            id: {
              type:
                'string',
            },
          },

          required:
            ['id'],
        },
      },
    ],
  })
);


this.server.setRequestHandler(
  CallToolRequestSchema,

  async (request: any) => {

    const {
      name,
      arguments: args,
    } = request.params;


    switch (name) {

      case 'search_targets':
        return this.handleSearchTargets(
          args
        );

      case 'search_diseases':
        return this.handleSearchDiseases(
          args
        );

      case 'get_target_disease_associations':
        return this.handleGetTargetDiseaseAssociations(
          args
        );

      case 'get_disease_targets_summary':
        return this.handleGetDiseaseTargetsSummary(
          args
        );

      case 'get_disease_drugs':
        return this.handleGetDiseaseDrugs(
          args
        );

      case 'get_target_drugs':
        return this.handleGetTargetDrugs(
          args
        );

      case 'get_target_details':
        return this.handleGetTargetDetails(
          args
        );

      case 'get_disease_details':
        return this.handleGetDiseaseDetails(
          args
        );

      default:
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${name}`
        );
    }
  }
);

}

/* ==========================================================

* Search targets
* ==========================================================
  */

private async handleSearchTargets(
args: unknown
) {

if (!isValidSearchArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Invalid target search arguments'
  );
}

try {

  const size =
    getSize(
      args.size,
      DEFAULT_SEARCH_LIMIT,
      MAX_SEARCH_LIMIT
    );

  const query = `
    query SearchTargets(
      $queryString: String!
      $pageSize: Int!
    ) {
      search(
        queryString: $queryString
        entityNames: ["target"]
        page: {
          index: 0
          size: $pageSize
        }
      ) {
        total
        hits {
          id
          name
          description
          entity
        }
      }
    }
  `;


  const data =
    await this.graphqlRequest<{
      search: {
        total: number;
        hits: Array<{
          id: string;
          name: string;
          description?: string;
          entity: string;
        }>;
      };
    }>(
      query,
      {
        queryString:
          args.query,
        pageSize:
          size,
      }
    );


  return this.textResponse({
    query:
      args.query,

    total:
      data.search.total,

    returned:
      data.search.hits.length,

    results:
      data.search.hits,
  });

} catch (error) {
  return this.errorResponse(
    'Error searching targets',
    error
  );
}

}

/* ==========================================================

* Search diseases
* ==========================================================
  */

private async handleSearchDiseases(
args: unknown
) {

if (!isValidSearchArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Invalid disease search arguments'
  );
}

try {

  const size =
    getSize(
      args.size,
      DEFAULT_SEARCH_LIMIT,
      MAX_SEARCH_LIMIT
    );

  const query = `
    query SearchDiseases(
      $queryString: String!
      $pageSize: Int!
    ) {
      search(
        queryString: $queryString
        entityNames: ["disease"]
        page: {
          index: 0
          size: $pageSize
        }
      ) {
        total
        hits {
          id
          name
          description
          entity
        }
      }
    }
  `;


  const data =
    await this.graphqlRequest<{
      search: {
        total: number;
        hits: Array<{
          id: string;
          name: string;
          description?: string;
          entity: string;
        }>;
      };
    }>(
      query,
      {
        queryString:
          args.query,
        pageSize:
          size,
      }
    );


  return this.textResponse({
    query:
      args.query,

    total:
      data.search.total,

    returned:
      data.search.hits.length,

    results:
      data.search.hits,
  });

} catch (error) {
  return this.errorResponse(
    'Error searching diseases',
    error
  );
}

}

/* ==========================================================

* Target / Disease associations
* ==========================================================
  */

private async handleGetTargetDiseaseAssociations(
args: unknown
) {

if (!isValidAssociationArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Invalid association arguments'
  );
}


try {

  const size =
    getSize(
      args.size,
      DEFAULT_ASSOCIATION_LIMIT,
      MAX_ASSOCIATION_LIMIT
    );

  const pageIndex =
    getPageIndex(
      args.pageIndex
    );


  /*
   * Target -> diseases
   */

  if (
    args.targetId &&
    !args.diseaseId
  ) {

    const query = `
      query GetTargetAssociations(
        $ensemblId: String!
        $pageIndex: Int!
        $pageSize: Int!
      ) {
        target(
          ensemblId: $ensemblId
        ) {
          id
          approvedSymbol
          approvedName

          associatedDiseases(
            page: {
              index: $pageIndex
              size: $pageSize
            }
          ) {
            count

            rows {
              score

              disease {
                id
                name
              }
            }
          }
        }
      }
    `;


    const data =
      await this.graphqlRequest<any>(
        query,
        {
          ensemblId:
            args.targetId,

          pageIndex,
          pageSize:
            size,
        }
      );


    const target =
      data.target;

    if (!target) {
      return this.textResponse({
        message:
          'Target not found',
        targetId:
          args.targetId,
      });
    }


    let rows =
      target.associatedDiseases?.rows ?? [];


    if (
      args.minScore !== undefined
    ) {
      rows =
        rows.filter(
          (row: any) =>
            row.score >=
            args.minScore
        );
    }


    return this.textResponse({
      entityType:
        'target',

      target: {
        id:
          target.id,

        symbol:
          target.approvedSymbol,

        name:
          target.approvedName,
      },

      totalAssociations:
        target.associatedDiseases?.count ?? 0,

      pageIndex,

      requestedSize:
        size,

      returned:
        rows.length,

      associations:
        rows.map(
          (row: any) => ({
            diseaseId:
              row.disease.id,

            diseaseName:
              row.disease.name,

            associationScore:
              row.score,
          })
        ),
    });
  }


  /*
   * Disease -> targets
   */

  if (
    args.diseaseId &&
    !args.targetId
  ) {

    const query = `
      query GetDiseaseAssociations(
        $efoId: String!
        $pageIndex: Int!
        $pageSize: Int!
      ) {
        disease(
          efoId: $efoId
        ) {
          id
          name

          associatedTargets(
            page: {
              index: $pageIndex
              size: $pageSize
            }
          ) {
            count

            rows {
              score

              target {
                id
                approvedSymbol
                approvedName
              }
            }
          }
        }
      }
    `;


    const data =
      await this.graphqlRequest<any>(
        query,
        {
          efoId:
            args.diseaseId,

          pageIndex,

          pageSize:
            size,
        }
      );


    const disease =
      data.disease;

    if (!disease) {
      return this.textResponse({
        message:
          'Disease not found',

        diseaseId:
          args.diseaseId,
      });
    }


    let rows =
      disease.associatedTargets?.rows ?? [];


    if (
      args.minScore !== undefined
    ) {
      rows =
        rows.filter(
          (row: any) =>
            row.score >=
            args.minScore
        );
    }


    return this.textResponse({
      entityType:
        'disease',

      disease: {
        id:
          disease.id,

        name:
          disease.name,
      },

      totalAssociations:
        disease.associatedTargets?.count ?? 0,

      pageIndex,

      requestedSize:
        size,

      returned:
        rows.length,

      associations:
        rows.map(
          (row: any) => ({
            targetId:
              row.target.id,

            targetSymbol:
              row.target.approvedSymbol,

            targetName:
              row.target.approvedName,

            associationScore:
              row.score,
          })
        ),
    });
  }


  /*
   * Both IDs supplied.
   *
   * Instead of pretending that pair lookup is implemented,
   * we explicitly return a compact message.
   */

  return this.textResponse({
    message:
      'Direct pair-specific lookup is not implemented by this compact MCP tool.',

    targetId:
      args.targetId,

    diseaseId:
      args.diseaseId,

    suggestion:
      'Use get_target_disease_associations with one identifier, or add a dedicated pair evidence query.',
  });

} catch (error) {
  return this.errorResponse(
    'Error getting target-disease associations',
    error
  );
}

}

/* ==========================================================

* Disease targets summary
* ==========================================================
  */

private async handleGetDiseaseTargetsSummary(
args: unknown
) {

if (!isValidDiseaseTargetsArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Invalid disease targets summary arguments'
  );
}


try {

  const size =
    getSize(
      args.size,
      DEFAULT_ASSOCIATION_LIMIT,
      MAX_ASSOCIATION_LIMIT
    );

  const pageIndex =
    getPageIndex(
      args.pageIndex
    );


  const query = `
    query GetDiseaseTargetsSummary(
      $efoId: String!
      $pageIndex: Int!
      $pageSize: Int!
    ) {
      disease(
        efoId: $efoId
      ) {
        id
        name

        associatedTargets(
          page: {
            index: $pageIndex
            size: $pageSize
          }
        ) {
          count

          rows {
            score

            target {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
  `;


  const data =
    await this.graphqlRequest<any>(
      query,
      {
        efoId:
          args.diseaseId,

        pageIndex,

        pageSize:
          size,
      }
    );


  const disease =
    data.disease;


  if (!disease) {
    return this.textResponse({
      message:
        'Disease not found',

      diseaseId:
        args.diseaseId,
    });
  }


  let rows =
    disease.associatedTargets?.rows ?? [];


  /*
   * minScore is applied only to the already compact page.
   * This intentionally avoids fetching the complete dataset.
   */

  if (
    args.minScore !== undefined
  ) {
    rows =
      rows.filter(
        (row: any) =>
          row.score >=
          args.minScore
      );
  }


  return this.textResponse({
    disease: {
      id:
        disease.id,

      name:
        disease.name,
    },

    totalAssociatedTargets:
      disease.associatedTargets?.count ?? 0,

    pageIndex,

    requestedSize:
      size,

    returned:
      rows.length,

    minScore:
      args.minScore ?? null,

    targets:
      rows.map(
        (row: any) => ({
          targetId:
            row.target.id,

          targetSymbol:
            row.target.approvedSymbol,

          targetName:
            row.target.approvedName,

          associationScore:
            row.score,
        })
      ),
  });

} catch (error) {
  return this.errorResponse(
    'Error getting disease targets summary',
    error
  );
}

}

/* ==========================================================

* Disease drugs / clinical candidates
* ==========================================================
  */

private async handleGetDiseaseDrugs(
args: unknown
) {

if (!isValidDiseaseDrugsArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Invalid disease drugs arguments'
  );
}


try {

  const size =
    getSize(
      args.size,
      DEFAULT_DRUG_LIMIT,
      MAX_DRUG_LIMIT
    );

  const pageIndex =
    getPageIndex(
      args.pageIndex
    );


  /*
   * drugAndClinicalCandidates is the compact disease-centric
   * route for approved drugs and clinical candidates.
   */

  const query = `
    query GetDiseaseDrugs(
      $efoId: String!
      $pageIndex: Int!
      $pageSize: Int!
    ) {
      disease(
        efoId: $efoId
      ) {
        id
        name

        drugAndClinicalCandidates {
          count

          rows {
            maxClinicalStage

            drug {
              id
              name
              drugType
              maximumClinicalStage
            }
          }
        }
      }
    }
  `;


  const data =
    await this.graphqlRequest<any>(
      query,
      {
        efoId:
          args.diseaseId,

        pageIndex,

        pageSize:
          size,
      }
    );


  const disease =
    data.disease;


  if (!disease) {
    return this.textResponse({
      message:
        'Disease not found',

      diseaseId:
        args.diseaseId,
    });
  }


  /*
   * Some Open Targets schema versions expose pagination
   * differently for drugAndClinicalCandidates.
   *
   * We always keep the MCP output compact even if the
   * API response contains more rows.
   */

  const allRows =
    disease
      .drugAndClinicalCandidates
      ?.rows ?? [];


  const start =
    pageIndex * size;

  const rows =
    allRows.slice(
      start,
      start + size
    );


  return this.textResponse({
    disease: {
      id:
        disease.id,

      name:
        disease.name,
    },

    totalCandidates:
      disease
        .drugAndClinicalCandidates
        ?.count ?? 0,

    pageIndex,

    requestedSize:
      size,

    returned:
      rows.length,

    candidates:
      rows.map(
        (row: any) => ({
          clinicalStage:
            row.maxClinicalStage ??
            row.drug
              ?.maximumClinicalStage ??
            null,

          drug: row.drug
            ? {
                id:
                  row.drug.id,

                name:
                  row.drug.name,

                type:
                  row.drug.drugType,

                maximumClinicalStage:
                  row.drug.maximumClinicalStage,
              }
            : null,
        })
      ),
  });

} catch (error) {
  return this.errorResponse(
    'Error getting disease drugs',
    error
  );
}

}

/* ==========================================================

* Target drugs
* ==========================================================
  */

private async handleGetTargetDrugs(
args: unknown
) {

if (!isValidTargetDrugsArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Invalid target drugs arguments'
  );
}


try {

  const size =
    getSize(
      args.size,
      DEFAULT_DRUG_LIMIT,
      MAX_DRUG_LIMIT
    );

  const pageIndex =
    getPageIndex(
      args.pageIndex
    );


  /*
   * knownDrugs is target-centric and contains drugs
   * with mechanism information.
   */

  const query = `
    query GetTargetDrugs(
      $ensemblId: String!
      $pageIndex: Int!
      $pageSize: Int!
    ) {
      target(
        ensemblId: $ensemblId
      ) {
        id
        approvedSymbol
        approvedName

        knownDrugs {
          count

          rows {
            phase

            drug {
              id
              name
              drugType
              maximumClinicalStage
            }

            mechanismOfAction

            disease {
              id
              name
            }
          }
        }
      }
    }
  `;


  const data =
    await this.graphqlRequest<any>(
      query,
      {
        ensemblId:
          args.targetId,

        pageIndex,

        pageSize:
          size,
      }
    );


  const target =
    data.target;


  if (!target) {
    return this.textResponse({
      message:
        'Target not found',

      targetId:
        args.targetId,
    });
  }


  const allRows =
    target.knownDrugs?.rows ?? [];


  const start =
    pageIndex * size;

  const rows =
    allRows.slice(
      start,
      start + size
    );


  return this.textResponse({
    target: {
      id:
        target.id,

      symbol:
        target.approvedSymbol,

      name:
        target.approvedName,
    },

    totalKnownDrugs:
      target.knownDrugs?.count ?? 0,

    pageIndex,

    requestedSize:
      size,

    returned:
      rows.length,

    drugs:
      rows.map(
        (row: any) => ({
          phase:
            row.phase ?? null,

          mechanismOfAction:
            row.mechanismOfAction ??
            null,

          drug: row.drug
            ? {
                id:
                  row.drug.id,

                name:
                  row.drug.name,

                type:
                  row.drug.drugType,

                maximumClinicalStage:
                  row.drug.maximumClinicalStage,
              }
            : null,

          disease:
            row.disease
              ? {
                  id:
                    row.disease.id,

                  name:
                    row.disease.name,
                }
              : null,
        })
      ),
  });

} catch (error) {
  return this.errorResponse(
    'Error getting target drugs',
    error
  );
}

}

/* ==========================================================

* Target details
* ==========================================================
  */

private async fetchTargetDetails(
targetId: string
) {

const query = `
  query GetTarget(
    $ensemblId: String!
  ) {
    target(
      ensemblId: $ensemblId
    ) {
      id
      approvedName
      approvedSymbol
      biotype
    }
  }
`;


const data =
  await this.graphqlRequest<any>(
    query,
    {
      ensemblId:
        targetId,
    }
  );


return {
  target:
    data.target ?? null,
};

}

private async handleGetTargetDetails(
args: unknown
) {

if (!isValidIdArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Target ID is required'
  );
}


try {

  const result =
    await this.fetchTargetDetails(
      args.id
    );

  return this.textResponse(
    result
  );

} catch (error) {
  return this.errorResponse(
    'Error getting target details',
    error
  );
}

}

/* ==========================================================

* Disease details
* ==========================================================
  */

private async fetchDiseaseDetails(
diseaseId: string
) {

const query = `
  query GetDisease(
    $efoId: String!
  ) {
    disease(
      efoId: $efoId
    ) {
      id
      name
      description
    }
  }
`;


const data =
  await this.graphqlRequest<any>(
    query,
    {
      efoId:
        diseaseId,
    }
  );


return {
  disease:
    data.disease ?? null,
};

}

private async handleGetDiseaseDetails(
args: unknown
) {

if (!isValidIdArgs(args)) {
  throw new McpError(
    ErrorCode.InvalidParams,
    'Disease ID is required'
  );
}


try {

  const result =
    await this.fetchDiseaseDetails(
      args.id
    );

  return this.textResponse(
    result
  );

} catch (error) {
  return this.errorResponse(
    'Error getting disease details',
    error
  );
}

}

/* ==========================================================

* Server startup
* ==========================================================
  */

async run() {

const transport =
  new StdioServerTransport();

await this.server.connect(
  transport
);

console.error(
  'Open Targets MCP server running on stdio'
);

}
}

/* ============================================================

* Main
* ============================================================
  */

const server =
new OpenTargetsServer();

server.run().catch(
error => {
console.error(
'Failed to start Open Targets MCP server:',
error
);

process.exit(1);

}
);
