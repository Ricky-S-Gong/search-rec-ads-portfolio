export const algorithmDomains = ['search', 'ads', 'recommendation'] as const;

export type AlgorithmDomain = (typeof algorithmDomains)[number];

export function parseAlgorithmDomain(value: string | null | undefined): AlgorithmDomain {
  return algorithmDomains.includes(value as AlgorithmDomain)
    ? (value as AlgorithmDomain)
    : 'recommendation';
}
