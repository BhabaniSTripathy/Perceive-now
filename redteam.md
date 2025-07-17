
| Failure Mode | Trigger | Auto-Remediation |

Corrupt source file | Bad upload or incomplete download | Fallback to last good snapshot in S3 |
Schema drift | New columns appear in dataset | Auto-version schema with dbt, alert developer |
Embedding API fails | Model not responding or rate-limited | Retry with backoff, fallback to older embeddings |
Vector index corrupt | FAISS index file lost or corrupted | Rebuild index from DuckDB data |
Poor data quality | Excess nulls, duplicates | Auto-clean & log anomalies, block pipeline run |

Each failure is handled by simple self-healing logic or clear escalation to a dev.
