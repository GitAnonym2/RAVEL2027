# Anonymous-release checklist

Before every push to the review repository:

- search text files for author names, affiliations, e-mail addresses, and local machine/user names;
- remove absolute home-directory paths from JSON/CSV/log outputs;
- exclude `__pycache__`, notebooks with local paths in outputs, checkpoints, and raw data caches;
- inspect PDF/PNG metadata if generated figures are committed;
- do not add `CITATION.cff`, author copyright lines, personal GitHub handles, or institutional URLs during double-blind review;
- use third-party repository links rather than copying external source trees unless redistribution is licensed.
