# Data Directory

This directory stores persistent data for Sparkstation:

- `sparkstation.db` - SQLite database for model registry
- `litellm.db` - LiteLLM usage tracking (optional)

Both files are automatically created on first run.

## Backup

Recommended to backup this directory periodically:

```bash
tar -czf sparkstation-backup-$(date +%Y%m%d).tar.gz data/
```

## Reset

To reset all data:

```bash
rm -f data/*.db
```

Then restart the supervisor to reinitialize.
