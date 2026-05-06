Archive layout for ephemeris upload debugging

This folder stores upload payloads, metadata, and related debug artifacts.
Keep these files for traceability; rotate or delete older archives as needed.

Structure:
- tmp_uploads_archive_<timestamp>/  (moved archives; contains previous dumps)
- tmp_uploads/                      (active dumps created by client)

What to store in archives:
- <crash_id>_payload.bin           : raw gzipped msgpack/json payload
- <crash_id>_metadata.json         : metadata sent with the upload
- <crash_id>_decoded_sample.json   : small human-readable decoded sample
- curl_debug*.txt, ngrok responses  : optional curl/HTTP debug outputs

Retention/rotation suggestion:
- Keep last 30 days of archives; compress older ones and move to cold storage.
- Remove personal secrets from `env.py` before archival or exclude it from
  archives.

How the client writes dumps:
- `MISC/ephemeris_client.py` writes dumps to `data/archives/tmp_uploads`.

If you'd like, I can add a small cleanup script to compress and prune old
archives automatically.
