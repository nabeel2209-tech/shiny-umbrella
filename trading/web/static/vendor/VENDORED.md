# Vendored front-end code

The dashboard loads nothing from a CDN (the Content-Security-Policy only allows this
origin), so third-party scripts are copied here.

| File | Package | Version | Source | Integrity |
|------|---------|---------|--------|-----------|
| `htmx.min.js` | htmx | 2.0.4 | `https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js` | sha384-HGfztofotfshcF7+8n44JQL2oJmowVChPTg48S+jvZoztPfvwD79OC/LTtG6dMp+ (matches the project's published SRI; also checked against jsDelivr) |

To upgrade: download the new `dist/htmx.min.js`, check its sha384 against the SRI
hash htmx publishes for that release, replace the file, update this table, and run
`make check` plus the manual walkthrough in the README.

```bash
openssl dgst -sha384 -binary htmx.min.js | openssl base64 -A
```
