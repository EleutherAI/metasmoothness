# messages/

Asynchronous channel between the node agents sharing this filesystem.

`NODES.md` covers the *claim* protocol (who runs which row). This directory covers
everything that is not a row claim: environment changes, shared-asset changes,
cross-node gotchas, and questions for Lucia.

## Conventions

- One file per topic: `YYYY-MM-DD-<slug>.md`.
- Put your hostname and the date at the top of anything you write.
- Commit the message — like claims, a message exists only once committed.
- Read this directory at the start of a session and before claiming rows.
- If a message needs a reply, append a `## Reply (<hostname>, <date>)` section to
  the same file rather than creating a new one.
- Unresolved items for Lucia go under a `## For Lucia` heading so they are greppable.
